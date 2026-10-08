#include "Cafe/HW/Latte/Renderer/Metal/MetalQuery.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"

bool LatteQueryObjectMtl::getResult(uint64& numSamplesPassed)
{
    // not ready until the query was ended and the command buffer holding its draws has completed
    // (Vulkan reports the same way)
    if (!m_ended)
        return false;
    if (m_commandBuffer && !CommandBufferCompleted(m_commandBuffer))
        return false;

    uint64* resultPtr = m_mtlr->GetOcclusionQueryResultsPtr();

    numSamplesPassed = 0;
    if (m_slotsConsumed == 0)
        return true; // no draws executed inside the query - zero samples passed
    if (m_slotsConsumed > MetalRenderer::OCCLUSION_QUERY_POOL_SIZE)
    {
        // More counted draws than the pool has slots. The draws after the first wrap overwrote what
        // the earlier ones filled, so this sum counts them twice and the true value cannot be
        // recovered. Reported rather than returned silently; an inflated count errs towards drawing
        // something the game might have culled, which is the safer direction. Key 0: a query object
        // is no stable identity (host pointer), so the detail line is logged once per session
        MetalDiag_CountOncePer(MetalDiagEvent::OcclusionQueryPoolWrapped, 0,
            "query consumed {} slots (the pool holds {}): the reported count is inflated",
            m_slotsConsumed, (uint32)MetalRenderer::OCCLUSION_QUERY_POOL_SIZE);
    }
    for (uint32 i = m_range.begin; i != m_range.end; i = (i + 1) % MetalRenderer::OCCLUSION_QUERY_POOL_SIZE)
        numSamplesPassed += resultPtr[i];

    return true;
}

LatteQueryObjectMtl::~LatteQueryObjectMtl()
{
    if (m_commandBuffer)
        m_commandBuffer->release();
}

void LatteQueryObjectMtl::begin()
{
    m_range.begin = m_mtlr->GetOcclusionQueryIndex();
    m_drawCountAtBegin = m_mtlr->GetOcclusionQueryDrawCount();
    m_slotsConsumed = 0;
    m_ended = false;
    m_mtlr->BeginOcclusionQuery();
}

void LatteQueryObjectMtl::end()
{
    m_range.end = m_mtlr->GetOcclusionQueryIndex();
    m_mtlr->EndOcclusionQuery();

    // count from the renderer's monotonic draw counter rather than from begin/end: a query that
    // consumed exactly the pool size has begin == end
    const uint64 consumed = m_mtlr->GetOcclusionQueryDrawCount() - m_drawCountAtBegin;
    m_slotsConsumed = (consumed > UINT32_MAX) ? UINT32_MAX : (uint32)consumed;
    m_ended = true;

    m_commandBuffer = m_mtlr->GetAndRetainCurrentCommandBufferIfNotCompleted();
    if (m_commandBuffer)
        m_mtlr->RequestSoonCommit();
}
