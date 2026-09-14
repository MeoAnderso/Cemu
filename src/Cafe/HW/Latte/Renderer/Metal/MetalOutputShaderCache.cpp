#include "Cafe/HW/Latte/Renderer/Metal/MetalOutputShaderCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/RendererShaderMtl.h"

MetalOutputShaderCache::~MetalOutputShaderCache()
{
    for (uint8 i = 0; i < METAL_OUTPUT_SHADER_CACHE_SIZE; i++)
    {
        if (m_cache[i])
            m_cache[i]->release();
    }
}

MTL::RenderPipelineState* MetalOutputShaderCache::GetPipeline(RendererOutputShader* shader, uint8 shaderIndex, bool usesSRGB)
{
    uint8 cacheIndex = (usesSRGB ? METAL_SHADER_TYPE_COUNT : 0) + shaderIndex;
    auto& renderPipelineState = m_cache[cacheIndex];
    if (renderPipelineState)
        return renderPipelineState;

    // Create a new render pipeline state
    auto vertexShaderMtl = static_cast<RendererShaderMtl*>(shader->GetVertexShader())->GetFunction();
    auto fragmentShaderMtl = static_cast<RendererShaderMtl*>(shader->GetFragmentShader())->GetFunction();

    NS_STACK_SCOPED auto renderPipelineDescriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    renderPipelineDescriptor->setVertexFunction(vertexShaderMtl);
    renderPipelineDescriptor->setFragmentFunction(fragmentShaderMtl);
    renderPipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(usesSRGB ? MTL::PixelFormatBGRA8Unorm_sRGB : MTL::PixelFormatBGRA8Unorm);

    NS::Error* error = nullptr;
    MTL::RenderPipelineState* newPipelineState = m_mtlr->GetDevice()->newRenderPipelineState(renderPipelineDescriptor, &error);
    if (error)
    {
        cemuLog_log(LogType::Force, "error creating output render pipeline state: {}", error->localizedDescription()->utf8String());
    }
    // only cache successful states - a nil result would be served forever and later crash
    // setRenderPipelineState(nullptr) in DrawBackbufferQuad (transient failures must retry)
    if (!newPipelineState)
        return nullptr;

    renderPipelineState = newPipelineState;
    return renderPipelineState;
}
