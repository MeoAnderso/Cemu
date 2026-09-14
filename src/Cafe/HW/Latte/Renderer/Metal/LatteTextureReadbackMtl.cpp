#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureReadbackMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"

LatteTextureReadbackInfoMtl::~LatteTextureReadbackInfoMtl()
{
    if (m_commandBuffer)
        m_commandBuffer->release();
}

void LatteTextureReadbackInfoMtl::StartTransfer()
{
	cemu_assert(m_textureView);

	auto* baseTexture = (LatteTextureMtl*)m_textureView->baseTexture;

	cemu_assert_debug(m_textureView->firstMip == 0);
	cemu_assert_debug(m_textureView->baseTexture->dim != Latte::E_DIM::DIM_3D);

	// row pitch / image size are taken from the Metal texture's own dimensions rather than from
	// GetReadbackRowPitch()/GetReadbackImageSize() (which walk the logical baseTexture at
	// m_textureView->firstMip): under a graphic-pack resolution overwrite the two disagree, and
	// Metal validates destinationBytesPerRow against the *source texture's* pixel format, so the
	// value has to come from the MTL resource. The helpers cover compressed block sizes.
	// sourceLevel stays 0 - the assert above documents that Metal readback is mip0-only here
	MTL::Texture* mtlTexture = baseTexture->GetTexture();
	const uint32 copyWidth = (uint32)mtlTexture->width();
	const uint32 copyHeight = (uint32)mtlTexture->height();
	m_rowPitch = (uint32)GetMtlTextureBytesPerRow(baseTexture->format, baseTexture->isDepth, copyWidth);
	m_image_size = (uint32)GetMtlTextureBytesPerImage(baseTexture->format, baseTexture->isDepth, copyHeight, m_rowPitch);

	auto blitCommandEncoder = m_mtlr->GetBlitCommandEncoder();

	// combined depth/stencil formats require the aspect blit option - without it Metal rejects
	// the copy and the readback buffer keeps its previous content (stale/garbage data)
	MTL::BlitOption blitOption = MTL::BlitOptionNone;
	const auto& formatInfo = GetMtlPixelFormatInfo(baseTexture->format, baseTexture->isDepth);
	if (baseTexture->isDepth && formatInfo.hasStencil)
		blitOption = MTL::BlitOptionDepthFromDepthStencil;

	blitCommandEncoder->copyFromTexture(mtlTexture, m_firstSlice, 0, MTL::Origin{0, 0, 0}, MTL::Size{copyWidth, copyHeight, 1}, m_mtlr->GetTextureReadbackBuffer(), m_bufferOffset, m_rowPitch, m_image_size, blitOption);

	m_commandBuffer = m_mtlr->GetCurrentCommandBuffer()->retain();
	// TODO: uncomment?
	//m_mtlr->RequestSoonCommit();
	m_mtlr->CommitCommandBuffer();
}

bool LatteTextureReadbackInfoMtl::IsFinished()
{
    // Command buffer wasn't even comitted, let's commit immediately
    //if (m_mtlr->GetCurrentCommandBuffer() == m_commandBuffer)
    //    m_mtlr->CommitCommandBuffer();

    return CommandBufferCompleted(m_commandBuffer);
}

void LatteTextureReadbackInfoMtl::ForceFinish()
{
    m_commandBuffer->waitUntilCompleted();
}

uint8* LatteTextureReadbackInfoMtl::GetData()
{
	return (uint8*)m_mtlr->GetTextureReadbackBuffer()->contents() + m_bufferOffset;
}
