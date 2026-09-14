#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"
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

	// A 3D texture is not served correctly by the blit below: it transfers a single depth slice (the
	// copy's depth is 1) while the readback is expected to carry the texture's content, so a volume's
	// remaining slices are never read and the consumer writes one slice as if it were the whole
	// texture. Vulkan carries the same restriction (cemu_assert_debug in TextureReadbackVk.cpp), so
	// this is parity, not a Metal gap - but it is reported rather than asserted because a
	// compiled-out assert is exactly how a readback ends up quietly holding the wrong data
	if (baseTexture->dim == Latte::E_DIM::DIM_3D)
	{
		MetalDiag_CountOncePer(MetalDiagEvent::ReadbackShapeUnsupported, baseTexture->physAddress,
			"{:016x} is a 3D texture - Metal readback serves a single depth slice, so the game reads back the wrong volume",
			baseTexture->physAddress);
	}

	// row pitch / image size are taken from the Metal texture's own dimensions rather than from
	// GetReadbackRowPitch()/GetReadbackImageSize() (which walk the logical baseTexture at
	// m_textureView->firstMip): under a graphic-pack resolution overwrite the two disagree, and
	// Metal validates destinationBytesPerRow against the *source texture's* pixel format, so the
	// value has to come from the MTL resource. The helpers cover compressed block sizes and take
	// pixel dimensions, so the mip shift happens here
	MTL::Texture* mtlTexture = baseTexture->GetTexture();
	const sint32 mtlMipCount = (sint32)mtlTexture->mipmapLevelCount();
	uint32 mipLevel = (uint32)m_firstMip;
	if ((sint32)mipLevel >= mtlMipCount)
	{
		// Defensive. LatteTextureMtl clamps mipLevels to what the dimensions allow, so a view could in
		// principle name a level the resource does not have; today the view invariants keep
		// firstMip < mipLevels <= maxPossibleMipLevels (see LatteTextureViewMtl), which makes this
		// unreachable. Clamp into the chain and say so rather than issue an invalid blit - the copy
		// stays valid, and nothing silently reads a level that was never transferred
		MetalDiag_CountOncePer(MetalDiagEvent::ReadbackShapeUnsupported, baseTexture->physAddress,
			"{:016x} readback asked for mip {} but the Metal texture has {} levels - serving level {}",
			baseTexture->physAddress, (uint32)m_firstMip, (uint32)mtlMipCount, (uint32)(mtlMipCount - 1));
		mipLevel = (uint32)(mtlMipCount - 1);
	}
	const uint32 copyWidth = std::max<uint32>(1, (uint32)mtlTexture->width() >> mipLevel);
	const uint32 copyHeight = std::max<uint32>(1, (uint32)mtlTexture->height() >> mipLevel);
	m_rowPitch = (uint32)GetMtlTextureBytesPerRow(baseTexture->format, baseTexture->isDepth, copyWidth);
	m_image_size = (uint32)GetMtlTextureBytesPerImage(baseTexture->format, baseTexture->isDepth, copyHeight, m_rowPitch);

	auto blitCommandEncoder = m_mtlr->GetBlitCommandEncoder();

	// combined depth/stencil formats require the aspect blit option - without it Metal rejects
	// the copy and the readback buffer keeps its previous content (stale/garbage data)
	MTL::BlitOption blitOption = MTL::BlitOptionNone;
	const auto& formatInfo = GetMtlPixelFormatInfo(baseTexture->format, baseTexture->isDepth);
	if (baseTexture->isDepth && formatInfo.hasStencil)
		blitOption = MTL::BlitOptionDepthFromDepthStencil;

	blitCommandEncoder->copyFromTexture(mtlTexture, m_firstSlice, mipLevel, MTL::Origin{0, 0, 0}, MTL::Size{copyWidth, copyHeight, 1}, m_mtlr->GetTextureReadbackBuffer(), m_bufferOffset, m_rowPitch, m_image_size, blitOption);

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
