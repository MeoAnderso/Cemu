#include "Cafe/HW/Latte/Renderer/Metal/MetalAttachmentsInfo.h"
#include "Cafe/HW/Latte/Renderer/Metal/CachedFBOMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"

// Attachment-less FBOs get a dummy RGBA8Unorm color attachment in the render pass descriptor
// (CachedFBOMtl, to keep streamout draws alive). The pipeline's declared pixel formats must match
// the render pass, so every MetalAttachmentsInfo that describes such a pass has to report the dummy.
// Applying this in only one of the two constructors made the pipeline-cache loader hash a different
// attachment set than the live path, so restored attachment-less pipelines were filed under a key
// the live path never asks for
static void ApplyDummyColorAttachmentIfEmpty(MetalAttachmentsInfo& info)
{
	bool hasAnyAttachment = info.depthFormat != Latte::E_GX2SURFFMT::INVALID_FORMAT;
	for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET && !hasAnyAttachment; i++)
		hasAnyAttachment = info.colorFormats[i] != Latte::E_GX2SURFFMT::INVALID_FORMAT;

	if (!hasAnyAttachment)
		info.colorFormats[0] = Latte::E_GX2SURFFMT::R8_G8_B8_A8_UNORM;
}

MetalAttachmentsInfo::MetalAttachmentsInfo(class CachedFBOMtl* fbo)
{
    for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
	{
	    const auto& colorBuffer = fbo->colorBuffer[i];
		auto texture = static_cast<LatteTextureViewMtl*>(colorBuffer.texture);
		if (!texture)
		    continue;

		colorFormats[i] = texture->format;
	}

	// Depth stencil attachment
	if (fbo->depthBuffer.texture)
	{
	    auto texture = static_cast<LatteTextureViewMtl*>(fbo->depthBuffer.texture);
        depthFormat = texture->format;
        hasStencil = fbo->depthBuffer.hasStencil;
	}

	ApplyDummyColorAttachmentIfEmpty(*this);
}

// Register-based variant used by the serialized pipeline cache loader, which has no FBO object
MetalAttachmentsInfo::MetalAttachmentsInfo(const LatteContextRegister& lcr, const LatteDecompilerShader* pixelShader)
{
    uint8 cbMask = LatteMRT::GetActiveColorBufferMask(pixelShader, lcr);
	bool dbMask = LatteMRT::GetActiveDepthBufferMask(lcr);

	// Color attachments
	for (int i = 0; i < 8; ++i)
	{
		if ((cbMask & (1 << i)) == 0)
			continue;

		colorFormats[i] = LatteMRT::GetColorBufferFormat(i, lcr);
	}

	// Depth stencil attachment
	if (dbMask)
	{
		Latte::E_GX2SURFFMT format = LatteMRT::GetDepthBufferFormat(lcr);
		depthFormat = format;
		hasStencil = GetMtlPixelFormatInfo(format, true).hasStencil;
	}

	ApplyDummyColorAttachmentIfEmpty(*this);
}
