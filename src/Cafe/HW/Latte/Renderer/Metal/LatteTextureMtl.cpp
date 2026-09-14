#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"

LatteTextureMtl::LatteTextureMtl(class MetalRenderer* mtlRenderer, Latte::E_DIM dim, MPTR physAddress, MPTR physMipAddress, Latte::E_GX2SURFFMT format, uint32 width, uint32 height, uint32 depth, uint32 pitch, uint32 mipLevels, uint32 swizzle,
	Latte::E_HWTILEMODE tileMode, bool isDepth)
	: LatteTexture(dim, physAddress, physMipAddress, format, width, height, depth, pitch, mipLevels, swizzle, tileMode, isDepth), m_mtlr(mtlRenderer)
{
    NS_STACK_SCOPED MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setStorageMode(MTL::StorageModePrivate);
    //desc->setCpuCacheMode(MTL::CPUCacheModeWriteCombined);

	sint32 effectiveBaseWidth = width;
	sint32 effectiveBaseHeight = height;
	sint32 effectiveBaseDepth = depth;
	if (overwriteInfo.hasResolutionOverwrite)
	{
		effectiveBaseWidth = overwriteInfo.width;
		effectiveBaseHeight = overwriteInfo.height;
		effectiveBaseDepth = overwriteInfo.depth;
	}
	effectiveBaseWidth = std::max(1, effectiveBaseWidth);
	effectiveBaseHeight = std::max(1, effectiveBaseHeight);
	effectiveBaseDepth = std::max(1, effectiveBaseDepth);

	MTL::TextureType textureType;
	switch (dim)
    {
    case Latte::E_DIM::DIM_1D:
        textureType = MTL::TextureType1D;
        effectiveBaseHeight = 1;
        break;
    case Latte::E_DIM::DIM_2D:
    case Latte::E_DIM::DIM_2D_MSAA:
        textureType = MTL::TextureType2D;
        break;
    case Latte::E_DIM::DIM_2D_ARRAY:
        textureType = MTL::TextureType2DArray;
        break;
    case Latte::E_DIM::DIM_3D:
        textureType = MTL::TextureType3D;
        break;
    case Latte::E_DIM::DIM_CUBEMAP:
        cemu_assert_debug(effectiveBaseDepth % 6 == 0 && "cubemaps must have an array length multiple of 6");

        textureType = MTL::TextureTypeCubeArray;
        break;
    default:
        cemu_assert_unimplemented();
        textureType = MTL::TextureType2D;
        break;
    }
    desc->setTextureType(textureType);

    // Clamp mip levels
    mipLevels = std::min(mipLevels, (uint32)maxPossibleMipLevels);
    mipLevels = std::max(mipLevels, (uint32)1);

	desc->setWidth(effectiveBaseWidth);
	desc->setHeight(effectiveBaseHeight);
	desc->setMipmapLevelCount(mipLevels);

	if (textureType == MTL::TextureType3D)
	{
		desc->setDepth(effectiveBaseDepth);
	}
	else if (textureType == MTL::TextureTypeCubeArray)
	{
		desc->setArrayLength(effectiveBaseDepth / 6);
	}
	else if (textureType == MTL::TextureType2DArray)
	{
		desc->setArrayLength(effectiveBaseDepth);
	}

	auto pixelFormat = GetMtlPixelFormat(format, isDepth);
	m_isAlternateFormat = GetMtlPixelFormatInfo(format, isDepth).isAlternateFormat;
	desc->setPixelFormat(pixelFormat);

	MTL::TextureUsage usage = MTL::TextureUsageShaderRead | MTL::TextureUsagePixelFormatView;
	if (FormatIsRenderable(format))
		usage |= MTL::TextureUsageRenderTarget;
	desc->setUsage(usage);

	m_texture = mtlRenderer->GetDevice()->newTexture(desc);

	// Label with the game-side identity so GPU traces are searchable: the label carries the guest
	// physical address, which identifies every draw touching this texture
	m_texture->setLabel(ToNSString(fmt::format("latte {:07x} {}x{} m{}{}",
		physAddress, width, height, mipLevels, isDepth ? " depth" : "")));
}

MTL::Texture* LatteTextureMtl::GetDepthMirrorSampleView(LatteTextureView* textureView, uint32 gpuSamplerSwizzle)
{
	MTL::Texture* colorCopy = GetDepthColorCopy();
	if (!colorCopy)
		return nullptr;
	const LatteMtlSampleViewKey key = {
		.swizzle = gpuSamplerSwizzle & 0x0FFF0000,
		.format = (uint16)textureView->format,
		.dim = (uint8)textureView->dim,
		.firstMip = (uint8)textureView->firstMip,
		.numMip = (uint8)textureView->numMip,
		.firstSlice = (uint16)textureView->firstSlice,
		.numSlice = (uint16)textureView->numSlice,
	};
	auto itr = m_depthMirrorSampleViews.find(key);
	if (itr != m_depthMirrorSampleViews.end())
		return itr->second;

	// sample the copy with the same component swizzle the game requested (Vulkan samples a
	// swizzled depth view) and over the full mip chain. The view type and slice range match the
	// bound view so 2D-array shadow maps sample the correct cascade layer
	const auto swizzle = LatteTextureViewMtl::GetSwizzleChannels(format, gpuSamplerSwizzle);
	const bool mirrorIsArray = (textureView->dim == Latte::E_DIM::DIM_2D_ARRAY);
	const MTL::TextureType sampleType = mirrorIsArray ? MTL::TextureType2DArray : MTL::TextureType2D;
	NS::Range sliceRange = mirrorIsArray ? NS::Range::Make(textureView->firstSlice, textureView->numSlice) : NS::Range::Make(textureView->firstSlice, 1);
	// mirror the game view's mip selection (like MetalRenderer::CreateFeedbackShadowView does):
	// Vulkan samples the depth view with firstMip/numMip, so LOD clamping must behave identically.
	// A full-chain view here would let mip-filtered samples (e.g. a mip=linear sampler minifying
	// into a quarter-res DoF buffer) pick generated mips that the game never sees on Vulkan -
	// divergent depth content in the DoF path
	const uint32 mirrorMipCount = (uint32)colorCopy->mipmapLevelCount();
	const uint32 baseMip = std::min<uint32>((uint32)textureView->firstMip, mirrorMipCount - 1);
	const uint32 mipCount = std::min<uint32>(std::max<uint32>((uint32)textureView->numMip, 1), mirrorMipCount - baseMip);
	MTL::Texture* view = colorCopy->newTextureView(colorCopy->pixelFormat(), sampleType, NS::Range::Make(baseMip, mipCount), sliceRange, swizzle);
	m_depthMirrorSampleViews.emplace(key, view);
	return view;
}

LatteTextureMtl::~LatteTextureMtl()
{
	m_texture->release();
	// release through the setter so the cached sample views are dropped too
	SetDepthColorCopy(nullptr, 0);

	// drop any feedback-loop shadow copy keyed by this texture (see MetalRenderer::
	// PrepareFeedbackLoopShadowCopies): without this the map retains released views and grows
	// with every destroyed feedback texture
	auto& shadowCopies = m_mtlr->m_feedbackShadowCopies;
	auto shadowItr = shadowCopies.find(this);
	if (shadowItr != shadowCopies.end())
	{
		if (shadowItr->second.texture)
			shadowItr->second.texture->release();
		shadowCopies.erase(shadowItr);
	}
	m_mtlr->m_feedbackShadowTextures.erase(this);
}

LatteTextureView* LatteTextureMtl::CreateView(Latte::E_DIM dim, Latte::E_GX2SURFFMT format, sint32 firstMip, sint32 mipCount, sint32 firstSlice, sint32 sliceCount)
{
	cemu_assert_debug(mipCount > 0);
	cemu_assert_debug(sliceCount > 0);
	cemu_assert_debug((firstMip + mipCount) <= this->mipLevels);
	cemu_assert_debug((firstSlice + sliceCount) <= this->depth);

	return new LatteTextureViewMtl(m_mtlr, this, dim, format, firstMip, mipCount, firstSlice, sliceCount);
}

// TODO: lazy allocation?
void LatteTextureMtl::AllocateOnHost()
{
	// The texture is already allocated
}
