#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"
#include "Metal/MTLTexture.hpp"
#include <atomic>

LatteTextureViewMtl::LatteTextureViewMtl(MetalRenderer* mtlRenderer, LatteTextureMtl* texture, Latte::E_DIM dim, Latte::E_GX2SURFFMT format, sint32 firstMip, sint32 mipCount, sint32 firstSlice, sint32 sliceCount)
	: LatteTextureView(texture, firstMip, mipCount, firstSlice, sliceCount, dim, format), m_mtlr(mtlRenderer), m_baseTexture(texture)
{
    m_rgbaView = CreatePlainView();
}

LatteTextureViewMtl::~LatteTextureViewMtl()
{
    m_rgbaView->release();
	for (sint32 i = 0; i < std::size(m_viewCache); i++)
    {
        if (m_viewCache[i].key != INVALID_SWIZZLE)
            m_viewCache[i].texture->release();
    }

    for (auto& [key, texture] : m_fallbackViewCache)
    {
        texture->release();
    }

    for (auto& [key, texture] : m_mipClampViewCache)
    {
        texture->release();
    }
}

MTL::Texture* LatteTextureViewMtl::GetSwizzledView(uint32 gpuSamplerSwizzle)
{
    // Mask out
    gpuSamplerSwizzle &= 0x0FFF0000;

    // First, try to find a view in the cache

    // Fast cache
    sint32 freeIndex = -1;
    for (sint32 i = 0; i < std::size(m_viewCache); i++)
    {
        const auto& entry = m_viewCache[i];
        if (entry.key == gpuSamplerSwizzle)
        {
            return entry.texture;
        }
        else if (entry.key == INVALID_SWIZZLE && freeIndex == -1)
        {
            freeIndex = i;
        }
    }

    // Fallback cache
    auto& fallbackEntry = m_fallbackViewCache[gpuSamplerSwizzle];
    if (fallbackEntry)
    {
        return fallbackEntry;
    }

    MTL::Texture* texture = CreateSwizzledView(gpuSamplerSwizzle);
    if (freeIndex != -1)
        m_viewCache[freeIndex] = {gpuSamplerSwizzle, texture};
    else
        fallbackEntry = texture;

    return texture;
}

MTL::Texture* LatteTextureViewMtl::CreatePlainView()
{
	return CreateViewInternal(nullptr);
}

MTL::TextureSwizzleChannels LatteTextureViewMtl::GetSwizzleChannels(Latte::E_GX2SURFFMT format, uint32 gpuSamplerSwizzle)
{
	uint32 compSelR = (gpuSamplerSwizzle >> 16) & 0x7;
	uint32 compSelG = (gpuSamplerSwizzle >> 19) & 0x7;
	uint32 compSelB = (gpuSamplerSwizzle >> 22) & 0x7;
	uint32 compSelA = (gpuSamplerSwizzle >> 25) & 0x7;
	compSelR = LatteTextureView_AdjustTextureCompSel(format, compSelR);
	compSelG = LatteTextureView_AdjustTextureCompSel(format, compSelG);
	compSelB = LatteTextureView_AdjustTextureCompSel(format, compSelB);
	compSelA = LatteTextureView_AdjustTextureCompSel(format, compSelA);

	MTL::TextureSwizzleChannels swizzle;
	swizzle.red = GetMtlTextureSwizzle(compSelR);
	swizzle.green = GetMtlTextureSwizzle(compSelG);
	swizzle.blue = GetMtlTextureSwizzle(compSelB);
	swizzle.alpha = GetMtlTextureSwizzle(compSelA);
	return swizzle;
}

MTL::Texture* LatteTextureViewMtl::CreateSwizzledView(uint32 gpuSamplerSwizzle)
{
	MTL::TextureSwizzleChannels swizzle = GetSwizzleChannels(format, gpuSamplerSwizzle);
	return CreateViewInternal(&swizzle);
}

MTL::Texture* LatteTextureViewMtl::GetSwizzledViewWithMipCount(uint32 gpuSamplerSwizzle, uint32 levelCount)
{
    // same swizzle masking as GetSwizzledView so both caches key on the identical value
    const uint64 key = (uint64)(gpuSamplerSwizzle & 0x0FFF0000) | ((uint64)levelCount << 32);
    auto itr = m_mipClampViewCache.find(key);
    if (itr != m_mipClampViewCache.end())
        return itr->second;
    MTL::Texture* view = CreateSwizzledViewWithMipCount(gpuSamplerSwizzle, levelCount);
    m_mipClampViewCache.emplace(key, view);
    return view;
}

MTL::Texture* LatteTextureViewMtl::CreateSwizzledViewWithMipCount(uint32 gpuSamplerSwizzle, uint32 levelCount)
{
	MTL::TextureSwizzleChannels swizzle = GetSwizzleChannels(format, gpuSamplerSwizzle);
	return CreateViewInternal(&swizzle, (sint32)levelCount);
}

MTL::Texture* LatteTextureViewMtl::CreateViewInternal(const MTL::TextureSwizzleChannels* swizzle, sint32 overrideLevelCount)
{
	MTL::TextureType textureType;
    switch (dim)
    {
    case Latte::E_DIM::DIM_1D:
        textureType = MTL::TextureType1D;
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
        cemu_assert_debug(this->numSlice % 6 == 0 && "cubemaps must have an array length multiple of 6");

        textureType = MTL::TextureTypeCubeArray;
        break;
    default:
        cemu_assert_unimplemented();
        textureType = MTL::TextureType2D;
        break;
    }

    uint32 baseLevel = firstMip;
    uint32 levelCount = this->numMip;
    if (overrideLevelCount > 0)
        levelCount = std::min<uint32>((uint32)overrideLevelCount, this->numMip);
    uint32 baseLayer = 0;
    uint32 layerCount = 1;
    // TODO: check if base texture is 3D texture as well
    if (textureType == MTL::TextureType3D)
    {
        cemu_assert_debug(firstMip == 0);
        cemu_assert_debug(this->numSlice == baseTexture->depth);
    }
    else
    {
        baseLayer = firstSlice;
        if (textureType == MTL::TextureTypeCubeArray || textureType == MTL::TextureType2DArray)
            layerCount = this->numSlice;
    }

    // Clamp mip levels. Done in signed arithmetic so a baseLevel past the end of the chain cannot
    // wrap the subtraction into a huge levelCount and ask Metal for mip levels the texture does not
    // have. firstMip < mipLevels <= maxPossibleMipLevels holds today (LatteTextureMtl::CreateView
    // asserts the first half in debug builds) - this only keeps the failure benign if that changes
    const sint32 availableLevels = std::max<sint32>(m_baseTexture->maxPossibleMipLevels - (sint32)baseLevel, 0);
    levelCount = std::min<uint32>(levelCount, (uint32)availableLevels);
    levelCount = std::max(levelCount, (uint32)1);

    auto pixelFormat = GetMtlPixelFormat(format, m_baseTexture->isDepth);

    // Swizzled views are restricted to ShaderRead|PixelFormatView usage, which prevents their use as
    // render pass attachments. Plain views inherit the full usage of the parent texture.
    MTL::Texture* view = nullptr;
    if (swizzle)
        view = m_baseTexture->GetTexture()->newTextureView(pixelFormat, textureType, NS::Range::Make(baseLevel, levelCount), NS::Range::Make(baseLayer, layerCount), *swizzle);
    else
        view = m_baseTexture->GetTexture()->newTextureView(pixelFormat, textureType, NS::Range::Make(baseLevel, levelCount), NS::Range::Make(baseLayer, layerCount));

    if (!view)
    {
        // the view pixel format is not compatible with the base texture (Metal's view rules are
        // stricter than Vulkan's VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, which allows e.g.
        // RGBA32Uint <-> BC3 reinterprets). Fall back to a view with the base texture's own
        // pixel format so the sampling keeps working (components will differ from the game's
        // reinterpret) instead of caching and binding a nil texture
        static std::atomic<uint32> s_viewFallbackLogCount{ 0 };
        if (s_viewFallbackLogCount.fetch_add(1) < 8)
            cemuLog_log(LogType::Force, "LatteTextureViewMtl: view format {:04x} not compatible with base texture format {:04x} ({:08x}), falling back to the base format",
                (uint32)format, (uint32)m_baseTexture->format, baseTexture->physAddress);
        if (swizzle)
            view = m_baseTexture->GetTexture()->newTextureView(m_baseTexture->GetTexture()->pixelFormat(), textureType, NS::Range::Make(baseLevel, levelCount), NS::Range::Make(baseLayer, layerCount), *swizzle);
        else
            view = m_baseTexture->GetTexture()->newTextureView(m_baseTexture->GetTexture()->pixelFormat(), textureType, NS::Range::Make(baseLevel, levelCount), NS::Range::Make(baseLayer, layerCount));
    }
    return view;
}
