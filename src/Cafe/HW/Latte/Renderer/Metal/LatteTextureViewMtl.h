#pragma once

#include <Metal/Metal.hpp>
#include <unordered_map>

#include "Cafe/HW/Latte/Core/LatteTexture.h"

#define INVALID_SWIZZLE 0xFFFFFFFF

// Cache key for renderer-side per-texture sample views derived from a bound LatteTextureView
// (depth-mirror sample views in LatteTextureMtl, feedback-shadow sample views in MetalRenderer).
// Every field that alters the produced MTL view is part of the key, so cached views are only
// reused for byte-identical view specifications
struct LatteMtlSampleViewKey
{
	uint32 swizzle;    // masked comp-sel word (0x0FFF0000)
	uint16 format;     // Latte::E_GX2SURFFMT of the bound view
	uint8  dim;        // Latte::E_DIM of the bound view (selects the MTL view type)
	uint8  firstMip;
	uint8  numMip;
	uint16 firstSlice;
	uint16 numSlice;

	bool operator==(const LatteMtlSampleViewKey&) const = default;
};

namespace std
{
template<> struct hash<LatteMtlSampleViewKey>
{
	size_t operator()(const LatteMtlSampleViewKey& k) const
	{
		// The fields need 79 bits in total (swizzle 12, format 16, dim 3, mips 8+8, slices 16+16),
		// so no assignment of disjoint bit ranges within a 64-bit value exists. An earlier version
		// shifted the fields into overlapping ranges and pushed numSlice off the top of size_t.
		// Fold each field in whole instead. Equality is still decided by the defaulted operator==,
		// so this only has to spread the buckets, not be injective
		size_t h = (size_t)k.swizzle;
		for (const size_t field : { (size_t)k.format, (size_t)k.dim, (size_t)k.firstMip,
			(size_t)k.numMip, (size_t)k.firstSlice, (size_t)k.numSlice })
		{
			h ^= field + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
		}
		return h;
	}
};
}

class LatteTextureViewMtl : public LatteTextureView
{
public:
	LatteTextureViewMtl(class MetalRenderer* mtlRenderer, class LatteTextureMtl* texture, Latte::E_DIM dim, Latte::E_GX2SURFFMT format, sint32 firstMip, sint32 mipCount, sint32 firstSlice, sint32 sliceCount);
	~LatteTextureViewMtl();

    MTL::Texture* GetSwizzledView(uint32 gpuSamplerSwizzle);

    // Cached swizzled view with a reduced mip count (the cache owns the returned texture; do not
    // release). Used by BindStageResources' never-managed-chain clamp: emulates Vulkan's shorter
    // incarnations so forced-LOD samples fall onto written content. The clamp runs per draw, so
    // the views are cached rather than created and released each time
    MTL::Texture* GetSwizzledViewWithMipCount(uint32 gpuSamplerSwizzle, uint32 levelCount);

    MTL::Texture* GetRGBAView()
    {
        return m_rgbaView;
    }

    // Builds the Metal swizzle channels for a texture CompSel word (as used by GetSwizzledView),
    // also usable for temporary views of unrelated textures (e.g. the depth color copy)
    static MTL::TextureSwizzleChannels GetSwizzleChannels(Latte::E_GX2SURFFMT format, uint32 gpuSamplerSwizzle);

private:
	class MetalRenderer* m_mtlr;

	class LatteTextureMtl* m_baseTexture;

	MTL::Texture* m_rgbaView;
	struct {
	    uint32 key;
	    MTL::Texture* texture;
	} m_viewCache[4] = {{INVALID_SWIZZLE, nullptr}, {INVALID_SWIZZLE, nullptr}, {INVALID_SWIZZLE, nullptr}, {INVALID_SWIZZLE, nullptr}};
	std::unordered_map<uint32, MTL::Texture*> m_fallbackViewCache;
	// owned clamp views keyed by (masked swizzle, levelCount) - see GetSwizzledViewWithMipCount
	std::unordered_map<uint64, MTL::Texture*> m_mipClampViewCache;

    MTL::Texture* CreateSwizzledView(uint32 gpuSamplerSwizzle);
    MTL::Texture* CreateSwizzledViewWithMipCount(uint32 gpuSamplerSwizzle, uint32 levelCount);
    MTL::Texture* CreatePlainView();
    MTL::Texture* CreateViewInternal(const MTL::TextureSwizzleChannels* swizzle, sint32 overrideLevelCount = -1);
};
