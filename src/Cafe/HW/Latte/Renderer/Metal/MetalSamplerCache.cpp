#include "Cafe/HW/Latte/Renderer/Metal/MetalSamplerCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"


MTL::SamplerBorderColor GetBorderColor(LatteConst::ShaderType shaderType, uint32 stageSamplerIndex, const _LatteRegisterSetSampler* samplerWords, bool logWorkaround = false)
{
    auto borderType = samplerWords->WORD0.get_BORDER_COLOR_TYPE();

    MTL::SamplerBorderColor borderColor;
    if (borderType == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_BORDER_COLOR_TYPE::TRANSPARENT_BLACK)
        borderColor = MTL::SamplerBorderColorTransparentBlack;
    else if (borderType == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_BORDER_COLOR_TYPE::OPAQUE_BLACK)
        borderColor = MTL::SamplerBorderColorOpaqueBlack;
    else if (borderType == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_BORDER_COLOR_TYPE::OPAQUE_WHITE)
        borderColor = MTL::SamplerBorderColorOpaqueWhite;
    else [[unlikely]]
    {
        _LatteRegisterSetSamplerBorderColor* borderColorReg;
		if (shaderType == LatteConst::ShaderType::Vertex)
			borderColorReg = LatteGPUState.contextNew.TD_VS_SAMPLER_BORDER_COLOR + stageSamplerIndex;
		else if (shaderType == LatteConst::ShaderType::Pixel)
			borderColorReg = LatteGPUState.contextNew.TD_PS_SAMPLER_BORDER_COLOR + stageSamplerIndex;
		else // geometry
			borderColorReg = LatteGPUState.contextNew.TD_GS_SAMPLER_BORDER_COLOR + stageSamplerIndex;
		float r = borderColorReg->red.get_channelValue();
		float g = borderColorReg->green.get_channelValue();
		float b = borderColorReg->blue.get_channelValue();
		float a = borderColorReg->alpha.get_channelValue();

		// Metal doesn't support custom border color
		// "White" must test all RGB channels: a tinted border like (1,0,0,1) would otherwise be
		// classified as white on the strength of R alone
		bool opaque = (a == 1.0f);
		bool white = (r == 1.0f && g == 1.0f && b == 1.0f);
		if (opaque)
		{
		    if (white)
                borderColor = MTL::SamplerBorderColorOpaqueWhite;
            else
                borderColor = MTL::SamplerBorderColorOpaqueBlack;
		}
		else
		{
		    borderColor = MTL::SamplerBorderColorTransparentBlack;
		}

		if (logWorkaround)
		{
		    float newR, newG, newB, newA;
			switch (borderColor)
			{
			case MTL::SamplerBorderColorTransparentBlack:
                newR = 0.0f;
                newG = 0.0f;
                newB = 0.0f;
                newA = 0.0f;
                break;
            case MTL::SamplerBorderColorOpaqueBlack:
                newR = 0.0f;
                newG = 0.0f;
                newB = 0.0f;
                newA = 1.0f;
                break;
            case MTL::SamplerBorderColorOpaqueWhite:
                newR = 1.0f;
                newG = 1.0f;
                newB = 1.0f;
                newA = 1.0f;
                break;
            }

            if (r != newR || g != newG || b != newB || a != newA)
                cemuLog_log(LogType::Force, "Custom border color ({}, {}, {}, {}) is not supported on Metal, using ({}, {}, {}, {}) instead", r, g, b, a, newR, newG, newB, newA);
		}
    }

    return borderColor;
}

MetalSamplerCache::~MetalSamplerCache()
{
    for (auto& pair : m_samplerCache)
    {
        pair.second->release();
    }
    m_samplerCache.clear();
}

MTL::SamplerState* MetalSamplerCache::GetSamplerState(const LatteContextRegister& lcr, LatteConst::ShaderType shaderType, uint32 stageSamplerIndex, const _LatteRegisterSetSampler* samplerWords, bool depthCompareMode)
{
    uint64 stateHash = CalculateSamplerHash(lcr, shaderType, stageSamplerIndex, samplerWords, depthCompareMode);
    auto& samplerState = m_samplerCache[stateHash];
    if (samplerState)
        return samplerState;

	// Sampler state


    NS_STACK_SCOPED MTL::SamplerDescriptor* samplerDescriptor = MTL::SamplerDescriptor::alloc()->init();

    // lod
    uint32 iMinLOD = samplerWords->WORD1.get_MIN_LOD();
    uint32 iMaxLOD = samplerWords->WORD1.get_MAX_LOD();

    auto filterMip = samplerWords->WORD0.get_MIP_FILTER();
    if (filterMip == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_Z_FILTER::NONE)
    {
        samplerDescriptor->setMipFilter(MTL::SamplerMipFilterNearest);
        samplerDescriptor->setLodMinClamp(0.0f);
        samplerDescriptor->setLodMaxClamp(0.25f);
    }
    else if (filterMip == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_Z_FILTER::POINT)
    {
        samplerDescriptor->setMipFilter(MTL::SamplerMipFilterNearest);
        samplerDescriptor->setLodMinClamp((float)iMinLOD / 64.0f);
        samplerDescriptor->setLodMaxClamp((float)iMaxLOD / 64.0f);
    }
    else if (filterMip == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_Z_FILTER::LINEAR)
    {
        samplerDescriptor->setMipFilter(MTL::SamplerMipFilterLinear);
        samplerDescriptor->setLodMinClamp((float)iMinLOD / 64.0f);
        samplerDescriptor->setLodMaxClamp((float)iMaxLOD / 64.0f);
    }
    else
    {
        // fallback for invalid constants
        samplerDescriptor->setMipFilter(MTL::SamplerMipFilterLinear);
        samplerDescriptor->setLodMinClamp((float)iMinLOD / 64.0f);
        samplerDescriptor->setLodMaxClamp((float)iMaxLOD / 64.0f);
    }

    auto filterMin = samplerWords->WORD0.get_XY_MIN_FILTER();
    cemu_assert_debug(filterMin != Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_XY_FILTER::BICUBIC); // todo
    samplerDescriptor->setMinFilter((filterMin == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_XY_FILTER::POINT || filterMin == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_XY_FILTER::ANISO_POINT) ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear);

    auto filterMag = samplerWords->WORD0.get_XY_MAG_FILTER();
    samplerDescriptor->setMagFilter((filterMag == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_XY_FILTER::POINT || filterMin == Latte::LATTE_SQ_TEX_SAMPLER_WORD0_0::E_XY_FILTER::ANISO_POINT) ? MTL::SamplerMinMagFilterNearest : MTL::SamplerMinMagFilterLinear);

    auto filterZ = samplerWords->WORD0.get_Z_FILTER();
    // todo: z-filter for texture array samplers is customizable for GPU7 but OpenGL/Vulkan doesn't expose this functionality?

    auto clampX = samplerWords->WORD0.get_CLAMP_X();
    auto clampY = samplerWords->WORD0.get_CLAMP_Y();
    auto clampZ = samplerWords->WORD0.get_CLAMP_Z();

    samplerDescriptor->setSAddressMode(GetMtlSamplerAddressMode(clampX));
    samplerDescriptor->setTAddressMode(GetMtlSamplerAddressMode(clampY));
    samplerDescriptor->setRAddressMode(GetMtlSamplerAddressMode(clampZ));

    auto maxAniso = samplerWords->WORD0.get_MAX_ANISO_RATIO();

    if (maxAniso > 0)
        samplerDescriptor->setMaxAnisotropy(1 << maxAniso);

    // Metal has no lod bias property. Do NOT fold the bias into the lod clamps: unlike Vulkan's
    // mipLodBias, Metal's sampler clamps also apply to EXPLICIT level() sampling, so a clamp-shift
    // would move explicit-LOD reads (e.g. the DoF chain's textureLod(1.0)) to different mips than
    // Vulkan. The bias is instead added to the LOD in the generated MSL (see LatteDecompilerEmitMSL:
    // bias() for implicit-LOD samples, folded into level() for explicit-LOD samples), so the clamps
    // here carry only the sampler's own MIN_LOD/MAX_LOD

    // depth compare. Only honor the guest compare function for units the shader samples with depth
    // compare - Vulkan does the same (compareEnable is set only when textureUsesDepthCompare, see
    // VulkanRendererCore.cpp). Applying it unconditionally turned every unit whose sampler register
    // happened to hold a non-NEVER compare function into a comparison sampler, so a plain
    // texture2d sample on such a unit read comparison results instead of texels
    if (depthCompareMode)
        samplerDescriptor->setCompareFunction(GetMtlCompareFunc((Latte::E_COMPAREFUNC)samplerWords->WORD0.get_DEPTH_COMPARE_FUNCTION()));
    else
        samplerDescriptor->setCompareFunction(MTL::CompareFunctionNever);

    // Border color
    auto borderColor = GetBorderColor(shaderType, stageSamplerIndex, samplerWords, true);
    samplerDescriptor->setBorderColor(borderColor);

    samplerState = m_mtlr->GetDevice()->newSamplerState(samplerDescriptor);

    return samplerState;
}

uint64 MetalSamplerCache::CalculateSamplerHash(const LatteContextRegister& lcr, LatteConst::ShaderType shaderType, uint32 stageSamplerIndex, const _LatteRegisterSetSampler* samplerWords, bool depthCompareMode)
{
    uint64 hash = 0;
    hash = std::rotl<uint64>(hash, 17);
    hash += (uint64)samplerWords->WORD0.getRawValue();
    hash = std::rotl<uint64>(hash, 17);
    hash += (uint64)samplerWords->WORD1.getRawValue();
    hash = std::rotl<uint64>(hash, 17);
    hash += (uint64)samplerWords->WORD2.getRawValue();

    auto borderColor = GetBorderColor(shaderType, stageSamplerIndex, samplerWords);

    hash = std::rotl<uint64>(hash, 5);
    hash += (uint64)borderColor;

    // the compare-mode gate is not part of the sampler registers: identical words must yield a
    // comparison sampler for a depth-compare unit and a plain one otherwise
    hash = std::rotl<uint64>(hash, 3);
    hash += depthCompareMode ? 1 : 0;

    // TODO: check this
	return hash;
}
