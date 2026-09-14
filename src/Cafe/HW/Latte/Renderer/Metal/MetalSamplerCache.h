#pragma once

#include <Metal/Metal.hpp>

#include "HW/Latte/Core/LatteConst.h"
#include "HW/Latte/ISA/LatteReg.h"

class MetalSamplerCache
{
public:
    MetalSamplerCache(class MetalRenderer* metalRenderer) : m_mtlr{metalRenderer} {}
    ~MetalSamplerCache();

    // depthCompareMode mirrors Vulkan's compareEnable: the guest DEPTH_COMPARE_FUNCTION register is
    // only honoured for texture units the shader actually samples with depth compare. It is part of
    // the cache key because the sampler registers alone do not determine the resulting sampler
    MTL::SamplerState* GetSamplerState(const LatteContextRegister& lcr, LatteConst::ShaderType shaderType, uint32 stageSamplerIndex, const _LatteRegisterSetSampler* samplerWords, bool depthCompareMode);

private:
    class MetalRenderer* m_mtlr;

    std::map<uint64, MTL::SamplerState*> m_samplerCache;

    uint64 CalculateSamplerHash(const LatteContextRegister& lcr, LatteConst::ShaderType shaderType, uint32 stageSamplerIndex, const _LatteRegisterSetSampler* samplerWords, bool depthCompareMode);
};
