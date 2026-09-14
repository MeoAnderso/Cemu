#pragma once

#include "Common/precompiled.h"

struct LatteDecompilerShader;
struct LatteDecompilerOutput_t;
struct LatteDecompilerShaderResourceMapping;

// Maps a resource from a graphic pack shader's Vulkan set/binding (as declared via the
// LatteDecompilerEmitGLSLHeader.hpp layout macros) to the Metal resource index of resourceMappingMTL
struct MslResourceBindingRemap
{
	uint32 set;
	uint32 binding;
	uint32 mslBuffer; // [[buffer(n)]] index (uniform buffers / storage buffers)
	uint32 mslTexture; // [[texture(n)]] index (combined image samplers)
	uint32 mslSampler; // [[sampler(n)]] index (combined image samplers)
	bool isImageSampler{ false }; // texture remap instead of buffer remap
	sint32 textureUnit{ -1 }; // texture unit the remap belongs to (-1 for buffers)
	bool isDepthCompare{ false }; // the draw path binds a comparison sampler for this unit
};

// Computes the binding remaps for a custom graphic pack shader from the shader's Vulkan resource mapping
// (the set/binding numbers declared by the pack GLSL) and the Metal resource mapping used by the draw path.
// Returns false (with a log) if the shader uses features that cannot be translated (e.g. textures that are
// framebuffer-fetched on Metal)
bool MetalShaderTranslator_BuildBindingRemaps(const LatteDecompilerShader* shader, const LatteDecompilerShaderResourceMapping& vkMapping, std::vector<MslResourceBindingRemap>& remaps);

// Prepares a graphic pack custom shader for Metal before renderer shader creation. If the pack provides a
// Vulkan-authored GLSL replacement shader for this shader, it is translated GLSL -> SPIR-V (SpirvCompiler)
// -> MSL (SPIRV-Cross) and written into shader.strBuf_shaderSource, replacing the decompiled MSL (which acts
// as the fallback whenever translation fails or the shader uses unsupported features). Authored MSL pack
// shaders are left untouched and picked up by LatteShader_CreateRendererShader as usual.
// Called from the hook at the tail of LatteShader_CreateShaderFromDecompilerOutput, which runs on the
// shader-creation thread for both fresh compiles and shader-cache restores. The structural path flags in
// decompilerOutput are exported by the MSL emitter from the same context that generated the fallback
// source, so mesh-path/manual-fetch detection always matches the emitted shader.
void MetalShaderTranslator_PrepareGraphicPackShader(LatteDecompilerShader& shader, const LatteDecompilerOutput_t& decompilerOutput);

// Returns true if any active graphic pack provides a translatable GLSL pixel-shader
// replacement for the given shader base hash (any aux variant) that references the given
// texture unit. Used by the render-target classification (LatteShader_CalcPSRenderTargetIndices)
// to suppress framebuffer-fetch for units a pack shader samples as regular textures - without
// it the translated pack MSL would have no binding for those units and the translator would
// reject the pack shader entirely (the cause of the SM3DW AA-pass divergence). Results are
// cached per base hash; edited graphic packs require a restart (as elsewhere in the pack system)
bool MetalShaderTranslator_PackShaderSamplesTextureUnit(uint64 pixelShaderBaseHash, uint8 textureUnit);

// Flushes and closes the translated-MSL disk cache (opened lazily by PrepareGraphicPackShader).
// Called when the shader cache is closed (RendererShaderMtl::ShaderCacheLoading_Close)
void MetalShaderTranslator_CloseCache();
