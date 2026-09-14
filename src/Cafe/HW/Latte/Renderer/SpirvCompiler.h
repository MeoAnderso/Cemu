#pragma once

#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Renderer/RendererShader.h"

// Shared GLSL -> SPIR-V front-end (glslang), used by the Vulkan backend and by MetalShaderTranslator.
// Note: unlike the original inline Vulkan code, compile failures are reported via the return value
// (with the glslang info log) instead of cemu_assert_debug(false), so invalid graphic pack GLSL
// degrades gracefully (especially for MetalShaderTranslator's speculative compiles)

// Initializes the glslang process once (safe to call multiple times and from any thread)
void SpirvCompiler_EnsureInitialized();

// Compiles GLSL source to SPIR-V via glslang, using the Vulkan client environment (which predefines
// the VULKAN macro, selecting the set/binding layout branch in graphic pack shaders).
// Returns false and logs on failure
bool SpirvCompiler_Compile(const std::string& glslSource, RendererShader::ShaderType shaderType, uint64 baseHash, uint64 auxHash, bool debugInfo, std::vector<uint32>& spirvOut);
