#pragma once

#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Renderer/RendererShader.h"

// Shared GLSL -> SPIR-V front-end (glslang), used by the Vulkan backend and by MetalShaderTranslator.
// Compile failures are reported through the return value (with the glslang info log) so that malformed
// graphic pack GLSL degrades to the fallback path instead of breaking into the debugger.
// assertOnFailure restores the debug assertion for callers that do not compile speculatively (the
// Vulkan backend, where a failure means the game's own shader could not be built)

// Initializes the glslang process once (safe to call multiple times and from any thread)
void SpirvCompiler_EnsureInitialized();

// Compiles GLSL source to SPIR-V via glslang, using the Vulkan client environment (which predefines
// the VULKAN macro, selecting the set/binding layout branch in graphic pack shaders).
// Returns false and logs on failure
bool SpirvCompiler_Compile(const std::string& glslSource, RendererShader::ShaderType shaderType, uint64 baseHash, uint64 auxHash, bool debugInfo, std::vector<uint32>& spirvOut, bool assertOnFailure = false);
