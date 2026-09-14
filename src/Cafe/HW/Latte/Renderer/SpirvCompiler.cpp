#include "Cafe/HW/Latte/Renderer/SpirvCompiler.h"
#include "Cemu/Logging/CemuLogging.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <mutex>

std::once_flag s_glslangInitOnce;

void SpirvCompiler_EnsureInitialized()
{
	std::call_once(s_glslangInitOnce, []()
	{
		glslang::InitializeProcess();
	});
}

consteval TBuiltInResource GetDefaultBuiltInResource()
{
	TBuiltInResource defaultResource = {};
	defaultResource.maxLights = 32;
	defaultResource.maxClipPlanes = 6;
	defaultResource.maxTextureUnits = 32;
	defaultResource.maxTextureCoords = 32;
	defaultResource.maxVertexAttribs = 64;
	defaultResource.maxVertexUniformComponents = 4096;
	defaultResource.maxVaryingFloats = 64;
	defaultResource.maxVertexTextureImageUnits = 32;
	defaultResource.maxCombinedTextureImageUnits = 80;
	defaultResource.maxTextureImageUnits = 32;
	defaultResource.maxFragmentUniformComponents = 4096;
	defaultResource.maxDrawBuffers = 32;
	defaultResource.maxVertexUniformVectors = 128;
	defaultResource.maxVaryingVectors = 8;
	defaultResource.maxFragmentUniformVectors = 16;
	defaultResource.maxVertexOutputVectors = 16;
	defaultResource.maxFragmentInputVectors = 15;
	defaultResource.minProgramTexelOffset = -8;
	defaultResource.maxProgramTexelOffset = 7;
	defaultResource.maxClipDistances = 8;
	defaultResource.maxComputeWorkGroupCountX = 65535;
	defaultResource.maxComputeWorkGroupCountY = 65535;
	defaultResource.maxComputeWorkGroupCountZ = 65535;
	defaultResource.maxComputeWorkGroupSizeX = 1024;
	defaultResource.maxComputeWorkGroupSizeY = 1024;
	defaultResource.maxComputeWorkGroupSizeZ = 64;
	defaultResource.maxComputeUniformComponents = 1024;
	defaultResource.maxComputeTextureImageUnits = 16;
	defaultResource.maxComputeImageUniforms = 8;
	defaultResource.maxComputeAtomicCounters = 8;
	defaultResource.maxComputeAtomicCounterBuffers = 1;
	defaultResource.maxVaryingComponents = 60;
	defaultResource.maxVertexOutputComponents = 64;
	defaultResource.maxGeometryInputComponents = 64;
	defaultResource.maxGeometryOutputComponents = 128;
	defaultResource.maxFragmentInputComponents = 128;
	defaultResource.maxImageUnits = 8;
	defaultResource.maxCombinedImageUnitsAndFragmentOutputs = 8;
	defaultResource.maxCombinedShaderOutputResources = 8;
	defaultResource.maxImageSamples = 0;
	defaultResource.maxVertexImageUniforms = 0;
	defaultResource.maxTessControlImageUniforms = 0;
	defaultResource.maxTessEvaluationImageUniforms = 0;
	defaultResource.maxGeometryImageUniforms = 0;
	defaultResource.maxFragmentImageUniforms = 8;
	defaultResource.maxCombinedImageUniforms = 8;
	defaultResource.maxGeometryTextureImageUnits = 16;
	defaultResource.maxGeometryOutputVertices = 256;
	defaultResource.maxGeometryTotalOutputComponents = 1024;
	defaultResource.maxGeometryUniformComponents = 1024;
	defaultResource.maxGeometryVaryingComponents = 64;
	defaultResource.maxTessControlInputComponents = 128;
	defaultResource.maxTessControlOutputComponents = 128;
	defaultResource.maxTessControlTextureImageUnits = 16;
	defaultResource.maxTessControlUniformComponents = 1024;
	defaultResource.maxTessControlTotalOutputComponents = 4096;
	defaultResource.maxTessEvaluationInputComponents = 128;
	defaultResource.maxTessEvaluationOutputComponents = 128;
	defaultResource.maxTessEvaluationTextureImageUnits = 16;
	defaultResource.maxTessEvaluationUniformComponents = 1024;
	defaultResource.maxTessPatchComponents = 120;
	defaultResource.maxPatchVertices = 32;
	defaultResource.maxTessGenLevel = 64;
	defaultResource.maxViewports = 16;
	defaultResource.maxVertexAtomicCounters = 0;
	defaultResource.maxTessControlAtomicCounters = 0;
	defaultResource.maxTessEvaluationAtomicCounters = 0;
	defaultResource.maxGeometryAtomicCounters = 0;
	defaultResource.maxFragmentAtomicCounters = 8;
	defaultResource.maxCombinedAtomicCounters = 8;
	defaultResource.maxAtomicCounterBindings = 1;
	defaultResource.maxVertexAtomicCounterBuffers = 0;
	defaultResource.maxTessControlAtomicCounterBuffers = 0;
	defaultResource.maxTessEvaluationAtomicCounterBuffers = 0;
	defaultResource.maxGeometryAtomicCounterBuffers = 0;
	defaultResource.maxFragmentAtomicCounterBuffers = 1;
	defaultResource.maxCombinedAtomicCounterBuffers = 1;
	defaultResource.maxAtomicCounterBufferSize = 16384;
	defaultResource.maxTransformFeedbackBuffers = 4;
	defaultResource.maxTransformFeedbackInterleavedComponents = 64;
	defaultResource.maxCullDistances = 8;
	defaultResource.maxCombinedClipAndCullDistances = 8;
	defaultResource.maxSamples = 4;
	defaultResource.maxMeshOutputVerticesNV = 256;
	defaultResource.maxMeshOutputPrimitivesNV = 512;
	defaultResource.maxMeshWorkGroupSizeX_NV = 32;
	defaultResource.maxMeshWorkGroupSizeY_NV = 1;
	defaultResource.maxMeshWorkGroupSizeZ_NV = 1;
	defaultResource.maxTaskWorkGroupSizeX_NV = 32;
	defaultResource.maxTaskWorkGroupSizeY_NV = 1;
	defaultResource.maxTaskWorkGroupSizeZ_NV = 1;
	defaultResource.maxMeshViewCountNV = 4;

	defaultResource.limits = {};
	defaultResource.limits.nonInductiveForLoops = true;
	defaultResource.limits.whileLoops = true;
	defaultResource.limits.doWhileLoops = true;
	defaultResource.limits.generalUniformIndexing = true;
	defaultResource.limits.generalAttributeMatrixVectorIndexing = true;
	defaultResource.limits.generalVaryingIndexing = true;
	defaultResource.limits.generalSamplerIndexing = true;
	defaultResource.limits.generalVariableIndexing = true;
	defaultResource.limits.generalConstantMatrixVectorIndexing = true;
	return defaultResource;
}

static EShLanguage GetEShLanguage(RendererShader::ShaderType type)
{
	switch (type)
	{
	case RendererShader::ShaderType::kVertex:
		return EShLangVertex;
	case RendererShader::ShaderType::kFragment:
		return EShLangFragment;
	case RendererShader::ShaderType::kGeometry:
		return EShLangGeometry;
	default:
		cemu_assert_debug(false);
	}
	return EShLangVertex;
}

bool SpirvCompiler_Compile(const std::string& glslSource, RendererShader::ShaderType shaderType, uint64 baseHash, uint64 auxHash, bool debugInfo, std::vector<uint32>& spirvOut)
{
	SpirvCompiler_EnsureInitialized();

	const EShLanguage stage = GetEShLanguage(shaderType);

	glslang::TShader Shader(stage);
	const char* glslCStr = glslSource.c_str();
	Shader.setStrings(&glslCStr, 1);
	// note: EShClientVulkan predefines the VULKAN macro, which selects the set/binding layout branch
	// in graphic pack shaders. This is a pack-GLSL dialect decision, not a backend one: the Metal
	// translation path (MetalShaderTranslator) deliberately compiles through the same Vulkan client
	// environment so pack shaders observe identical layout and preprocessor state on both backends
	Shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
	Shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetClientVersion::EShTargetVulkan_1_1);
	Shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetLanguageVersion::EShTargetSpv_1_3);

	std::string preprocessedGLSL;
	glslang::TShader::ForbidIncluder Includer;
	TBuiltInResource Resources = GetDefaultBuiltInResource();
	EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);
	if (!Shader.preprocess(&Resources, 450, ENoProfile, false, false, messages, &preprocessedGLSL, Includer))
	{
		cemuLog_log(LogType::Force, fmt::format("GLSL Preprocessing Failed For {:016x}_{:016x}: \"{}\"", baseHash, auxHash, Shader.getInfoLog()));
		return false;
	}

	const char* preprocessedCStr = preprocessedGLSL.c_str();
	Shader.setStrings(&preprocessedCStr, 1);
	if (!Shader.parse(&Resources, 100, false, messages))
	{
		cemuLog_log(LogType::Force, fmt::format("GLSL parsing failed for {:016x}_{:016x}: \"{}\"", baseHash, auxHash, Shader.getInfoLog()));
		cemuLog_logDebug(LogType::Force, "GLSL source:\n{}", glslSource);
		return false;
	}

	glslang::TProgram Program;
	Program.addShader(&Shader);
	if (!Program.link(messages))
	{
		cemuLog_log(LogType::Force, fmt::format("GLSL linking failed for {:016x}_{:016x}: \"{}\"", baseHash, auxHash, Program.getInfoLog()));
		return false;
	}
	if (!Program.mapIO())
	{
		cemuLog_log(LogType::Force, fmt::format("GLSL linking failed for {:016x}_{:016x}: \"{}\"", baseHash, auxHash, Program.getInfoLog()));
		return false;
	}

	spv::SpvBuildLogger logger;
	glslang::SpvOptions spvOptions;
	spvOptions.disableOptimizer = false;
	spvOptions.validate = false;
	spvOptions.optimizeSize = true;
	if (debugInfo)
	{
		spvOptions.generateDebugInfo = true;
		spvOptions.emitNonSemanticShaderDebugInfo = true;
		spvOptions.emitNonSemanticShaderDebugSource = true;

		Shader.addSourceText(glslSource.c_str(), (uint32)glslSource.size());
		Shader.setSourceFile(fmt::format("shader_{:016x}_{:016x}.glsl", baseHash, auxHash).c_str());
	}

	GlslangToSpv(*Program.getIntermediate(stage), spirvOut, &logger, &spvOptions);

	if (spirvOut.empty())
	{
		// translation failed without a glslang error - surface the SPIR-V build logger messages
		const std::string spvMessages = logger.getAllMessages();
		if (!spvMessages.empty())
			cemuLog_log(LogType::Force, "GLSL to SPIR-V translation failed for {:016x}_{:016x}: \"{}\"", baseHash, auxHash, spvMessages);
		return false;
	}
	return true;
}
