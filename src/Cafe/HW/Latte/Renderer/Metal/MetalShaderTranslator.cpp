#include "Cafe/HW/Latte/Renderer/Metal/MetalShaderTranslator.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"
#include "Cafe/HW/Latte/Renderer/RendererShader.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"
#include "Cafe/HW/Latte/Renderer/SpirvCompiler.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "Cafe/GraphicPack/GraphicPack2.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include "Cemu/FileCache/FileCache.h"
#include "Cemu/Logging/CemuLogging.h"
#include "config/ActiveSettings.h"
#include "util/helpers/StringBuf.h"

#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>
#include <spirv_cross/spirv_msl.hpp>

// Graphic pack shaders are authored in GLSL for the OpenGL/Vulkan backends. This translates them
// to MSL via GLSL -> SPIR-V (SpirvCompiler) -> MSL (SPIRV-Cross). The SPIR-V resource bindings
// (which match the decompiled shader's Vulkan resource mapping) are remapped to the Metal resource
// indices of resourceMappingMTL, so the translated shader can be bound by the existing draw path.
// On any failure the decompiled MSL in strBuf_shaderSource is left in place and used as fallback

static bool TranslateGLSLToMSL(const std::string& glslSource, RendererShader::ShaderType shaderType, const std::vector<MslResourceBindingRemap>& remaps, uint64 baseHash, uint64 auxHash, std::string& mslOut);

// Fallback accounting. A failed translation silently falls back to the decompiled shader, which means
// the pack customization is not applied on Metal - so every reason is counted through the shared
// facility (see MetalDiagnostics.h): the first occurrence of each reason logs one Force-level line
// naming what the user loses, and all of them roll up into the session summary. The per-shader
// specifics below (which shader, which texture unit, how to fix the pack) go to the verbose
// LogType::MetalBackend channel instead of being printed once and lost.

// Translated-MSL disk cache. The GLSL -> SPIR-V -> MSL pipeline is far too expensive to re-run for
// every graphic pack shader on every session (mirrors the Vulkan backend, which persists SPIR-V to a
// FileCache). Successful translations are stored keyed by shader hash + pack GLSL content hash, so
// edited graphic packs invalidate their cached translations. Concurrent shader compile threads use
// the same read/write pattern as the Vulkan SPIR-V cache
static constexpr uint64 kTranslatedMslCacheVersion = 0x01; // bump when translation output changes (SPIRV-Cross options, remap logic, SpirvCompiler environment)
// shared_ptr, not a raw pointer: CloseCache runs when the shader cache is closed (which happens
// when a title stops, see LatteThread_Exit) while shader-compile threads may still be inside
// GetFile/AddFile. Dropping the last reference lets the in-flight users finish and the FileCache
// destructor flush afterwards, instead of pulling the object out from under them
static std::shared_ptr<FileCache> s_translatedMslCache;
static std::mutex s_translatedMslCacheMutex;

static void ComputeTranslatedMslCacheKey(const std::string& glslSource, uint64 baseHash, uint64 auxHash, uint64& key1, uint64& key2)
{
	// FNV-1a 64 over the pack GLSL source: the shader hashes alone don't change when a user edits
	// the pack (pack shaders are matched by the hash of the original game shader), so the source
	// hash is what makes edited packs invalidate their cached translations
	uint64 sourceHash = 0xCBF29CE484222325ull;
	for (char c : glslSource)
	{
		sourceHash ^= (uint8)c;
		sourceHash *= 0x100000001B3ull;
	}
	key1 = auxHash;
	// 0x9E3779B97F4A7C15: golden-ratio mixer, so a version bump changes the whole key
	key2 = baseHash ^ (sourceHash + kTranslatedMslCacheVersion * 0x9E3779B97F4A7C15ull);
}

// Opens the cache lazily on first use and returns a strong reference to it.
// Deliberately not std::call_once: CloseCache runs on title exit and Cemu can start another
// title in the same process, so with call_once the cache would stay closed for the rest of the
// session and every pack shader would be re-translated and never persisted. Vulkan re-opens its
// own cache per session the same way (ShaderCacheLoading_begin)
static std::shared_ptr<FileCache> GetTranslatedMslCache()
{
	std::lock_guard<std::mutex> lock(s_translatedMslCacheMutex);
	if (!s_translatedMslCache)
	{
		const uint32 cacheMagic = RendererShader::GeneratePrecompiledCacheId();
		const fs::path cachePath = ActiveSettings::GetCachePath("shaderCache/precompiled/translated_msl.bin");
		s_translatedMslCache.reset(FileCache::Open(cachePath, true, cacheMagic));
		if (!s_translatedMslCache)
			cemuLog_log(LogType::Force, "MetalShaderTranslator: unable to open translated MSL cache");
	}
	return s_translatedMslCache;
}

void MetalShaderTranslator_CloseCache()
{
	std::lock_guard<std::mutex> lock(s_translatedMslCacheMutex);
	s_translatedMslCache.reset();
}

static const char* GetShaderTypeName(RendererShader::ShaderType type)
{
	switch (type)
	{
	case RendererShader::ShaderType::kVertex:
		return "vertex";
	case RendererShader::ShaderType::kFragment:
		return "pixel";
	case RendererShader::ShaderType::kGeometry:
		return "geometry";
	default:
		cemu_assert_debug(false);
	}
	return "unknown";
}

bool MetalShaderTranslator_BuildBindingRemaps(const LatteDecompilerShader* shader, const LatteDecompilerShaderResourceMapping& vkMapping, std::vector<MslResourceBindingRemap>& remaps)
{
	remaps.clear();

	const auto& vk = vkMapping; // set/binding numbers as declared by the pack GLSL (Vulkan layout)
	const auto& mtl = shader->resourceMapping; // [[buffer]]/[[texture]] indices used by the Metal draw path (only valid when the active renderer is Metal)

	const uint32 vkSetIndex = (uint32)(sint32)vk.setIndex;

	// textures (combined image samplers)
	for (sint32 i = 0; i < LATTE_NUM_MAX_TEX_UNITS; i++)
	{
		if (vk.textureUnitToBindingPoint[i] < 0)
			continue; // texture unit not used by the decompiled shader
		if (mtl.textureUnitToBindingPoint[i] < 0)
		{
			// this texture unit is accessed via framebuffer fetch on Metal (it is bound to a render target)
			MetalDiag_Count(MetalDiagEvent::PackShaderFallbackFramebuffer,
				"{:016x}_{:016x} uses texture unit {} which is a render target on Metal (framebuffer fetch). Consider authoring this shader in MSL directly (_ps_msl.txt)",
				shader->baseHash, shader->auxHash, i);
			return false;
		}
		// depth-compare units are supported: the draw path binds the depth view and a comparison
		// sampler, and SPIRV-Cross emits depth2d + sample_compare when the pack GLSL declares a
		// shadow sampler (sampler2DShadow). TranslateGLSLToMSL rejects the shader if the SPIR-V
		// image is not actually declared as depth, since that would bind a comparison sampler
		// to a regular texture2d. Skipping the remap for these units made every shadow-sampling
		// pack shader fail with "unmapped resource" instead
		uint32 mtlBinding = (uint32)(sint32)mtl.textureUnitToBindingPoint[i];
		remaps.emplace_back(MslResourceBindingRemap{ vkSetIndex, (uint32)(sint32)vk.textureUnitToBindingPoint[i], 0, mtlBinding, mtlBinding, true, (sint32)i, shader->textureUsesDepthCompare[i] != 0 });
	}

	// support buffer (ufBlock)
	if (vk.uniformVarsBufferBindingPoint >= 0)
	{
		if (mtl.uniformVarsBufferBindingPoint < 0)
		{
			MetalDiag_Count(MetalDiagEvent::PackShaderFallbackResource,
				"{:016x}_{:016x} uses the support buffer on Vulkan but has no Metal support buffer binding",
				shader->baseHash, shader->auxHash);
			return false;
		}
		remaps.emplace_back(MslResourceBindingRemap{ vkSetIndex, (uint32)(sint32)vk.uniformVarsBufferBindingPoint, (uint32)(sint32)mtl.uniformVarsBufferBindingPoint, 0, 0, false });
	}

	// uniform buffers
	for (sint32 i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
	{
		if (vk.uniformBuffersBindingPoint[i] < 0)
			continue;
		if (mtl.uniformBuffersBindingPoint[i] < 0)
		{
			MetalDiag_Count(MetalDiagEvent::PackShaderFallbackResource,
				"{:016x}_{:016x} uses uniform buffer {} which has no Metal binding",
				shader->baseHash, shader->auxHash, i);
			return false;
		}
		remaps.emplace_back(MslResourceBindingRemap{ vkSetIndex, (uint32)(sint32)vk.uniformBuffersBindingPoint[i], (uint32)(sint32)mtl.uniformBuffersBindingPoint[i], 0, 0, false });
	}

	// storage buffer for transform feedback (alternative streamout path)
	if (vk.tfStorageBindingPoint >= 0)
	{
		if (mtl.tfStorageBindingPoint < 0)
		{
			MetalDiag_Count(MetalDiagEvent::PackShaderFallbackResource,
				"{:016x}_{:016x} uses the transform feedback storage buffer which has no Metal binding",
				shader->baseHash, shader->auxHash);
			return false;
		}
		remaps.emplace_back(MslResourceBindingRemap{ vkSetIndex, (uint32)(sint32)vk.tfStorageBindingPoint, (uint32)(sint32)mtl.tfStorageBindingPoint, 0, 0, false });
	}

	// vertex attributes need no remap - attributeMapping is identical between Vulkan and Metal
	return true;
}

// Returns the remap entry for a Vulkan (set, binding) pair and resource class, or nullptr.
// Hand-written pack shaders can declare collisions (e.g. a uniform block declared at the same
// set/binding as a texture) - matching without the resource class would silently bind the
// shader's buffer to the texture's Metal indices (or vice versa). A wrong-class match is
// treated as unmapped so the shader falls back loudly instead of rendering garbage
static const MslResourceBindingRemap* FindRemap(const std::vector<MslResourceBindingRemap>& remaps, uint32 set, uint32 binding, bool isImageSampler)
{
	for (const auto& r : remaps)
	{
		if (r.set == set && r.binding == binding && r.isImageSampler == isImageSampler)
			return &r;
	}
	return nullptr;
}

// Logs a failed lookup for the given SPIRV-Cross resource
static bool ReportUnmappedResource(spirv_cross::CompilerMSL& compiler, const spirv_cross::Resource& resource, uint64 baseHash, uint64 auxHash, const char* category)
{
	uint32 set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
	uint32 binding = compiler.get_decoration(resource.id, spv::DecorationBinding);
	MetalDiag_Count(MetalDiagEvent::PackShaderFallbackResource,
		"{:016x}_{:016x} has unmapped {} resource \"{}\" (set {}, binding {})",
		baseHash, auxHash, category, resource.name.c_str(), set, binding);
	return false;
}

// Returns true if any active graphic pack provides a translatable GLSL pixel-shader
// replacement for the given shader base hash (any aux variant) that references the given
// texture unit. See the header comment. The unit set per base hash is extracted once from
// the pack GLSL sources ("textureUnitPS<N>" tokens) and cached
bool MetalShaderTranslator_PackShaderSamplesTextureUnit(uint64 pixelShaderBaseHash, uint8 textureUnit)
{
	struct BaseEntry
	{
		uint32 unitMask = 0;
		bool scanned = false;
	};
	static std::unordered_map<uint64, BaseEntry> s_baseUnitMasks;
	static std::mutex s_mutex;
	std::lock_guard<std::mutex> lock(s_mutex);
	BaseEntry& entry = s_baseUnitMasks[pixelShaderBaseHash];
	if (!entry.scanned)
	{
		entry.scanned = true;
		for (const auto& gp : GraphicPack2::GetActiveGraphicPacks())
		{
			for (const auto& customShader : gp->GetCustomShaders())
			{
				// same filter as the translator's pack lookup: GLSL shaders authored for the
				// Vulkan backend (authored MSL shaders take a different path, pre-Vulkan GLSL
				// is incompatible with the translated pipeline)
				if (customShader.shader_base_hash != pixelShaderBaseHash || customShader.type != GraphicPack2::GP_SHADER_TYPE::PIXEL)
					continue;
				if (customShader.isMetalShader || customShader.isPreVulkanShader)
					continue;
				// scan for textureUnitPS<N> references
				const std::string token = "textureUnitPS";
				size_t pos = 0;
				while ((pos = customShader.source.find(token, pos)) != std::string::npos)
				{
					pos += token.size();
					size_t end = pos;
					while (end < customShader.source.size() && customShader.source[end] >= '0' && customShader.source[end] <= '9')
						end++;
					if (end > pos)
					{
						const uint32 unit = (uint32)strtoul(customShader.source.c_str() + pos, nullptr, 10);
						if (unit < 32)
							entry.unitMask |= (1u << unit);
					}
				}
			}
		}
	}
	return (entry.unitMask & (1u << textureUnit)) != 0;
}

void MetalShaderTranslator_PrepareGraphicPackShader(LatteDecompilerShader& shader, const LatteDecompilerOutput_t& decompilerOutput)
{
	if (shader.hasError)
		return;
	const uint64 baseHash = shader.baseHash;
	const uint64 auxHash = shader.auxHash;
	// pack files are named after the upstream (backend-independent) aux hash; the Metal-extended
	// auxHash only identifies pipeline/shader variants
	const uint64 packAuxHash = shader.packAuxHash ? shader.packAuxHash : auxHash;

	RendererShader::ShaderType shaderType;
	GraphicPack2::GP_SHADER_TYPE gpShaderType;
	if (shader.shaderType == LatteConst::ShaderType::Vertex)
	{
		shaderType = RendererShader::ShaderType::kVertex;
		gpShaderType = GraphicPack2::GP_SHADER_TYPE::VERTEX;
	}
	else if (shader.shaderType == LatteConst::ShaderType::Geometry)
	{
		shaderType = RendererShader::ShaderType::kGeometry;
		gpShaderType = GraphicPack2::GP_SHADER_TYPE::GEOMETRY;
	}
	else if (shader.shaderType == LatteConst::ShaderType::Pixel)
	{
		shaderType = RendererShader::ShaderType::kFragment;
		gpShaderType = GraphicPack2::GP_SHADER_TYPE::PIXEL;
	}
	else
		return;

	// explicitly authored MSL pack shaders win over translation (handled by LatteShader_CreateRendererShader)
	if (GraphicPack2::FindCustomShaderSource(baseHash, packAuxHash, gpShaderType, false, true))
		return;
	// otherwise look for a GLSL pack shader authored for the Vulkan backend
	const std::string* packShaderSrc = GraphicPack2::FindCustomShaderSource(baseHash, packAuxHash, gpShaderType, true, false);
	if (!packShaderSrc)
		return;

	if (!shader.strBuf_shaderSource)
	{
		cemuLog_log(LogType::Force, "MetalShaderTranslator: {:016x}_{:016x} has no decompiled fallback source. Skipping translation", baseHash, auxHash);
		return;
	}
	// geometry shaders are emulated with Metal mesh shaders, which cannot be generated by the
	// translation. Pixel shaders are unaffected by the mesh emulation: their [[user(locn)]]
	// input linkage comes from the same PS input table regardless of the path, so translated
	// pack pixel shaders work in the mesh path too. Vertex shaders in the mesh path are refused
	// by the manual vertex fetch check below. The decompiler exports the structural path flags it
	// used when emitting the fallback MSL (see LatteDecompiler_emitMSLShader), so this decision
	// always matches the shader that would be used as fallback
	if (shaderType == RendererShader::ShaderType::kGeometry)
	{
		MetalDiag_Count(MetalDiagEvent::PackShaderFallbackMeshPath,
			"{:016x}_{:016x} uses the geometry shader path", baseHash, auxHash);
		return;
	}
	// a translated GLSL vertex shader consumes [[stage_in]]-style attributes and cannot match the
	// manual vertex fetch draw path
	if (shaderType == RendererShader::ShaderType::kVertex && decompilerOutput.fetchVertexManually)
	{
		MetalDiag_Count(MetalDiagEvent::PackShaderFallbackVertexFetch,
			"{:016x}_{:016x} uses manual vertex fetch", baseHash, auxHash);
		return;
	}

	// Force-level (release-visible) activity logs: pack shader matches are the only other way to
	// tell a silent fallback from a pack shader that simply doesn't apply
	cemuLog_log(LogType::Force, "MetalShaderTranslator: applying graphic pack shader {:016x}_{:016x} ({})", baseHash, auxHash, GetShaderTypeName(shaderType));

	// check the translated-MSL cache before running the GLSL -> SPIR-V -> MSL pipeline (the cache
	// key covers the pack GLSL source, so edited graphic packs invalidate their cached entries)
	uint64 cacheKey1, cacheKey2;
	ComputeTranslatedMslCacheKey(*packShaderSrc, baseHash, auxHash, cacheKey1, cacheKey2);
	// held for the whole function so the cache cannot be closed underneath the read/write below
	std::shared_ptr<FileCache> translatedMslCache = GetTranslatedMslCache();
	if (translatedMslCache)
	{
		std::vector<uint8> cacheFileData;
		if (translatedMslCache->GetFile({ cacheKey1, cacheKey2 }, cacheFileData))
		{
			// validate before installing: a corrupted entry would permanently fail MSL compilation
			// (RendererShaderMtl treats library failures as permanent, skipping the draws forever
			// with no chance of re-translation). Translated MSL always includes the metal_stdlib
			// header, so an entry without it is not a valid translation - fall through and
			// re-translate, which also overwrites the bad entry
			const std::string cachedSource((const char*)cacheFileData.data(), cacheFileData.size());
			if (cachedSource.find("metal_stdlib") != std::string::npos)
			{
				shader.strBuf_shaderSource->reset();
				shader.strBuf_shaderSource->add(cachedSource);
				shader.isCustomShader = true;
				cemuLog_log(LogType::Force, "MetalShaderTranslator: loaded translated graphic pack shader {:016x}_{:016x} ({}) from cache", baseHash, auxHash, GetShaderTypeName(shaderType));
				MetalDiag_Count(MetalDiagEvent::PackShaderTranslated, "{:016x}_{:016x} ({}, cache hit)", baseHash, auxHash, GetShaderTypeName(shaderType));
				return;
			}
			cemuLog_log(LogType::Force, "MetalShaderTranslator: translated cache entry for {:016x}_{:016x} is invalid ({}). Re-translating", baseHash, auxHash, cachedSource.size());
		}
	}

	std::string translatedMsl;
	std::vector<MslResourceBindingRemap> remaps;
	try
	{
		if (!MetalShaderTranslator_BuildBindingRemaps(&shader, decompilerOutput.resourceMappingVK, remaps) ||
			!TranslateGLSLToMSL(*packShaderSrc, shaderType, remaps, baseHash, auxHash, translatedMsl))
			return; // failure already logged, decompiled MSL stays in place
	}
	catch (const std::exception& ex)
	{
		cemuLog_log(LogType::Force, "MetalShaderTranslator: translation of {:016x}_{:016x} threw an exception: \"{}\". Using decompiled shader", baseHash, auxHash, ex.what());
		return;
	}
	shader.strBuf_shaderSource->reset();
	shader.strBuf_shaderSource->add(translatedMsl);
	shader.isCustomShader = true;
	if (translatedMslCache)
		translatedMslCache->AddFile({ cacheKey1, cacheKey2 }, (const uint8*)translatedMsl.data(), (sint32)translatedMsl.size());
	// log the resulting mapping (Force level so it is visible in release builds): vk set/binding
	// as declared by the pack GLSL -> Metal indices used by the draw path
	{
		std::string remapSummary;
		for (const auto& r : remaps)
		{
			if (!remapSummary.empty())
				remapSummary += ", ";
			if (r.isImageSampler)
				remapSummary += fmt::format("tex(set{} b{} -> tex{} texUnit{})", r.set, r.binding, r.mslTexture, r.textureUnit);
			else
				remapSummary += fmt::format("buf(set{} b{} -> buf{})", r.set, r.binding, r.mslBuffer);
		}
		cemuLog_log(LogType::Force, "MetalShaderTranslator: translated graphic pack shader {:016x}_{:016x} ({}) to MSL [{}]", baseHash, auxHash, GetShaderTypeName(shaderType), remapSummary);
		MetalDiag_Count(MetalDiagEvent::PackShaderTranslated, "{:016x}_{:016x} ({})", baseHash, auxHash, GetShaderTypeName(shaderType));
	}
}

static bool TranslateGLSLToMSL(const std::string& glslSource, RendererShader::ShaderType shaderType, const std::vector<MslResourceBindingRemap>& remaps, uint64 baseHash, uint64 auxHash, std::string& mslOut)
{
	const char* shaderTypeName = GetShaderTypeName(shaderType);

	// GLSL -> SPIR-V via SpirvCompiler
	std::vector<uint32> spirvBuffer;
	if (!SpirvCompiler_Compile(glslSource, shaderType, baseHash, auxHash, false, spirvBuffer))
	{
		MetalDiag_Count(MetalDiagEvent::PackShaderFallbackFrontend, "{:016x}_{:016x} ({})", baseHash, auxHash, shaderTypeName);
		return false;
	}

	// SPIR-V -> MSL via SPIRV-Cross
	try
	{
		spirv_cross::CompilerMSL compiler(std::move(spirvBuffer));

		// validate that every resource declared by the shader has a remap to a Metal binding
		const auto& resources = compiler.get_shader_resources();

		if (!resources.push_constant_buffers.empty())
			return ReportUnmappedResource(compiler, resources.push_constant_buffers[0], baseHash, auxHash, "push constant");
		if (!resources.subpass_inputs.empty())
			return ReportUnmappedResource(compiler, resources.subpass_inputs[0], baseHash, auxHash, "subpass input");
		if (!resources.acceleration_structures.empty())
			return ReportUnmappedResource(compiler, resources.acceleration_structures[0], baseHash, auxHash, "acceleration structure");

		const auto checkResourceList = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& list, const char* category, bool isBuffer) -> bool
		{
			for (const auto& res : list)
			{
				uint32 set = compiler.get_decoration(res.id, spv::DecorationDescriptorSet);
				uint32 binding = compiler.get_decoration(res.id, spv::DecorationBinding);
				const MslResourceBindingRemap* remap = FindRemap(remaps, set, binding, !isBuffer);
				if (!remap)
				{
					// a remap of the other resource class at the same (set, binding) means the pack
					// shader declares this resource at an index the original shader uses for the
					// other class - the pack file is misdeclared, not merely unsupported
					if (FindRemap(remaps, set, binding, isBuffer))
					{
						MetalDiag_Count(MetalDiagEvent::PackShaderFallbackResource,
							"{:016x}_{:016x} declares {} resource \"{}\" at (set {}, binding {}), but the original shader uses this binding for the other resource class. Check the pack shader's binding layout",
							baseHash, auxHash, category, res.name.c_str(), set, binding);
						return false;
					}
					return ReportUnmappedResource(compiler, res, baseHash, auxHash, category);
				}
				// make sure the Metal binding is within the range used by the draw path
				uint32 mslIndex = isBuffer ? remap->mslBuffer : remap->mslTexture;
				uint32 maxIndex = isBuffer ? MAX_MTL_BUFFERS : MAX_MTL_TEXTURES;
				if (mslIndex >= maxIndex)
				{
					MetalDiag_Count(MetalDiagEvent::PackShaderFallbackBinding,
						"{:016x}_{:016x} resource \"{}\" maps to invalid Metal binding {}", baseHash, auxHash, res.name.c_str(), mslIndex);
					return false;
				}
				// texture units that the draw path serves with a comparison sampler must be declared
				// as shadow samplers in the pack GLSL (SPIRV-Cross then emits depth2d + sample_compare);
				// a regular sampler2d would silently get compare semantics applied to it
				if (!isBuffer && remap->isDepthCompare)
				{
					// get_type_from_variable, not get_type: res.id is the OpVariable id, and
					// get_type expects a type id (it throws Bad cast on a variable)
					auto& type = compiler.get_type_from_variable(res.id);
					if (type.image.depth != 1)
					{
						MetalDiag_Count(MetalDiagEvent::PackShaderFallbackDepthCompare,
							"{:016x}_{:016x} samples texture unit {} with depth compare, but the custom shader declares it as a regular sampler (use sampler2DShadow)",
							baseHash, auxHash, remap->textureUnit);
						return false;
					}
				}
			}
			return true;
		};
		if (!checkResourceList(resources.sampled_images, "texture", false) ||
			!checkResourceList(resources.separate_images, "texture", false) ||
			!checkResourceList(resources.separate_samplers, "sampler", false) ||
			!checkResourceList(resources.uniform_buffers, "uniform buffer", true) ||
			!checkResourceList(resources.storage_buffers, "storage buffer", true) ||
			!checkResourceList(resources.storage_images, "storage image", false))
			return false;

		// make sure no two resources of the same class remap to the same Metal index
		for (size_t i = 0; i < remaps.size(); i++)
		{
			for (size_t j = i + 1; j < remaps.size(); j++)
			{
				const auto& a = remaps[i];
				const auto& b = remaps[j];
				if (a.isImageSampler != b.isImageSampler)
					continue; // textures and buffers use separate Metal index namespaces
				const bool collides = a.isImageSampler ? (a.mslTexture == b.mslTexture) : (a.mslBuffer == b.mslBuffer);
				if (collides)
				{
					MetalDiag_Count(MetalDiagEvent::PackShaderFallbackCollision,
						"{:016x}_{:016x} has colliding Metal bindings (set {} binding {} and set {} binding {})",
						baseHash, auxHash, a.set, a.binding, b.set, b.binding);
					return false;
				}
			}
		}

		auto mslOptions = compiler.get_msl_options();
		mslOptions.msl_version = 20200; // MSL 2.2, available on all supported macOS versions
		mslOptions.enable_point_size_builtin = true; // allow gl_PointSize in vertex shader replacements
		mslOptions.argument_buffers = false; // the draw path binds resources per-argument
		compiler.set_msl_options(mslOptions);
		// fixup_clipspace and flip_vert_y stay disabled: SET_POSITION already handles the depth range conversion

		// the execution model of the shader, also used to match the binding remaps
		spv::ExecutionModel execModel = (shaderType == RendererShader::ShaderType::kVertex) ? spv::ExecutionModelVertex :
			(shaderType == RendererShader::ShaderType::kGeometry) ? spv::ExecutionModelGeometry : spv::ExecutionModelFragment;

		// apply the binding remaps (matched against the exact execution model of the entry point)
		for (const auto& r : remaps)
		{
			spirv_cross::MSLResourceBinding binding{};
			binding.stage = execModel;
			binding.desc_set = r.set;
			binding.binding = r.binding;
			binding.msl_buffer = r.mslBuffer;
			binding.msl_texture = r.mslTexture;
			binding.msl_sampler = r.mslSampler;
			compiler.add_msl_resource_binding(binding);
		}

		// Metal shaders use the entry point name main0
		compiler.rename_entry_point("main", "main0", execModel);

		mslOut = compiler.compile();
	}
	catch (const std::exception& e)
	{
		MetalDiag_Count(MetalDiagEvent::PackShaderFallbackCross,
			"{:016x}_{:016x} ({}): SPIRV-Cross failed to generate MSL: \"{}\"", baseHash, auxHash, shaderTypeName, e.what());
		return false;
	}

	cemuLog_logDebug(LogType::Force, "MetalShaderTranslator:translated {:016x}_{:016x} ({}) to MSL", baseHash, auxHash, shaderTypeName);
	return true;
}
