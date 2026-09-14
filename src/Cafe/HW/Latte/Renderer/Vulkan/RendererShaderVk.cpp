#include "Cafe/HW/Latte/Renderer/Vulkan/RendererShaderVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "Cafe/HW/Latte/Renderer/SpirvCompiler.h"
#include "config/ActiveSettings.h"
#include "config/CemuConfig.h"
#include "util/helpers/ConcurrentQueue.h"
#include "Cemu/FileCache/FileCache.h"

#include "util/helpers/helpers.h"

bool s_isLoadingShadersVk{ false };
class FileCache* s_spirvCache{nullptr};

extern std::atomic_int g_compiled_shaders_total;
extern std::atomic_int g_compiled_shaders_async;

class _ShaderVkThreadPool
{
public:
	void StartThreads()
	{
		if (m_threadsActive.exchange(true))
			return;
		// create thread pool
		const uint32 threadCount = 2;
		for (uint32 i = 0; i < threadCount; ++i)
			s_threads.emplace_back(&_ShaderVkThreadPool::CompilerThreadFunc, this);
	}

	void StopThreads()
	{
		if (!m_threadsActive.exchange(false))
			return;
		for (uint32 i = 0; i < s_threads.size(); ++i)
			s_compilationQueueCount.increment();
		for (auto& it : s_threads)
			it.join();
		s_threads.clear();
	}

	~_ShaderVkThreadPool()
	{
		StopThreads();
	}

	void CompilerThreadFunc()
	{
		SetThreadName("vkShaderComp");
		while (m_threadsActive.load(std::memory_order::relaxed))
		{
			s_compilationQueueCount.decrementWithWait();
			s_compilationQueueMutex.lock();
			if (s_compilationQueue.empty())
			{
				// queue empty again, shaders compiled synchronously via PreponeCompilation()
				s_compilationQueueMutex.unlock();
				continue;
			}
			RendererShaderVk* job = s_compilationQueue.front();
			s_compilationQueue.pop_front();
			// set compilation state
			cemu_assert_debug(job->m_compilationState.getValue() == RendererShaderVk::COMPILATION_STATE::QUEUED);
			job->m_compilationState.setValue(RendererShaderVk::COMPILATION_STATE::COMPILING);
			s_compilationQueueMutex.unlock();
			// compile
			job->CompileInternal(false);
			++g_compiled_shaders_async;
			// mark as compiled
			cemu_assert_debug(job->m_compilationState.getValue() == RendererShaderVk::COMPILATION_STATE::COMPILING);
			job->m_compilationState.setValue(RendererShaderVk::COMPILATION_STATE::DONE);
		}
	}

	bool HasThreadsRunning() const { return m_threadsActive; }

public:
	std::vector<std::thread> s_threads;

	std::deque<RendererShaderVk*> s_compilationQueue;
	CounterSemaphore s_compilationQueueCount;
	std::mutex s_compilationQueueMutex;

private:
	std::atomic<bool> m_threadsActive;
}ShaderVkThreadPool;

RendererShaderVk::RendererShaderVk(ShaderType type, uint64 baseHash, uint64 auxHash, bool isGameShader, bool isGfxPackShader, const std::string& glslCode)
	: RendererShader(type, baseHash, auxHash, isGameShader, isGfxPackShader), m_glslCode(glslCode)
{
	// start async compilation
	ShaderVkThreadPool.s_compilationQueueMutex.lock();
	m_compilationState.setValue(COMPILATION_STATE::QUEUED);
	ShaderVkThreadPool.s_compilationQueue.push_back(this);
	ShaderVkThreadPool.s_compilationQueueCount.increment();
	ShaderVkThreadPool.s_compilationQueueMutex.unlock();
	cemu_assert_debug(ShaderVkThreadPool.HasThreadsRunning()); // make sure .StartThreads() was called
}

RendererShaderVk::~RendererShaderVk()
{
	while (!list_pipelineInfo.empty())
		delete list_pipelineInfo[0];

	VkDevice vkDev = VulkanRenderer::GetInstance()->GetLogicalDevice();
	vkDestroyShaderModule(vkDev, m_shader_module, nullptr);
}

void RendererShaderVk::Init()
{
	ShaderVkThreadPool.StartThreads();
}

void RendererShaderVk::Shutdown()
{
	ShaderVkThreadPool.StopThreads();
}

void RendererShaderVk::CreateVkShaderModule(std::span<uint32> spirvBuffer)
{
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = spirvBuffer.size_bytes();
	createInfo.pCode = spirvBuffer.data();

	VulkanRenderer* vkr = (VulkanRenderer*)g_renderer.get();

	VkDevice m_device = vkr->GetLogicalDevice();

	VkResult result = vkCreateShaderModule(m_device, &createInfo, nullptr, &m_shader_module);
	if (result != VK_SUCCESS)
	{
		cemuLog_log(LogType::Force, "Vulkan: Shader error");
		throw std::runtime_error(fmt::format("Failed to create shader module: {}", result));
	}

	// set debug name
	if (vkr->IsDebugMarkersEnabled())
	{
		VkDebugUtilsObjectNameInfoEXT objName{};
		objName.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
		objName.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
		objName.pNext = nullptr;
		objName.objectHandle = (uint64_t)m_shader_module;
		auto objNameStr = fmt::format("shader_{:016x}_{:016x}", m_baseHash, m_auxHash);
		objName.pObjectName = objNameStr.c_str();
		vkSetDebugUtilsObjectNameEXT(vkr->GetLogicalDevice(), &objName);
	}
}

void RendererShaderVk::FinishCompilation()
{
	m_glslCode.clear();
	m_glslCode.shrink_to_fit();
}

void RendererShaderVk::CompileInternal(bool isRenderThread)
{
	const bool compileWithDebugInfo = ((VulkanRenderer*)g_renderer.get())->IsTracingToolEnabled();

	// try to retrieve SPIR-V module from cache
	if (s_isLoadingShadersVk && (m_isGameShader && !m_isGfxPackShader) && s_spirvCache && !compileWithDebugInfo)
	{
		cemu_assert_debug(m_baseHash != 0);
		uint64 h1, h2;
		GenerateShaderPrecompiledCacheFilename(m_type, m_baseHash, m_auxHash, h1, h2);
		std::vector<uint8> cacheFileData;
		if (s_spirvCache->GetFile({ h1, h2 }, cacheFileData))
		{
			// generate shader from cached SPIR-V buffer
			CreateVkShaderModule(std::span<uint32>((uint32*)cacheFileData.data(), cacheFileData.size() / sizeof(uint32)));
			FinishCompilation();
			return;
		}
	}

	// compile GLSL to SPIR-V via the shared SpirvCompiler front-end
	std::vector<uint32> spirvBuffer;
	if (!SpirvCompiler_Compile(m_glslCode, GetType(), m_baseHash, m_auxHash, compileWithDebugInfo, spirvBuffer))
	{
		FinishCompilation();
		return;
	}

	// store in cache, unless it got compiled with debug info or is a modified shader from a gfx pack
	if (s_spirvCache && m_isGameShader && m_isGfxPackShader == false && !compileWithDebugInfo)
	{
		uint64 h1, h2;
		GenerateShaderPrecompiledCacheFilename(m_type, m_baseHash, m_auxHash, h1, h2);
		s_spirvCache->AddFile({ h1, h2 }, (const uint8*)spirvBuffer.data(), spirvBuffer.size() * sizeof(uint32));
	}

	CreateVkShaderModule(spirvBuffer);

	// count compiled shader
	if (!s_isLoadingShadersVk)
	{
		if( m_isGameShader )
			++g_compiled_shaders_total;
	}

	FinishCompilation();
}

void RendererShaderVk::PreponeCompilation(bool isRenderThread)
{
	ShaderVkThreadPool.s_compilationQueueMutex.lock();
	bool isStillQueued = m_compilationState.hasState(COMPILATION_STATE::QUEUED);
	if (isStillQueued)
	{
		// remove from queue
		ShaderVkThreadPool.s_compilationQueue.erase(std::remove(ShaderVkThreadPool.s_compilationQueue.begin(), ShaderVkThreadPool.s_compilationQueue.end(), this), ShaderVkThreadPool.s_compilationQueue.end());
		m_compilationState.setValue(COMPILATION_STATE::COMPILING);
	}
	ShaderVkThreadPool.s_compilationQueueMutex.unlock();
	if (!isStillQueued)
	{
		m_compilationState.waitUntilValue(COMPILATION_STATE::DONE);
		--g_compiled_shaders_async; // compilation caused a stall so we don't consider this one async
		return;
	}
	else
	{
		// compile synchronously
		CompileInternal(isRenderThread);
		m_compilationState.setValue(COMPILATION_STATE::DONE);
	}
}

bool RendererShaderVk::IsCompiled()
{
	return m_compilationState.hasState(COMPILATION_STATE::DONE);
};

bool RendererShaderVk::WaitForCompiled()
{
	m_compilationState.waitUntilValue(COMPILATION_STATE::DONE);
	return true;
}

void RendererShaderVk::ShaderCacheLoading_begin(uint64 cacheTitleId)
{
	if (s_spirvCache)
	{
		delete s_spirvCache;
		s_spirvCache = nullptr;
	}
	uint32 spirvCacheMagic = GeneratePrecompiledCacheId();
	const std::string cacheFilename = fmt::format("{:016x}_spirv.bin", cacheTitleId);
	const fs::path cachePath = ActiveSettings::GetCachePath("shaderCache/precompiled/{}", cacheFilename);
	s_spirvCache = FileCache::Open(cachePath, true, spirvCacheMagic);
	if (s_spirvCache == nullptr)
		cemuLog_log(LogType::Force, "Unable to open SPIR-V cache {}", cacheFilename);
	s_isLoadingShadersVk = true;
}

void RendererShaderVk::ShaderCacheLoading_end()
{
	// keep g_spirvCache open since we will write to it while the game is running
	s_isLoadingShadersVk = false;
}

void RendererShaderVk::ShaderCacheLoading_Close()
{
    delete s_spirvCache;
    s_spirvCache = nullptr;
}
