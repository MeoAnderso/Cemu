#include "Cafe/HW/Latte/Renderer/Metal/RendererShaderMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalShaderTranslator.h"

#include "Cemu/Logging/CemuLogging.h"
#include "Common/precompiled.h"
#include "GameProfile/GameProfile.h"
#include "util/helpers/helpers.h"


static bool s_isLoadingShadersMtl{false};

extern std::atomic_int g_compiled_shaders_total;
extern std::atomic_int g_compiled_shaders_async;

class ShaderMtlThreadPool
{
public:
	void StartThreads()
	{
		if (m_threadsActive.exchange(true))
			return;

		// Create thread pool
		const uint32 threadCount = 2;
		for (uint32 i = 0; i < threadCount; ++i)
			s_threads.emplace_back(&ShaderMtlThreadPool::CompilerThreadFunc, this);

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

	~ShaderMtlThreadPool()
	{
		StopThreads();
	}

	void CompilerThreadFunc()
	{
		SetThreadName("mtlShaderComp");
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
			RendererShaderMtl* job = s_compilationQueue.front();
			s_compilationQueue.pop_front();
			// set compilation state
			cemu_assert_debug(job->m_compilationState.getValue() == RendererShaderMtl::COMPILATION_STATE::QUEUED);
			job->m_compilationState.setValue(RendererShaderMtl::COMPILATION_STATE::COMPILING);
			s_compilationQueueMutex.unlock();
			// compile
			job->CompileInternal();
			if (job->ShouldCountCompilation())
			    ++g_compiled_shaders_async;
			// mark as compiled
			cemu_assert_debug(job->m_compilationState.getValue() == RendererShaderMtl::COMPILATION_STATE::COMPILING);
			job->m_compilationState.setValue(RendererShaderMtl::COMPILATION_STATE::DONE);
		}
	}


	bool HasThreadsRunning() const { return m_threadsActive; }

public:
	std::vector<std::thread> s_threads;

	std::deque<RendererShaderMtl*> s_compilationQueue;
	CounterSemaphore s_compilationQueueCount;
	std::mutex s_compilationQueueMutex;


private:
	std::atomic<bool> m_threadsActive;
} shaderMtlThreadPool;

// Compiled shaders are cached by the renderer: the pipeline binary archive (see
// MetalPipelineCache::GetBinaryArchiveLookupArray) keeps the compiled functions across launches
void RendererShaderMtl::ShaderCacheLoading_begin(uint64 cacheTitleId)
{
    s_isLoadingShadersMtl = true;

    // Maximize shader compilation speed
    static_cast<MetalRenderer*>(g_renderer.get())->SetShouldMaximizeConcurrentCompilation(true);
}

void RendererShaderMtl::ShaderCacheLoading_end()
{
    s_isLoadingShadersMtl = false;

    // Reset shader compilation speed
    static_cast<MetalRenderer*>(g_renderer.get())->SetShouldMaximizeConcurrentCompilation(false);
}

void RendererShaderMtl::ShaderCacheLoading_Close()
{
	// flush and close the translated graphic pack shader cache (mirrors the Vulkan backend
	// closing s_spirvCache here)
	MetalShaderTranslator_CloseCache();

}

void RendererShaderMtl::Initialize()
{
    shaderMtlThreadPool.StartThreads();
}

void RendererShaderMtl::Shutdown()
{
    shaderMtlThreadPool.StopThreads();
}

RendererShaderMtl::RendererShaderMtl(MetalRenderer* mtlRenderer, ShaderType type, uint64 baseHash, uint64 auxHash, bool isGameShader, bool isGfxPackShader, const std::string& mslCode)
	: RendererShader(type, baseHash, auxHash, isGameShader, isGfxPackShader), m_mtlr{mtlRenderer}, m_mslCode{mslCode}
{
	// start async compilation
	shaderMtlThreadPool.s_compilationQueueMutex.lock();
	m_compilationState.setValue(COMPILATION_STATE::QUEUED);
	shaderMtlThreadPool.s_compilationQueue.push_back(this);
	shaderMtlThreadPool.s_compilationQueueCount.increment();
	shaderMtlThreadPool.s_compilationQueueMutex.unlock();
	cemu_assert_debug(shaderMtlThreadPool.HasThreadsRunning()); // make sure .StartThreads() was called
}

RendererShaderMtl::~RendererShaderMtl()
{
    if (m_function)
        m_function->release();
}

void RendererShaderMtl::PreponeCompilation(bool isRenderThread)
{
	shaderMtlThreadPool.s_compilationQueueMutex.lock();
	bool isStillQueued = m_compilationState.hasState(COMPILATION_STATE::QUEUED);
	if (isStillQueued)
	{
		// remove from queue
		shaderMtlThreadPool.s_compilationQueue.erase(std::remove(shaderMtlThreadPool.s_compilationQueue.begin(), shaderMtlThreadPool.s_compilationQueue.end(), this), shaderMtlThreadPool.s_compilationQueue.end());
		m_compilationState.setValue(COMPILATION_STATE::COMPILING);
	}
	shaderMtlThreadPool.s_compilationQueueMutex.unlock();
	if (!isStillQueued)
	{
		m_compilationState.waitUntilValue(COMPILATION_STATE::DONE);
		// count the stall only once per shader - PreponeCompilation can be called again after
		// the shader is already done (e.g. by another pipeline reusing it) and would otherwise
		// drive the counter negative
		bool expected = false;
		if (ShouldCountCompilation() && m_preponeCounted.compare_exchange_strong(expected, true))
		    --g_compiled_shaders_async; // compilation caused a stall so we don't consider this one async
		return;
	}
	else
	{
		// compile synchronously
		CompileInternal();
		m_compilationState.setValue(COMPILATION_STATE::DONE);
	}
}

bool RendererShaderMtl::IsCompiled()
{
	return m_compilationState.hasState(COMPILATION_STATE::DONE);
};

bool RendererShaderMtl::WaitForCompiled()
{
	m_compilationState.waitUntilValue(COMPILATION_STATE::DONE);
	return true;
}

bool RendererShaderMtl::ShouldCountCompilation() const
{
    return !s_isLoadingShadersMtl && m_isGameShader;
}

MTL::CompileOptions* RendererShaderMtl::NewCompileOptions(MetalRenderer* mtlr)
{
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    // Fast math stays off (Metal defaults it on): it reassociates and contracts float expressions,
    // approximates 1.0/x (RECIP_IEEE is emitted verbatim), and may fold the a==0.0 || b==0.0 guard
    // the mul_nonIEEE emulation relies on
    options->setFastMathEnabled(false);
    // Pinned rather than left to the toolchain default: the rect/geometry emulation shaders use MSL 3.0
    // features ([[mesh]], mesh<>), which only compile at language version 3.0 or later
    options->setLanguageVersion(MTL::LanguageVersion3_0);

    if (mtlr->GetPositionInvariance())
    {
        // TODO: filter out based on GPU state
        options->setPreserveInvariance(true);
    }
    return options;
}

MTL::Library* RendererShaderMtl::LibraryFromSource()
{
    // Compile from source
    NS_STACK_SCOPED MTL::CompileOptions* options = RendererShaderMtl::NewCompileOptions(m_mtlr);

    // the out-param error is autoreleased (metal-cpp does not retain it) - do not release it. Metal
    // also fills it with warnings when the compilation succeeded, so only a missing library is a failure
    NS::Error* error = nullptr;
	MTL::Library* library = m_mtlr->GetDevice()->newLibrary(ToNSString(m_mslCode), options, &error);
	if (!library)
    {
        cemuLog_log(LogType::Force, "failed to create library from source: {} -> {}", error ? error->localizedDescription()->utf8String() : "unknown error", m_mslCode.c_str());
        m_hasError = true;
        return nullptr;
    }
    if (error && cemuLog_isLoggingEnabled(LogType::MetalBackend))
        cemuLog_log(LogType::MetalBackend, "shader {:016x}_{:016x} compiled with warnings: {}", m_baseHash, m_auxHash, error->localizedDescription()->utf8String());

    return library;
}


void RendererShaderMtl::CompileInternal()
{
    // runs on the shader-compile threads (and synchronously via PreponeCompilation), which have no
    // autorelease pool: drain what compilation autoreleases (the source NSString, NSError) per shader
    NS_STACK_SCOPED NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();

    MTL::Library* library = LibraryFromSource();
    FinishCompilation();
    if (!library)
        return;

    m_function = library->newFunction(ToNSString("main0"));
    if (!m_function)
        m_hasError = true; // pipeline compiler fails the pipeline early via HasError()
    library->release();

	// Count shader compilation
	if (ShouldCountCompilation())
	    g_compiled_shaders_total++;
}

void RendererShaderMtl::FinishCompilation()
{
    m_mslCode.clear();
    m_mslCode.shrink_to_fit();
}
