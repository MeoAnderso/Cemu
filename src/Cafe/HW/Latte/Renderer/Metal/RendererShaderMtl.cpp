#include "Cafe/HW/Latte/Renderer/Metal/RendererShaderMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalShaderTranslator.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"

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
    // the stripped-variant functions each retain their compiled library - releasing them here
    // prevents an accumulation across shader-cache evictions
    for (auto& [key, function] : m_strippedFunctionCache)
    {
        if (function)
            function->release();
    }
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


MTL::Function* RendererShaderMtl::GetStrippedVariant(uint32 removedColorMask, bool removeDepth)
{
	// bit 8 of the cache key signals depth removal (color masks only use bits 0..7)
	const uint32 cacheKey = removedColorMask | (removeDepth ? 0x100 : 0);
	std::lock_guard<std::mutex> lock(m_variantMutex);
	auto it = m_strippedFunctionCache.find(cacheKey);
	if (it != m_strippedFunctionCache.end())
		return it->second;

	// Only rewrite sources this project's decompiler emitted (see kMslDecompilerSourceMarker). A
	// translated graphic-pack shader is SPIRV-Cross output: its members carry the same
	// passPixelColorN names, so the member scan below matches and removes them from a struct it does
	// not own, but the "FragmentOut out;" anchor and the "struct FragmentOut {" check both miss -
	// so the return statement of a function that still returns main0_out gets rewritten to "return;"
	// and the result cannot compile. Refuse up front instead, and let the diagnostics say that this
	// shader cannot be served on a pass it has orphan outputs for
	if (m_mslCode.find(kMslDecompilerSourceMarker) == std::string::npos)
	{
		MetalDiag_Count(MetalDiagEvent::OrphanOutputStripUnsupported,
			"{:016x}_{:016x} (colorMask {:05x}, depth {})", m_baseHash, m_auxHash, removedColorMask, removeDepth ? 1 : 0);
		return nullptr;
	}

	// remove the FragmentOut members for the orphaned outputs from the MSL source. The emitter
	// writes color outputs as "<type> passPixelColor<N> [[color(N)]];" and the depth output as
	// "float passDepth [[depth(any)]];". Metal requires every member of a fragment return
	// struct to carry an output attribute, so simply stripping the attribute is not enough -
	// the member must be removed from the struct entirely. The body's out.passPixelColor<N> /
	// out.passDepth references are redirected to dummy locals so the assignments (and any
	// reads) still compile with identical semantics
	struct RemovedMember
	{
		std::string type;
		uint32 index;
	};
	std::vector<RemovedMember> removedMembers;
	bool depthMemberRemoved = false;
	std::string patchedSource;
	patchedSource.reserve(m_mslCode.size());
	size_t lineStart = 0;
	while (lineStart <= m_mslCode.size())
	{
		size_t lineEnd = m_mslCode.find('\n', lineStart);
		if (lineEnd == std::string::npos)
			lineEnd = m_mslCode.size();
		std::string_view line((const char*)m_mslCode.data() + lineStart, lineEnd - lineStart);
		bool memberRemoved = false;
		for (uint32 i = 0; i < 8; i++)
		{
			if ((removedColorMask & (1u << i)) == 0)
				continue;
			if (line.find(fmt::format("passPixelColor{} [[color({})]];", i, i)) != std::string_view::npos)
			{
				// capture the member type (first token of the line) for the dummy declaration
				size_t typeEnd = line.find_first_of(" \t");
				removedMembers.push_back({ std::string(line.substr(0, typeEnd)), i });
				memberRemoved = true; // one member per line, no further matching needed
				break;
			}
		}
		if (!memberRemoved && removeDepth && line.find("float passDepth [[depth(any)]];") != std::string_view::npos)
		{
			depthMemberRemoved = true;
			memberRemoved = true;
		}
		if (!memberRemoved)
			patchedSource.append(line);
		if (lineEnd == m_mslCode.size())
			break;
		patchedSource.push_back('\n');
		lineStart = lineEnd + 1;
	}

	if (removedMembers.empty() && !depthMemberRemoved)
	{
		// nothing matched - this source is not decompiled MSL (e.g. a translated graphic-pack
		// shader with different symbol names), so stripping is not applicable. Return nullptr
		// without caching: the text scan is cheap and the caller falls back to the original
		// function
		return nullptr;
	}

	if (!removedMembers.empty())
	{
		for (const auto& member : removedMembers)
		{
			const std::string ref = fmt::format("out.passPixelColor{}", member.index);
			const std::string dummy = fmt::format("orphanColor{}", member.index);
			size_t refPos;
			while ((refPos = patchedSource.find(ref)) != std::string::npos)
				patchedSource.replace(refPos, ref.size(), dummy);
		}
	}
	if (depthMemberRemoved)
	{
		const std::string ref = "out.passDepth";
		const std::string dummy = "orphanDepth";
		size_t refPos;
		while ((refPos = patchedSource.find(ref)) != std::string::npos)
			patchedSource.replace(refPos, ref.size(), dummy);
	}
	// declare the dummies right after the local FragmentOut variable in main0, which the
	// emitter always emits for pixel shaders; all out.* uses come after it
	{
		std::string dummies;
		for (const auto& member : removedMembers)
			dummies += fmt::format("\n\t{} orphanColor{};", member.type, member.index);
		if (depthMemberRemoved)
			dummies += "\n\tfloat orphanDepth;";
		if (!dummies.empty())
		{
			const std::string outDecl = "FragmentOut out;";
			size_t declPos = patchedSource.find(outDecl);
			if (declPos != std::string::npos)
				patchedSource.insert(declPos + outDecl.size(), dummies);
		}
	}

	// if the FragmentOut struct no longer carries any color or depth output, Metal rejects it
	// as a fragment return type ("invalid return type"). Switch to a void-returning function;
	// the local FragmentOut variable and its assignments stay (harmless dead stores). The
	// check is scoped to the struct because [[color(N)]] may legitimately remain elsewhere in
	// the source (e.g. a framebuffer-fetch input argument)
	{
		bool structHasOutputs = false;
		size_t structPos = patchedSource.find("struct FragmentOut {");
		if (structPos != std::string::npos)
		{
			size_t structEnd = patchedSource.find("\n};", structPos);
			if (structEnd == std::string::npos)
				structEnd = patchedSource.size();
			std::string_view structBody((const char*)patchedSource.data() + structPos, structEnd - structPos);
			structHasOutputs = structBody.find("[[color(") != std::string_view::npos || structBody.find("[[depth") != std::string_view::npos;
		}
		if (!structHasOutputs)
		{
			const std::string sigOld = "FragmentOut main0(";
			size_t sigPos = patchedSource.find(sigOld);
			if (sigPos != std::string::npos)
				patchedSource.replace(sigPos, sigOld.size(), "void main0(");
			const std::string retOld = "return out;";
			size_t retPos;
			while ((retPos = patchedSource.find(retOld)) != std::string::npos)
				patchedSource.replace(retPos, retOld.size(), "return;");
		}
	}

	// compile the variant with the same options as the primary compilation. The explicit
	// autorelease pool is required: this may run on pipeline-compile threads which have no
	// ObjC autorelease pool (an imbalance crashes at thread exit)
	MTL::Function* function = nullptr;
	{
		NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();
		NS_STACK_SCOPED MTL::CompileOptions* options = RendererShaderMtl::NewCompileOptions(m_mtlr);

		NS::Error* error = nullptr;
		MTL::Library* library = m_mtlr->GetDevice()->newLibrary(ToNSString(patchedSource), options, &error);
		if (!library)
		{
			cemuLog_log(LogType::Force, "failed to compile output-stripped fragment variant of shader {:016x}_{:016x} (colorMask {:05x} depth {}) : {}",
				m_baseHash, m_auxHash, removedColorMask, removeDepth ? 1 : 0, error ? error->localizedDescription()->utf8String() : "unknown error");
			// note: the out-param error is autoreleased (metal-cpp does not retain it) - do not
			// release it here, the autorelease pool owns its lifetime
			// cache the failure so a broken variant is not recompiled on every draw
			m_strippedFunctionCache.emplace(cacheKey, nullptr);
			autoreleasePool->release();
			return nullptr;
		}

		function = library->newFunction(ToNSString("main0"));
		library->release(); // the function retains the library
		autoreleasePool->release();
	}
	m_strippedFunctionCache.emplace(cacheKey, function);
	return function;
}

void RendererShaderMtl::FinishCompilation()
{
    // retain the MSL source of pixel shaders: the pipeline compiler may need stripped-output
    // variants of the fragment function (see GetStrippedVariant). Vertex and
    // geometry shaders have no color outputs - free their source
    if (m_type != ShaderType::kFragment)
    {
        m_mslCode.clear();
        m_mslCode.shrink_to_fit();
    }
}
