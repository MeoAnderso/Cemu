#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCompiler.h"

#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Common/RegisterSerializer.h"
#include "Cafe/HW/Latte/Core/LatteShaderCache.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/ISA/LatteReg.h"
#include "Cemu/FileCache/FileCache.h"
#include "Common/precompiled.h"
#include "util/helpers/helpers.h"
#include "config/ActiveSettings.h"

#include <openssl/sha.h>

static bool g_compilePipelineThreadInit{false};

// Guards the lifetime of s_cache. It is opened on the loader thread (BeginLoading), read on the
// loader thread, and used by the detached cache-writer thread (WorkerThread), while Close() runs on
// the emulation thread at title exit. Without this, Close() could delete the FileCache between the
// writer's null check and its AddFileAsync call (a use-after-free), and the same window exists for
// the loader. Every check-then-use pair must therefore sit inside one critical section
static std::mutex s_fileCacheMutex;
static std::mutex g_compilePipelineMutex;
static std::condition_variable g_compilePipelineCondVar;
static std::queue<MetalPipelineCompiler*> g_compilePipelineRequests;

static void compileThreadFunc(sint32 threadIndex)
{
	SetThreadName("compilePl");

	// one thread runs at normal priority while the others run at lower priority
	if (threadIndex != 0)
		; // TODO: set thread priority

	while (true)
	{
		std::unique_lock lock(g_compilePipelineMutex);
		while (g_compilePipelineRequests.empty())
			g_compilePipelineCondVar.wait(lock);

		MetalPipelineCompiler* request = g_compilePipelineRequests.front();

		g_compilePipelineRequests.pop();

		lock.unlock();

		request->Compile(true, false, true);
		delete request;
	}
}

static void initCompileThread()
{
	uint32 numCompileThreads;

	uint32 cpuCoreCount = GetPhysicalCoreCount();
	if (cpuCoreCount <= 2)
		numCompileThreads = 1;
	else
		numCompileThreads = 2 + (cpuCoreCount - 3); // 2 plus one additionally for every extra core above 3

	numCompileThreads = std::min(numCompileThreads, 8u); // cap at 8

	for (uint32 i = 0; i < numCompileThreads; i++)
	{
		std::thread compileThread(compileThreadFunc, i);
		compileThread.detach();
	}
}

static void queuePipeline(MetalPipelineCompiler* v)
{
	std::unique_lock lock(g_compilePipelineMutex);
	g_compilePipelineRequests.push(std::move(v));
	lock.unlock();
	g_compilePipelineCondVar.notify_one();
}

// make a guess if a pipeline is not essential
// non-essential means that skipping these drawcalls shouldn't lead to permanently corrupted graphics
bool IsAsyncPipelineAllowed(const MetalAttachmentsInfo& attachmentsInfo, Vector2i extend, uint32 indexCount)
{
	// frame debuggers/GPU captures don't handle async-compiled (initially skipped) draws well -
	// the draw would be missing from the capture (same guard as the Vulkan backend)
	if (static_cast<MetalRenderer*>(g_renderer.get())->IsTracingToolEnabled())
		return false;

	if (extend.x == 1600 && extend.y == 1600)
		return false; // Splatoon ink mechanics use 1600x1600 R8 and R8G8 framebuffers, this resolution is rare enough that we can just blacklist it globally

	if (attachmentsInfo.depthFormat != Latte::E_GX2SURFFMT::INVALID_FORMAT)
		return true; // aggressive filter but seems to work well so far

	// small index count (3,4,5,6) is often associated with full-viewport quads (which are considered essential due to often being used to generate persistent textures)
	if (indexCount <= 6)
		return false;

	return true;
}

MetalPipelineCache* g_mtlPipelineCache = nullptr;

MetalPipelineCache& MetalPipelineCache::GetInstance()
{
    return *g_mtlPipelineCache;
}

MetalPipelineCache::MetalPipelineCache(class MetalRenderer* metalRenderer) : m_mtlr{metalRenderer}
{
    g_mtlPipelineCache = this;
}

MetalPipelineCache::~MetalPipelineCache()
{
    // the writer thread calls back into this object (SerializePipeline), so it has to be gone before
    // anything below runs
    StopStoreThread();

    // drop whatever it had not picked up yet - those jobs are heap objects owned by nobody else
    DiscardPendingJobs();

    m_pipelineCacheLock.lock();
    for (auto& [key, pipelineObj] : m_pipelineCache)
    {
        if (pipelineObj->m_pipeline)
            pipelineObj->m_pipeline.load()->release();
        delete pipelineObj;
    }
    m_pipelineCache.clear();
    m_pipelineCacheLock.unlock();
}


PipelineObject* MetalPipelineCache::GetRenderPipelineState(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, Vector2i extend, uint32 indexCount, const LatteContextRegister& lcr)
{
    uint64 hash = CalculatePipelineHash(fetchShader, vertexShader, geometryShader, pixelShader, lastUsedAttachmentsInfo, activeAttachmentsInfo, lcr);
    m_pipelineCacheLock.lock();
    auto it = m_pipelineCache.find(hash);
    PipelineObject* pipelineObj = (it != m_pipelineCache.end()) ? it->second : nullptr;
    if (pipelineObj && pipelineObj->compileFailed && !pipelineObj->permanentFailure)
    {
        // evict transiently failed compilations so this call retries instead of skipping draws
        // forever (permanent failures stay cached: retrying them would fail identically per draw)
        m_pipelineCache.erase(it);
        delete pipelineObj;
        pipelineObj = nullptr;
    }
    if (pipelineObj)
    {
        // still compiling asynchronously (or permanently unsupported)
        bool persistNow = pipelineObj->persistWhenCompiled && pipelineObj->m_pipeline != nullptr && !pipelineObj->compileFailed;
        if (persistNow)
            pipelineObj->persistWhenCompiled = false;
        m_pipelineCacheLock.unlock();
        // asynchronous compile finished successfully and wasn't persisted yet: AddCurrentStateToCache
        // snapshots the active register/shader state, so it must run here on the render thread while
        // the state that produced this pipeline is current
        if (persistNow)
            AddCurrentStateToCache(hash, lastUsedAttachmentsInfo);
        return pipelineObj;
    }
    pipelineObj = new PipelineObject();
    m_pipelineCache[hash] = pipelineObj;
    m_pipelineCacheLock.unlock();

    MetalPipelineCompiler* compiler = new MetalPipelineCompiler(m_mtlr, *pipelineObj);
    compiler->InitFromState(fetchShader, vertexShader, geometryShader, pixelShader, lastUsedAttachmentsInfo, activeAttachmentsInfo, lcr);

    bool allowAsyncCompile = false;
    if (GetConfig().async_compile)
		allowAsyncCompile = IsAsyncPipelineAllowed(activeAttachmentsInfo, extend, indexCount);

    bool compileSucceeded;
	if (allowAsyncCompile)
	{
	    if (!g_compilePipelineThreadInit)
		{
			initCompileThread();
			g_compilePipelineThreadInit = true;
		}

		queuePipeline(compiler);
		// the result is determined by the compile thread (which sets compileFailed on error), so
		// persistence is deferred until the next render-thread request observes the outcome
		pipelineObj->persistWhenCompiled = true;
		compileSucceeded = false;
	}
	else
	{
	    // Also force compile to ensure that the pipeline is ready
        compileSucceeded = compiler->Compile(true, true, true);
        if (!compileSucceeded)
            cemuLog_log(LogType::Force, "failed to compile render pipeline synchronously");
        delete compiler;
	}

	// Save to cache (failed compilations are not persisted, they will be re-compiled from the live state)
    if (compileSucceeded)
        AddCurrentStateToCache(hash, lastUsedAttachmentsInfo);

    return pipelineObj;
}

uint64 MetalPipelineCache::CalculatePipelineHash(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr)
{
    // Hash
    uint64 stateHash = 0;
    for (int i = 0; i < Latte::GPU_LIMITS::NUM_COLOR_ATTACHMENTS; ++i)
	{
	    Latte::E_GX2SURFFMT format = lastUsedAttachmentsInfo.colorFormats[i];
		if (format == Latte::E_GX2SURFFMT::INVALID_FORMAT)
            continue;

		stateHash += GetMtlPixelFormat(format, false) + i * 31;
		stateHash = std::rotl<uint64>(stateHash, 7);

		if (activeAttachmentsInfo.colorFormats[i] == Latte::E_GX2SURFFMT::INVALID_FORMAT)
		{
            stateHash += 1;
		    stateHash = std::rotl<uint64>(stateHash, 1);
		}
	}

	if (lastUsedAttachmentsInfo.depthFormat != Latte::E_GX2SURFFMT::INVALID_FORMAT)
	{
		stateHash += GetMtlPixelFormat(lastUsedAttachmentsInfo.depthFormat, true);
		stateHash = std::rotl<uint64>(stateHash, 7);

		if (lastUsedAttachmentsInfo.hasStencil)
			stateHash += 1;
		stateHash = std::rotl<uint64>(stateHash, 1);

		if (activeAttachmentsInfo.depthFormat == Latte::E_GX2SURFFMT::INVALID_FORMAT)
		{
            stateHash += 1;
		    stateHash = std::rotl<uint64>(stateHash, 1);
		}
	}

	for (auto& group : fetchShader->bufferGroups)
	{
		uint32 bufferStride = group.getCurrentBufferStride(lcr.GetRawView());
		stateHash = std::rotl<uint64>(stateHash, 7);
		stateHash += bufferStride * 3;
	}

	stateHash += fetchShader->getVkPipelineHashFragment();
	stateHash = std::rotl<uint64>(stateHash, 7);

	stateHash += lcr.GetRawView()[mmVGT_STRMOUT_EN];
	stateHash = std::rotl<uint64>(stateHash, 7);

	if(lcr.PA_CL_CLIP_CNTL.get_DX_RASTERIZATION_KILL())
		stateHash += 0x333333;

	stateHash = (stateHash >> 8) + (stateHash * 0x370531ull) % 0x7F980D3BF9B4639Dull;

	uint32* ctxRegister = lcr.GetRawView();

	// auxHash as well as baseHash: the MSL emitter bakes the sampler LOD bias into the emitted
	// shader, so two vertex shader variants differing only in auxHash generate different MSL and
	// must not share one cached pipeline
	if (vertexShader)
		stateHash += vertexShader->baseHash + vertexShader->auxHash;

	stateHash = std::rotl<uint64>(stateHash, 13);

	if (geometryShader)
		stateHash += geometryShader->baseHash + geometryShader->auxHash; // the GS is baked into the pipeline as the mesh function

	stateHash = std::rotl<uint64>(stateHash, 13);

	if (pixelShader)
		stateHash += pixelShader->baseHash + pixelShader->auxHash;

	stateHash = std::rotl<uint64>(stateHash, 13);

	uint32 polygonCtrl = lcr.PA_SU_SC_MODE_CNTL.getRawValue();
	stateHash += polygonCtrl;
	stateHash = std::rotl<uint64>(stateHash, 7);

	stateHash += ctxRegister[Latte::REGADDR::PA_CL_CLIP_CNTL];
	stateHash = std::rotl<uint64>(stateHash, 7);

	// toggling VPORT_X_OFFSET_ENA changes IsRasterizationEnabled() and with it the pipeline shape,
	// so the register must be part of the key
	stateHash += ctxRegister[Latte::REGADDR::PA_CL_VTE_CNTL];
	stateHash = std::rotl<uint64>(stateHash, 7);

	const auto colorControlReg = ctxRegister[Latte::REGADDR::CB_COLOR_CONTROL];
	stateHash += colorControlReg;

	stateHash += ctxRegister[Latte::REGADDR::CB_TARGET_MASK];

	const uint32 blendEnableMask = (colorControlReg >> 8) & 0xFF;
	if (blendEnableMask)
	{
		for (auto i = 0; i < 8; ++i)
		{
			if (((blendEnableMask & (1 << i))) == 0)
				continue;
			stateHash = std::rotl<uint64>(stateHash, 7);
			stateHash += ctxRegister[Latte::REGADDR::CB_BLEND0_CONTROL + i];
		}
	}

	// Mesh pipeline
	// read the primitive mode from the passed context - on the pipeline cache loader thread the
	// global contextRegister holds whatever the render thread is currently doing
	const LattePrimitiveMode primitiveMode = static_cast<LattePrimitiveMode>(lcr.GetRawView()[mmVGT_PRIMITIVE_TYPE]);
    bool isPrimitiveRect = (primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::RECTS);

    bool usesGeometryShader = (geometryShader != nullptr || isPrimitiveRect);

    if (usesGeometryShader)
    {
        stateHash += lcr.GetRawView()[mmVGT_PRIMITIVE_TYPE];
        stateHash = std::rotl<uint64>(stateHash, 7);
    }

	return stateHash;
}

struct
{
	uint32 pipelineLoadIndex;
	uint32 pipelineMaxFileIndex;

	std::atomic_uint32_t pipelinesQueued;
	std::atomic_uint32_t pipelinesLoaded;
} g_mtlCacheState;

uint32 MetalPipelineCache::BeginLoading(uint64 cacheTitleId)
{
	std::error_code ec;
	fs::create_directories(ActiveSettings::GetCachePath("shaderCache/transferable"), ec);
	const auto pathCacheFile = ActiveSettings::GetCachePath("shaderCache/transferable/{:016x}_mtlpipeline.bin", cacheTitleId);

	// init cache loader state
	g_mtlCacheState.pipelineLoadIndex = 0;
	g_mtlCacheState.pipelineMaxFileIndex = 0;
	g_mtlCacheState.pipelinesLoaded = 0;
	g_mtlCacheState.pipelinesQueued = 0;

	m_compilationQueue.clear();

	// get core count
	uint32 cpuCoreCount = GetPhysicalCoreCount();
	uint32 numCompilationThreads = std::clamp(cpuCoreCount, 1u, 8u);
	// TODO: uncomment?
	//if (VulkanRenderer::GetInstance()->GetDisableMultithreadedCompilation())
	//	numCompilationThreads = 1;

	// open cache file or create it. This happens BEFORE the compile threads are spawned on purpose:
	// with nothing to load there is nothing for them to do, and a thread spawned here would block on
	// the empty queue until EndLoading pushes a shutdown token - which on this path is only reached
	// after the progress loop, and that loop dereferences the (still null) cache - see UpdateLoading
	uint32 fileCount = 0;
	cemu_assert_debug(s_cache == nullptr);
	{
		// same critical section as the writer thread and Close() - see s_fileCacheMutex
		std::lock_guard<std::mutex> lock(s_fileCacheMutex);
		s_cache = FileCache::Open(pathCacheFile, true, LatteShaderCache_getPipelineCacheExtraVersion(cacheTitleId));
		if (!s_cache)
		{
			cemuLog_log(LogType::Force, "Failed to open or create Metal pipeline cache file: {}", _pathToUtf8(pathCacheFile));
			return 0;
		}
		s_cache->UseCompression(false);
		g_mtlCacheState.pipelineMaxFileIndex = s_cache->GetMaximumFileIndex();
		fileCount = s_cache->GetFileCount();
	}

	// start async compilation threads
	m_numCompilationThreads = numCompilationThreads;
	for (uint32 i = 0; i < m_numCompilationThreads; i++)
	{
		std::thread compileThread(&MetalPipelineCache::CompilerThread, this);
		compileThread.detach();
	}
	return fileCount;
}

bool MetalPipelineCache::UpdateLoading(uint32& pipelinesLoadedTotal, uint32& pipelinesMissingShaders)
{
	pipelinesLoadedTotal = g_mtlCacheState.pipelinesLoaded;
	pipelinesMissingShaders = 0;
	while (g_mtlCacheState.pipelineLoadIndex <= g_mtlCacheState.pipelineMaxFileIndex)
	{
		if (m_compilationQueue.size() >= 50)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			return true; // queue up to 50 entries at a time
		}

		uint64 fileNameA, fileNameB;
		std::vector<uint8> fileData;
		bool fileFound = false;
		{
			// same critical section as the writer thread and Close() - the cache pointer must not be
			// deleted between the check and the use (see s_fileCacheMutex). A null cache means either
			// the open failed or the title was exited; either way there is nothing left to load
			std::lock_guard<std::mutex> lock(s_fileCacheMutex);
			if (!s_cache)
				return false;
			fileFound = s_cache->GetFileByIndex(g_mtlCacheState.pipelineLoadIndex, &fileNameA, &fileNameB, fileData);
		}
		if (fileFound)
		{
			// queue for async compilation
			g_mtlCacheState.pipelinesQueued++;
			m_compilationQueue.push(std::move(fileData));
			g_mtlCacheState.pipelineLoadIndex++;
			return true;
		}
		g_mtlCacheState.pipelineLoadIndex++;
	}
	if (g_mtlCacheState.pipelinesLoaded != g_mtlCacheState.pipelinesQueued)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		return true; // pipelines still compiling
	}
	return false; // done
}

void MetalPipelineCache::EndLoading()
{
	// shut down compilation threads
	uint32 threadCount = m_numCompilationThreads;
	m_numCompilationThreads = 0; // signal thread shutdown
	for (uint32 i = 0; i < threadCount; i++)
	{
		m_compilationQueue.push({}); // push empty workload for every thread. Threads then will shutdown after checking for m_numCompilationThreads == 0
	}
	// keep cache file open for writing of new pipelines
}

void MetalPipelineCache::Close()
{
    // the mutex is what makes the writer/loader threads' null checks meaningful - see s_fileCacheMutex
    std::lock_guard<std::mutex> lock(s_fileCacheMutex);
    if(s_cache)
    {
        delete s_cache;
        s_cache = nullptr;
    }
}

struct CachedPipeline
{
	struct ShaderHash
	{
		uint64 baseHash;
		uint64 auxHash;
		bool isPresent{};

		void set(uint64 baseHash, uint64 auxHash)
		{
			this->baseHash = baseHash;
			this->auxHash = auxHash;
			this->isPresent = true;
		}
	};

	ShaderHash vsHash; // includes fetch shader
	ShaderHash gsHash;
	ShaderHash psHash;

	MetalAttachmentsInfo lastUsedAttachmentsInfo;

	Latte::GPUCompactedRegisterState gpuState;
};

void MetalPipelineCache::LoadPipelineFromCache(std::span<uint8> fileData)
{
	// deserialize file
	auto cachedPipeline = std::make_unique<CachedPipeline>();
	MemStreamReader streamReader(fileData.data(), fileData.size());
	if (!DeserializePipeline(streamReader, *cachedPipeline))
		return; // failed to deserialize

	// restored register view from compacted state
	auto lcr = std::make_unique<LatteContextRegister>();
	Latte::LoadGPURegisterState(*lcr, cachedPipeline->gpuState);

	LatteDecompilerShader* vertexShader = nullptr;
	LatteDecompilerShader* geometryShader = nullptr;
	LatteDecompilerShader* pixelShader = nullptr;
	// find vertex shader
	if (cachedPipeline->vsHash.isPresent)
	{
		vertexShader = LatteSHRC_FindVertexShader(cachedPipeline->vsHash.baseHash, cachedPipeline->vsHash.auxHash);
		if (!vertexShader)
		{
			cemuLog_log(LogType::Force, "Vertex shader not found in cache");
			return;
		}
	}
	// find geometry shader
	if (cachedPipeline->gsHash.isPresent)
	{
		geometryShader = LatteSHRC_FindGeometryShader(cachedPipeline->gsHash.baseHash, cachedPipeline->gsHash.auxHash);
		if (!geometryShader)
		{
			cemuLog_log(LogType::Force, "Geometry shader not found in cache");
			return;
		}
	}
	// find pixel shader
	if (cachedPipeline->psHash.isPresent)
	{
		pixelShader = LatteSHRC_FindPixelShader(cachedPipeline->psHash.baseHash, cachedPipeline->psHash.auxHash);
		if (!pixelShader)
		{
			cemuLog_log(LogType::Force, "Pixel shader not found in cache");
			return;
		}
	}

	if (!pixelShader)
	{
		cemu_assert_debug(false);
		return;
	}

	MetalAttachmentsInfo attachmentsInfo(*lcr, pixelShader);

	PipelineObject* pipelineObject = new PipelineObject();

	// compile
	{
		MetalPipelineCompiler pp(m_mtlr, *pipelineObject);
		pp.InitFromState(vertexShader->compatibleFetchShader, vertexShader, geometryShader, pixelShader, cachedPipeline->lastUsedAttachmentsInfo, attachmentsInfo, *lcr);
		if (!pp.Compile(true, true, false))
		{
			// do not cache broken pipelines; they will be re-compiled from the live state during gameplay
			cemuLog_log(LogType::Force, "Failed to compile pipeline restored from cache, skipping");
			delete pipelineObject;
			return;
		}
		// destroy pp early
	}

	// Cache the pipeline
   	uint64 pipelineStateHash = CalculatePipelineHash(vertexShader->compatibleFetchShader, vertexShader, geometryShader, pixelShader, cachedPipeline->lastUsedAttachmentsInfo, attachmentsInfo, *lcr);
   	m_pipelineCacheLock.lock();
   	// an identical pipeline may already be present (duplicate cache entries or a live compile
	// racing the loader) - overwriting would leak the previous object
   	auto [insertItr, inserted] = m_pipelineCache.try_emplace(pipelineStateHash, pipelineObject);
   	m_pipelineCacheLock.unlock();
   	if (!inserted)
   	{
		delete pipelineObject;
   	}
}

ConcurrentQueue<CachedPipeline*> g_mtlPipelineCachingQueue;

void MetalPipelineCache::DiscardPendingJobs()
{
	CachedPipeline* pending = nullptr;
	while (g_mtlPipelineCachingQueue.peek2(pending))
		delete pending;
}

void MetalPipelineCache::StopStoreThread()
{
	if (!m_pipelineCacheStoreThread)
		return;
	m_storeThreadStop.store(true);
	// the thread blocks on an empty queue, so it needs a job to wake up on. A null job is the
	// sentinel for "check the stop flag" (see WorkerThread)
	g_mtlPipelineCachingQueue.push(nullptr);
	if (m_pipelineCacheStoreThread->joinable())
		m_pipelineCacheStoreThread->join();
	delete m_pipelineCacheStoreThread;
	m_pipelineCacheStoreThread = nullptr;
}

void MetalPipelineCache::AddCurrentStateToCache(uint64 pipelineStateHash, const MetalAttachmentsInfo& lastUsedAttachmentsInfo)
{
	if (!m_pipelineCacheStoreThread)
	{
		// deliberately left joinable: StopStoreThread joins it, and that join is the only thing
		// keeping the worker from touching this object after the destructor has run
		m_pipelineCacheStoreThread = new std::thread(&MetalPipelineCache::WorkerThread, this);
	}
	// fill job structure with cached GPU state
	// for each cached pipeline we store:
	// - Active shaders (referenced by hash)
	// - An almost-complete register state of the GPU (minus some ALU uniform constants which aren't relevant)
	CachedPipeline* job = new CachedPipeline();
	auto vs = LatteSHRC_GetActiveVertexShader();
	auto gs = LatteSHRC_GetActiveGeometryShader();
	auto ps = LatteSHRC_GetActivePixelShader();
	if (vs)
		job->vsHash.set(vs->baseHash, vs->auxHash);
	if (gs)
		job->gsHash.set(gs->baseHash, gs->auxHash);
	if (ps)
		job->psHash.set(ps->baseHash, ps->auxHash);
	job->lastUsedAttachmentsInfo = lastUsedAttachmentsInfo;
	Latte::StoreGPURegisterState(LatteGPUState.contextNew, job->gpuState);
	// queue job
	g_mtlPipelineCachingQueue.push(job);
}

bool MetalPipelineCache::SerializePipeline(MemStreamWriter& memWriter, CachedPipeline& cachedPipeline)
{
	memWriter.writeBE<uint8>(0x02); // version
	uint8 presentMask = 0;
	if (cachedPipeline.vsHash.isPresent)
		presentMask |= 1;
	if (cachedPipeline.gsHash.isPresent)
		presentMask |= 2;
	if (cachedPipeline.psHash.isPresent)
		presentMask |= 4;
	memWriter.writeBE<uint8>(presentMask);
	if (cachedPipeline.vsHash.isPresent)
	{
		memWriter.writeBE<uint64>(cachedPipeline.vsHash.baseHash);
		memWriter.writeBE<uint64>(cachedPipeline.vsHash.auxHash);
	}
	if (cachedPipeline.gsHash.isPresent)
	{
		memWriter.writeBE<uint64>(cachedPipeline.gsHash.baseHash);
		memWriter.writeBE<uint64>(cachedPipeline.gsHash.auxHash);
	}
	if (cachedPipeline.psHash.isPresent)
	{
		memWriter.writeBE<uint64>(cachedPipeline.psHash.baseHash);
		memWriter.writeBE<uint64>(cachedPipeline.psHash.auxHash);
	}

	for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
	    memWriter.writeBE<uint16>((uint16)cachedPipeline.lastUsedAttachmentsInfo.colorFormats[i]);
	memWriter.writeBE<uint16>((uint16)cachedPipeline.lastUsedAttachmentsInfo.depthFormat);
	memWriter.writeBE<uint8>(cachedPipeline.lastUsedAttachmentsInfo.hasStencil ? 1 : 0);

	Latte::SerializeRegisterState(cachedPipeline.gpuState, memWriter);

	return true;
}

bool MetalPipelineCache::DeserializePipeline(MemStreamReader& memReader, CachedPipeline& cachedPipeline)
{
	// version
	if (memReader.readBE<uint8>() != 2)
	{
		cemuLog_log(LogType::Force, "Cached Metal pipeline corrupted or has unknown version");
		return false;
	}
	// shader hashes
	uint8 presentMask = memReader.readBE<uint8>();
	if (presentMask & 1)
	{
		uint64 baseHash = memReader.readBE<uint64>();
		uint64 auxHash = memReader.readBE<uint64>();
		cachedPipeline.vsHash.set(baseHash, auxHash);
	}
	if (presentMask & 2)
	{
		uint64 baseHash = memReader.readBE<uint64>();
		uint64 auxHash = memReader.readBE<uint64>();
		cachedPipeline.gsHash.set(baseHash, auxHash);
	}
	if (presentMask & 4)
	{
		uint64 baseHash = memReader.readBE<uint64>();
		uint64 auxHash = memReader.readBE<uint64>();
		cachedPipeline.psHash.set(baseHash, auxHash);
	}

	for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
	    cachedPipeline.lastUsedAttachmentsInfo.colorFormats[i] = (Latte::E_GX2SURFFMT)memReader.readBE<uint16>();
	cachedPipeline.lastUsedAttachmentsInfo.depthFormat = (Latte::E_GX2SURFFMT)memReader.readBE<uint16>();
	cachedPipeline.lastUsedAttachmentsInfo.hasStencil = memReader.readBE<uint8>() != 0;

	// deserialize GPU state
	if (!Latte::DeserializeRegisterState(cachedPipeline.gpuState, memReader))
	{
		return false;
	}
	cemu_assert_debug(!memReader.hasError());

	return true;
}

int MetalPipelineCache::CompilerThread()
{
	SetThreadName("plCacheCompiler");
	while (m_numCompilationThreads != 0)
	{
		std::vector<uint8> pipelineData = m_compilationQueue.pop();
		if(pipelineData.empty())
			continue;
		LoadPipelineFromCache(pipelineData);
		++g_mtlCacheState.pipelinesLoaded;
	}
	return 0;
}

void MetalPipelineCache::WorkerThread()
{
	SetThreadName("plCacheWriter");
	while (true)
	{
		CachedPipeline* job;
		g_mtlPipelineCachingQueue.pop(job);
		if (!job)
		{
			// wake-up sentinel, pushed only by StopStoreThread
			if (m_storeThreadStop.load())
				break;
			continue;
		}
		if (m_storeThreadStop.load())
		{
			// shutting down: persisting this would mean calling into a cache that is being taken
			// down with us (and SerializePipeline runs on the object being destroyed)
			delete job;
			break;
		}
		// cheap early-out for the drain-after-close case, taken under the lock so it cannot race
		{
			std::lock_guard<std::mutex> lock(s_fileCacheMutex);
			if (!s_cache)
			{
				delete job;
				continue;
			}
		}
		// serialize outside the lock - it is the expensive part and does not touch s_cache
		MemStreamWriter memWriter(1024 * 4);
		SerializePipeline(memWriter, *job);
		auto blob = memWriter.getResult();
		// file name is derived from data hash
		uint8 hash[SHA256_DIGEST_LENGTH];
		SHA256(blob.data(), blob.size(), hash);
		uint64 nameA = *(uint64be*)(hash + 0);
		uint64 nameB = *(uint64be*)(hash + 8);
		// null check and use in one critical section so Close() cannot delete s_cache in between
		{
			std::lock_guard<std::mutex> lock(s_fileCacheMutex);
			if (s_cache)
				s_cache->AddFileAsync({ nameA, nameB }, blob.data(), blob.size());
		}
		delete job;
	}
}
