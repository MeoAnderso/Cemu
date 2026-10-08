#pragma once

#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCompiler.h"
#include "util/helpers/ConcurrentQueue.h"
#include "util/helpers/fspinlock.h"
#include "util/math/vector2.h"

class MetalPipelineCache
{
public:
	static MetalPipelineCache& GetInstance();

    MetalPipelineCache(class MetalRenderer* metalRenderer);
    ~MetalPipelineCache();

    PipelineObject* GetRenderPipelineState(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, Vector2i extend, uint32 indexCount, const LatteContextRegister& lcr);

    // MTLBinaryArchive plumbing. Two archives, deliberately:
    //
    // - m_binaryArchiveRead is the file loaded at BeginLoading. It is read-only: it serves hits and is
    //   never written to.
    // - m_binaryArchiveWrite is created fresh (nil url) on this session's first add and is what Close()
    //   serializes, so the file always holds exactly one session's pipeline set.
    //
    // Re-serializing a *loaded* archive does not round-trip (the file inflates, or serialization fails
    // with "expecting 'fragment' stage" inside the loaded range), so the file is written from this
    // session's pipelines only, and loaded content is only ever read.
    //
    // Lookup attaches both, so a hit can come from either; see the docs on MTLRenderPipelineDescriptor's
    // binaryArchives array. Best effort in every direction: a miss, a failed add or a failed serialize
    // costs nothing but a compilation
    NS::Array* GetBinaryArchiveLookupArray();

    // Called by the pipeline compiler after a successful creation, so the pipeline can be served from
    // the archive next session. The shader hashes are for the trace: with the MetalBackend category on
    // the adds are logged in order, which is what maps a packer failure's entry index back to a shader
    bool AddRenderPipelineToArchive(MTL::RenderPipelineDescriptor* descriptor, uint64 vertexBaseHash, uint64 pixelBaseHash);
    bool AddMeshRenderPipelineToArchive(MTL::MeshRenderPipelineDescriptor* descriptor, uint64 vertexBaseHash, uint64 pixelBaseHash);

    // Cache loading
	uint32 BeginLoading(uint64 cacheTitleId); // returns count of pipelines stored in cache
	bool UpdateLoading(uint32& pipelinesLoadedTotal, uint32& pipelinesMissingShaders);
	void EndLoading();
	void LoadPipelineFromCache(std::span<uint8> fileData);
       void Close(); // called on title exit
    // Stops everything that compiles pipelines from shader objects - the async compile pool and the
    // cache-loader threads - and waits for in-flight compiles. MetalRenderer::Shutdown calls it, because
    // LatteThread_Exit frees all shaders before this cache is destroyed. Idempotent
    void StopCompilation();

    // Debug
    size_t GetPipelineCacheSize() const { return m_pipelineCache.size(); }

private:
    class MetalRenderer* m_mtlr;

    std::map<uint64, PipelineObject*> m_pipelineCache;
    FSpinlock m_pipelineCacheLock;

	std::thread* m_pipelineCacheStoreThread{ nullptr };
	// set by StopStoreThread to retire the writer thread. The thread calls back into this object,
	// so it is joined before the object goes away - see StopStoreThread. It must stay joinable
	// (never detach it) or that join silently becomes a no-op
	std::atomic_bool m_storeThreadStop{ false };

	class FileCache* s_cache{ nullptr };

	// see the comments on GetBinaryArchiveLookupArray. Read is loaded at BeginLoading, Write is created
	// on this session's first add; either can be null
	MTL::BinaryArchive* m_binaryArchiveRead = nullptr;
	MTL::BinaryArchive* m_binaryArchiveWrite = nullptr;
	MTL::BinaryArchive* EnsureBinaryArchiveWrite(); // creates the fresh write archive on first use
	void NoteBinaryArchiveAdd(uint64 vertexBaseHash, uint64 pixelBaseHash);
	std::string m_binaryArchivePath;
	// set when a pipeline was added to the write archive this session, so Close() can skip writing when
	// nothing was added (the file on disk stays as it is, which is fine - it is still loadable)
	bool m_binaryArchiveDirty = false;
	// entries added this session, reported in the write log line and in the per-add trace
	uint32 m_binaryArchiveAddCount = 0;
	// guards m_binaryArchiveWrite (lazy creation), m_binaryArchiveDirty, m_binaryArchiveAddCount and every
	// add/serialize on the write archive (MTLBinaryArchive's thread safety is undocumented)
	std::mutex m_binaryArchiveMutex;
	// pipeline-creation totals at BeginLoading, so Close can report what this session spent creating
	// pipelines - the cost the archive exists to avoid (see the declarations in MetalPipelineCompiler)
	uint64 m_pipelineCreationTimeAtBegin = 0;
	uint32 m_pipelineCreationCountAtBegin = 0;

	std::atomic_uint32_t m_numCompilationThreads{ 0 };
	ConcurrentQueue<std::vector<uint8>> m_compilationQueue;
	// cache-loader threads (CompilerThread). Joinable: they call back into this object (LoadPipelineFromCache)
	std::vector<std::thread> m_compilationThreads;

    static uint64 CalculatePipelineHash(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr);

    void AddCurrentStateToCache(uint64 pipelineStateHash, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo);

	// pipeline serialization for file
	bool SerializePipeline(class MemStreamWriter& memWriter, struct CachedPipeline& cachedPipeline);
	bool DeserializePipeline(class MemStreamReader& memReader, struct CachedPipeline& cachedPipeline);

    int CompilerThread();
	void WorkerThread();
	void StopStoreThread();
	void StopLoaderThreads();
	void DiscardPendingJobs();
};
