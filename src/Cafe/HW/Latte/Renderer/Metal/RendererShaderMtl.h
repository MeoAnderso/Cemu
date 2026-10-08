#pragma once

#include "Cafe/HW/Latte/Renderer/RendererShader.h"
#include "HW/Latte/Renderer/Metal/CachedFBOMtl.h"
#include "HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "util/helpers/ConcurrentQueue.h"
#include "util/helpers/Semaphore.h"

#include <Metal/Metal.hpp>

class RendererShaderMtl : public RendererShader
{
    friend class ShaderMtlThreadPool;

	enum class COMPILATION_STATE : uint32
	{
		NONE,
		QUEUED,
		COMPILING,
		DONE
	};

public:
    static void ShaderCacheLoading_begin(uint64 cacheTitleId);
    static void ShaderCacheLoading_end();
    static void ShaderCacheLoading_Close();

    static void Initialize();
	static void Shutdown();

	// the Metal compile options every MSL compilation uses (shaders and the pack translator's check)
	static MTL::CompileOptions* NewCompileOptions(class MetalRenderer* mtlr); // +1, caller releases

	RendererShaderMtl(class MetalRenderer* mtlRenderer, ShaderType type, uint64 baseHash, uint64 auxHash, bool isGameShader, bool isGfxPackShader, const std::string& mslCode);
	virtual ~RendererShaderMtl();

	MTL::Function* GetFunction() const
	{
	    return m_function;
	}

	// shader identity for GPU-trace labels
	uint64 GetBaseHash() const
	{
		return m_baseHash;
	}

	bool HasError() const
	{
		return m_hasError;
	}

	void PreponeCompilation(bool isRenderThread) override;
	bool IsCompiled() override;
	bool WaitForCompiled() override;

private:
    class MetalRenderer* m_mtlr;

	MTL::Function* m_function = nullptr;

	// written by the shader-compile thread, read from the pipeline-compile pool and the render thread.
	// The semaphore already orders those, but a plain bool would silently break if the ordering ever
	// changed, and an atomic costs nothing here
	std::atomic_bool m_hasError{ false };
	// guards the one-time g_compiled_shaders_async stall decrement in PreponeCompilation
	std::atomic_bool m_preponeCounted{ false };

	StateSemaphore<COMPILATION_STATE> m_compilationState{ COMPILATION_STATE::NONE };

	std::string m_mslCode;

	bool ShouldCountCompilation() const;

	MTL::Library* LibraryFromSource();

	void CompileInternal();

	void FinishCompilation();
};
