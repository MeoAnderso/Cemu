#pragma once

#include <map>
#include <mutex>

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

	// Combined entry point for output stripping: variant with the color outputs of
	// removedColorMask (bitmask 0..7) removed and - when removeDepth is set - the depth output
	// removed. Cached per combination. Returns nullptr when the source cannot be stripped
	// (e.g. translated graphic-pack MSL) or compilation failed
	MTL::Function* GetStrippedVariant(uint32 removedColorMask, bool removeDepth);

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

	bool m_hasError = false;
	// guards the one-time g_compiled_shaders_async stall decrement in PreponeCompilation
	std::atomic_bool m_preponeCounted{ false };

	StateSemaphore<COMPILATION_STATE> m_compilationState{ COMPILATION_STATE::NONE };

	std::string m_mslCode;
	// lazily compiled fragment-function variants with stripped color outputs and/or depth
	// output (see GetStrippedVariant), keyed by the removed-output bitmask (bit 8 signals
	// depth removal)
	std::mutex m_variantMutex;
	std::map<uint32, MTL::Function*> m_strippedFunctionCache;

	bool ShouldCountCompilation() const;

	MTL::Library* LibraryFromSource();

	//MTL::Library* LibraryFromAIR(std::span<uint8> data);

	void CompileInternal();

	//void CompileToAIR();

	void FinishCompilation();
};
