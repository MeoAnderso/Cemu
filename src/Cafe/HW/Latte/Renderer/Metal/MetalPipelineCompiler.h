#pragma once

#include "Cafe/HW/Latte/Renderer/Metal/MetalAttachmentsInfo.h"

#include "Cafe/HW/Latte/ISA/LatteReg.h"
#include "Cafe/HW/Latte/LegacyShaderDecompiler/LatteDecompiler.h"

#include <atomic>

struct PipelineObject
{
    // atomic: written by the pipeline-compilation pool thread, read on the render thread (with
    // the default sequentially-consistent ordering the "clear flags, then publish pipeline"
    // store order in MetalPipelineCompiler::Compile is what the render thread observes)
    std::atomic<MTL::RenderPipelineState*> m_pipeline{ nullptr };
    // set when pipeline compilation failed: the entry is evicted from the cache so the next
    // draw retries compilation instead of silently skipping draws forever
    std::atomic_bool compileFailed{ false };
    // set when the failure is permanent (a shader stage failed to compile) - retrying would
    // fail identically on every draw, so the cache keeps the entry and skips the draws instead
    // of paying a recompile + log per frame. Driver-side failures stay retryable
    std::atomic_bool permanentFailure{ false };
    // Retries spent on a driver-side (non-permanent) failure. Render thread only. Without a cap, a
    // failure that is not a shader-stage error - which the driver rejects identically every time -
    // recompiles and logs at Force severity once per draw for the rest of the session
    uint32 transientFailureCount{ 0 };
    // set when the pipeline was queued for asynchronous compilation: persistence is deferred until
    // the compile result is observed on the render thread (AddCurrentStateToCache snapshots the
    // active register/shader state, which the compile thread must not touch). Render thread only
    bool persistWhenCompiled = false;
};

class MetalPipelineCompiler
{
public:
    MetalPipelineCompiler(class MetalRenderer* metalRenderer, PipelineObject& pipelineObj) : m_mtlr{metalRenderer}, m_pipelineObj{pipelineObj} {}
    ~MetalPipelineCompiler();

    void InitFromState(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr);

    bool Compile(bool forceCompile, bool isRenderThread, bool showInOverlay);

    // The pixel shader's fragment function, or null when there is none. See the definition
    MTL::Function* ResolveFragmentFunction();

private:
    // Creates the pipeline state, probing and then feeding the renderer's MTLBinaryArchive when one
    // is open. A miss compiles and records the result; a hit skips the compilation entirely. See the
    // definition for the descriptor-kind and availability details
    template <typename TDescriptor>
    MTL::RenderPipelineState* CreatePipelineStateWithArchive(TDescriptor* desc, NS::Error** error);

    class MetalRenderer* m_mtlr;
    PipelineObject& m_pipelineObj;

    class RendererShaderMtl* m_vertexShaderMtl;
    class RendererShaderMtl* m_geometryShaderMtl;
    // set when m_geometryShaderMtl was generated for rect emulation (owned by this compiler);
    // geometry shaders taken from the vertex shader's GS are owned elsewhere and never deleted here
    class RendererShaderMtl* m_ownedGeometryShaderMtl = nullptr;
    class RendererShaderMtl* m_pixelShaderMtl;
    bool m_usesGeometryShader;
    bool m_rasterizationEnabled;
    const LatteDecompilerShader* m_pixelShader = nullptr;
    MetalAttachmentsInfo m_activeAttachmentsInfo;

    NS::Object* m_pipelineDescriptor = nullptr;

    void InitFromStateRender(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr);

    void InitFromStateMesh(const LatteFetchShader* fetchShader, const class MetalAttachmentsInfo& lastUsedAttachmentsInfo, const class MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr);
};
