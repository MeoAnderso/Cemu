#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCompiler.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/CachedFBOMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/RendererShaderMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalCppCompat.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"

#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/ISA/RegDefines.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"

#include <chrono>

extern std::atomic_int g_compiling_pipelines;
extern std::atomic_int g_compiling_pipelines_async;
extern std::atomic_uint64_t g_compiling_pipelines_syncTimeSum;

// Session totals of successfully created pipelines and what they cost, for the pipeline cache's
// summary (MetalPipelineCache::Close). Deliberately separate from the counters above: those are
// accumulated only on the render-thread path with overlay reporting on (see the showInOverlay block in
// Compile), so the async compile pool and the shader-cache replay - where most creations happen - are
// not in them, and the OSD exchanges the count to zero every time it draws its notice
std::atomic_uint64_t g_mtlPipelineCreationTimeNs{ 0 };
std::atomic_uint32_t g_mtlPipelineCreationCount{ 0 };

static void rectsEmulationGS_outputSingleVertex(std::string& gsSrc, const LatteDecompilerShader* vertexShader, LatteShaderPSInputTable& psInputTable, sint32 vIdx, const LatteContextRegister& latteRegister)
{
	auto parameterMask = vertexShader->outputParameterMask;
	for (uint32 i = 0; i < 32; i++)
	{
		if ((parameterMask & (1 << i)) == 0)
			continue;
		sint32 vsSemanticId = psInputTable.getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
		if (vsSemanticId < 0)
			continue;
		// make sure PS has matching input
		if (!psInputTable.hasPSImportForSemanticId(vsSemanticId))
			continue;
		gsSrc.append(fmt::format("out.passParameterSem{} = objectPayload.vertexOut[{}].passParameterSem{};\r\n", vsSemanticId, vIdx, vsSemanticId));
	}
	gsSrc.append(fmt::format("out.position = objectPayload.vertexOut[{}].position;\r\n", vIdx));
	gsSrc.append(fmt::format("mesh.set_vertex({}, out);\r\n", vIdx));
}

static void rectsEmulationGS_outputGeneratedVertex(std::string& gsSrc, const LatteDecompilerShader* vertexShader, LatteShaderPSInputTable& psInputTable, const char* variant, const LatteContextRegister& latteRegister)
{
	auto parameterMask = vertexShader->outputParameterMask;
	for (uint32 i = 0; i < 32; i++)
	{
		if ((parameterMask & (1 << i)) == 0)
			continue;
		sint32 vsSemanticId = psInputTable.getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
		if (vsSemanticId < 0)
			continue;
		// make sure PS has matching input
		if (!psInputTable.hasPSImportForSemanticId(vsSemanticId))
			continue;
		gsSrc.append(fmt::format("out.passParameterSem{} = gen4thVertex{}(objectPayload.vertexOut[0].passParameterSem{}, objectPayload.vertexOut[1].passParameterSem{}, objectPayload.vertexOut[2].passParameterSem{});\r\n", vsSemanticId, variant, vsSemanticId, vsSemanticId, vsSemanticId));
	}
	gsSrc.append(fmt::format("out.position = gen4thVertex{}(objectPayload.vertexOut[0].position, objectPayload.vertexOut[1].position, objectPayload.vertexOut[2].position);\r\n", variant));
	gsSrc.append(fmt::format("mesh.set_vertex(3, out);\r\n"));
}

static void rectsEmulationGS_outputVerticesCode(std::string& gsSrc, const LatteDecompilerShader* vertexShader, LatteShaderPSInputTable& psInputTable, sint32 p0, sint32 p1, sint32 p2, sint32 p3, const char* variant, const LatteContextRegister& latteRegister)
{
	sint32 pList[4] = { p0, p1, p2, p3 };
	for (sint32 i = 0; i < 4; i++)
	{
		if (pList[i] == 3)
			rectsEmulationGS_outputGeneratedVertex(gsSrc, vertexShader, psInputTable, variant, latteRegister);
		else
			rectsEmulationGS_outputSingleVertex(gsSrc, vertexShader, psInputTable, pList[i], latteRegister);
	}
	gsSrc.append(fmt::format("mesh.set_index(0, {});\r\n", pList[0]));
	gsSrc.append(fmt::format("mesh.set_index(1, {});\r\n", pList[1]));
	gsSrc.append(fmt::format("mesh.set_index(2, {});\r\n", pList[2]));
	gsSrc.append(fmt::format("mesh.set_index(3, {});\r\n", pList[1]));
	gsSrc.append(fmt::format("mesh.set_index(4, {});\r\n", pList[2]));
	gsSrc.append(fmt::format("mesh.set_index(5, {});\r\n", pList[3]));
}

static RendererShaderMtl* rectsEmulationGS_generate(MetalRenderer* metalRenderer, const LatteDecompilerShader* vertexShader, const LatteContextRegister& latteRegister)
{
	std::string gsSrc;
	gsSrc.append("#include <metal_stdlib>\r\n");
	gsSrc.append("using namespace metal;\r\n");

	LatteShaderPSInputTable psInputTable;
	LatteShader_CreatePSInputTable(&psInputTable, latteRegister.GetRawView());

	// inputs & outputs
	std::string vertexOutDefinition = "struct VertexOut {\r\n";
	vertexOutDefinition += "float4 position;\r\n";
	std::string geometryOutDefinition = "struct GeometryOut {\r\n";
	geometryOutDefinition += "float4 position [[position]];\r\n";
	auto parameterMask = vertexShader->outputParameterMask;
	for (uint32 i = 0; i < 32; i++)
	{
		if ((parameterMask & (1 << i)) == 0)
			continue;
		sint32 vsSemanticId = psInputTable.getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
		if (vsSemanticId < 0)
			continue;
		auto psImport = psInputTable.getPSImportBySemanticId(vsSemanticId);
		if (psImport == nullptr)
			continue;

		// VertexOut
		vertexOutDefinition += fmt::format("float4 passParameterSem{};\r\n", vsSemanticId);

		// GeometryOut
		geometryOutDefinition += fmt::format("float4 passParameterSem{}", vsSemanticId);

        geometryOutDefinition += fmt::format(" [[user(locn{})]]", psInputTable.getPSImportLocationBySemanticId(vsSemanticId));
        if (psImport->isFlat)
            geometryOutDefinition += " [[flat]]";
        if (psImport->isNoPerspective)
			geometryOutDefinition += " [[center_no_perspective]]";
        geometryOutDefinition += ";\r\n";
	}
	vertexOutDefinition += "};\r\n";
	geometryOutDefinition += "};\r\n";

	gsSrc.append(vertexOutDefinition);
	gsSrc.append(geometryOutDefinition);

	gsSrc.append("struct ObjectPayload {\r\n");
	gsSrc.append("VertexOut vertexOut[3];\r\n");
	gsSrc.append("};\r\n");

	// gen function
	gsSrc.append("float4 gen4thVertexA(float4 a, float4 b, float4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return b - (c - a);\r\n");
	gsSrc.append("}\r\n");

	gsSrc.append("float4 gen4thVertexB(float4 a, float4 b, float4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return c - (b - a);\r\n");
	gsSrc.append("}\r\n");

	gsSrc.append("float4 gen4thVertexC(float4 a, float4 b, float4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return c + (b - a);\r\n");
	gsSrc.append("}\r\n");

	// main
	gsSrc.append("using MeshType = mesh<GeometryOut, void, 4, 2, topology::triangle>;\r\n");
	gsSrc.append("[[mesh, max_total_threads_per_threadgroup(1)]]\r\n");
	gsSrc.append("void main0(MeshType mesh, const object_data ObjectPayload& objectPayload [[payload]])\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("GeometryOut out;\r\n");

	// there are two possible winding orders that need different triangle generation:
	// 0 1
	// 2 3
	// and
	// 0 1
	// 3 2
	// all others are just symmetries of these cases

	// we can determine the case by comparing the distance 0<->1 and 0<->2

	gsSrc.append("float dist0_1 = length(objectPayload.vertexOut[1].position.xy - objectPayload.vertexOut[0].position.xy);\r\n");
	gsSrc.append("float dist0_2 = length(objectPayload.vertexOut[2].position.xy - objectPayload.vertexOut[0].position.xy);\r\n");
	gsSrc.append("float dist1_2 = length(objectPayload.vertexOut[2].position.xy - objectPayload.vertexOut[1].position.xy);\r\n");

	// emit vertices
	gsSrc.append("if(dist0_1 > dist0_2 && dist0_1 > dist1_2)\r\n");
	gsSrc.append("{\r\n");
	// p0 to p1 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 2, 1, 0, 3, "A", latteRegister);
	gsSrc.append("} else if ( dist0_2 > dist0_1 && dist0_2 > dist1_2 ) {\r\n");
	// p0 to p2 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 1, 2, 0, 3, "B", latteRegister);
	gsSrc.append("} else {\r\n");
	// p1 to p2 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 0, 1, 2, 3, "C", latteRegister);
	gsSrc.append("}\r\n");

	gsSrc.append("mesh.set_primitive_count(2);\r\n");

	gsSrc.append("}\r\n");

	auto mtlShader = new RendererShaderMtl(metalRenderer, RendererShader::ShaderType::kGeometry, 0, 0, false, false, gsSrc);
	mtlShader->PreponeCompilation(true);

	return mtlShader;
}

#define INVALID_TITLE_ID 0xFFFFFFFFFFFFFFFF

uint64 s_cacheTitleId = INVALID_TITLE_ID;

extern std::atomic_int g_compiled_shaders_total;
extern std::atomic_int g_compiled_shaders_async;

template<typename T>
void SetFragmentState(T* desc, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, bool rasterizationEnabled, const LatteContextRegister& lcr)
{
	// NOTE: the attachment pixel formats must be declared from the ACTIVE attachments: the render
	// pass the pipeline runs in is created from the active FBO, and Metal requires the pipeline's
	// pixel formats to match the framebuffer (regardless of rasterization being enabled - same as
	// Vulkan's render pass compatibility rules). The lastUsed attachments only feed the hash.
	// Formats are declared even when rasterization is disabled; only the rasterization stage is
	// turned off below.

    // Color attachments
	const Latte::LATTE_CB_COLOR_CONTROL& colorControlReg = lcr.CB_COLOR_CONTROL;
	uint32 blendEnableMask = colorControlReg.get_BLEND_MASK();
	uint32 renderTargetMask = lcr.CB_TARGET_MASK.get_MASK();
	for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
	{
	    Latte::E_GX2SURFFMT format = activeAttachmentsInfo.colorFormats[i];
		if (format == Latte::E_GX2SURFFMT::INVALID_FORMAT)
		    continue;

		MTL::PixelFormat pixelFormat = GetMtlPixelFormat(format, false);
		auto colorAttachment = desc->colorAttachments()->object(i);
		colorAttachment->setPixelFormat(pixelFormat);

		colorAttachment->setWriteMask(GetMtlColorWriteMask((renderTargetMask >> (i * 4)) & 0xF));

		// Blending
		bool blendEnabled = ((blendEnableMask & (1 << i))) != 0;
		// integer formats are not blendable (Vulkan's InitBlendState does the same); UNORM/SNORM are FLOAT in the format table
		const auto dataType = GetMtlPixelFormatInfo(format, false).dataType;
		if (blendEnabled && dataType != MetalDataType::UINT && dataType != MetalDataType::INT)
		{
       		colorAttachment->setBlendingEnabled(true);

       		const auto& blendControlReg = lcr.CB_BLENDN_CONTROL[i];

       		auto rgbBlendOp = GetMtlBlendOp(blendControlReg.get_COLOR_COMB_FCN());
       		auto srcRgbBlendFactor = GetMtlBlendFactor(blendControlReg.get_COLOR_SRCBLEND());
       		auto dstRgbBlendFactor = GetMtlBlendFactor(blendControlReg.get_COLOR_DSTBLEND());

       		colorAttachment->setRgbBlendOperation(rgbBlendOp);
       		colorAttachment->setSourceRGBBlendFactor(srcRgbBlendFactor);
       		colorAttachment->setDestinationRGBBlendFactor(dstRgbBlendFactor);
       		if (blendControlReg.get_SEPARATE_ALPHA_BLEND())
       		{
       			colorAttachment->setAlphaBlendOperation(GetMtlBlendOp(blendControlReg.get_ALPHA_COMB_FCN()));
      		    colorAttachment->setSourceAlphaBlendFactor(GetMtlBlendFactor(blendControlReg.get_ALPHA_SRCBLEND()));
      		    colorAttachment->setDestinationAlphaBlendFactor(GetMtlBlendFactor(blendControlReg.get_ALPHA_DSTBLEND()));
       		}
       		else
       		{
           		colorAttachment->setAlphaBlendOperation(rgbBlendOp);
           		colorAttachment->setSourceAlphaBlendFactor(srcRgbBlendFactor);
           		colorAttachment->setDestinationAlphaBlendFactor(dstRgbBlendFactor);
       		}
		}
	}

	// Depth stencil attachment
	if (activeAttachmentsInfo.depthFormat != Latte::E_GX2SURFFMT::INVALID_FORMAT)
	{
	    MTL::PixelFormat pixelFormat = GetMtlPixelFormat(activeAttachmentsInfo.depthFormat, true);
        desc->setDepthAttachmentPixelFormat(pixelFormat);
        if (activeAttachmentsInfo.hasStencil)
            desc->setStencilAttachmentPixelFormat(pixelFormat);
	}

	if (!rasterizationEnabled/* TODO: check if the pixel shader is valid as well? */)
		desc->setRasterizationEnabled(false);
}

MetalPipelineCompiler::~MetalPipelineCompiler()
{
    if (m_pipelineDescriptor)
        m_pipelineDescriptor->release();
    // the rect-emulation geometry shader is generated per pipeline compile and owned here;
    // compilation is done synchronously inside rectsEmulationGS_generate, so it is safe to
    // delete as soon as the pipeline state (which retains the compiled function) exists
    delete m_ownedGeometryShaderMtl;
}

void MetalPipelineCompiler::InitFromState(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const LatteDecompilerShader* geometryShader, const LatteDecompilerShader* pixelShader, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr)
{
    m_usesGeometryShader = UseGeometryShader(lcr, geometryShader != nullptr);
    if (m_usesGeometryShader && !m_mtlr->SupportsMeshShaders())
    {
        // the draw path's GeometryShaderUnsupported, reached from pipeline construction when the cache
        // loader replays a cache written on a mesh-capable machine. Reported here and not in Compile(),
        // which tests the same condition on the same compiler - reporting in both would count it twice
        MetalDiag_CountOncePer(MetalDiagEvent::GeometryShaderUnsupported,
            (uintptr_t)(geometryShader ? geometryShader->baseHash : 0),
            "geometry shader {:016x}_{:016x}{} cannot be built - no mesh shader support on this device",
            geometryShader ? geometryShader->baseHash : 0ull,
            geometryShader ? geometryShader->auxHash : 0ull,
            geometryShader ? "" : " (RECTS emulation)");
        return;
    }

    // Rasterization
	m_rasterizationEnabled = lcr.IsRasterizationEnabled();

    // Shaders
    m_vertexShaderMtl = static_cast<RendererShaderMtl*>(vertexShader->shader);
    if (geometryShader)
        m_geometryShaderMtl = static_cast<RendererShaderMtl*>(geometryShader->shader);
    else if (UseRectEmulation(lcr))
        m_geometryShaderMtl = m_ownedGeometryShaderMtl = rectsEmulationGS_generate(m_mtlr, vertexShader, lcr);
    else
        m_geometryShaderMtl = nullptr;
    m_pixelShaderMtl = static_cast<RendererShaderMtl*>(pixelShader->shader);
    // Kept for ResolveFragmentFunction (orphan-output stripping) and the failure trace below
    m_pixelShader = pixelShader;
    // Both must be set here rather than in InitFromStateRender: the mesh path (geometry shaders,
    // RECTS emulation) never reaches that function, and ResolveFragmentFunction would otherwise
    // strip every color output the pixel shader writes
    m_activeAttachmentsInfo = activeAttachmentsInfo;

    if (m_usesGeometryShader)
        InitFromStateMesh(fetchShader, lastUsedAttachmentsInfo, activeAttachmentsInfo, lcr);
    else
        InitFromStateRender(fetchShader, vertexShader, lastUsedAttachmentsInfo, activeAttachmentsInfo, lcr);
}

// Creates the pipeline state, consulting and then feeding the renderer's MTLBinaryArchive when one is
// open. A miss just compiles; a failed add only means the entry is missing from the archive next
// launch. Both are ignored on purpose - neither affects rendering.
//
// Both descriptor kinds are archived. The mesh path matters as much as the plain one: it covers
// geometry-shader and RECTS emulation, whose mesh function is *generated* per pipeline, so a cache hit
// saves generating and compiling it too.
// The mesh descriptor's archive support is API_AVAILABLE(macos(15.0)), newer than the deployment
// target (13.4) and than the pinned metal-cpp: the two calls are shimmed in MetalCppCompat.h, whose
// availability attributes make the compiler enforce the __builtin_available guards
template <typename TDescriptor>
MTL::RenderPipelineState* MetalPipelineCompiler::CreatePipelineStateWithArchive(TDescriptor* desc, NS::Error** error)
{
    MetalRenderer* mtlr = m_mtlr;
    MetalPipelineCache* pipelineCache = mtlr->GetPipelineCache();
    // both archives, so a hit can come from the previous session's file or from one already added this
    // session. Autoreleased, consumed by the descriptor during the creation call below
    NS::Array* archives = pipelineCache ? pipelineCache->GetBinaryArchiveLookupArray() : nullptr;

    const bool meshDescriptor = !std::is_same_v<TDescriptor, MTL::RenderPipelineDescriptor>;
    // Can this pipeline be archived at all? Two independent reasons it cannot:
    // - it has no fragment stage. The packer requires one ("expecting 'fragment' stage in pipeline no.
    //   N") and serializeToURL is all-or-nothing, so one such entry would cost the whole file. This
    //   mirrors exactly what gives the descriptor its fragment function: the emitters set it only when
    //   rasterization is enabled, and ResolveFragmentFunction returns null when there is no pixel shader
    //   (which Compile's early check does not catch, because its "m_pixelShaderMtl &&" test
    //   short-circuits). Both cases are legal to create and are created normally - just not archived.
    // - it is a mesh pipeline on an OS that predates the mesh archive API (see above)
    // Note this is decided from the pipeline, not from whether an archive already exists: on a title's
    // first run there is none, and it is this session's additions that create the file
    const bool hasFragmentStage = m_rasterizationEnabled && m_pixelShaderMtl && m_pixelShaderMtl->GetFunction();
    bool meshArchiveAvailable = false;
    if (__builtin_available(macOS 15.0, *))
        meshArchiveAvailable = true;
    const bool canArchive = hasFragmentStage && (!meshDescriptor || meshArchiveAvailable);
    if (!canArchive)
    {
        MetalDiag_Count(MetalDiagEvent::PipelineArchiveSkipped, "vs {:016x} ps {:016x} (rasterization {} pixel shader {})",
            m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0, m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0,
            m_rasterizationEnabled ? "on" : "off", m_pixelShaderMtl ? "present" : "absent");
    }

    // Attaching an archive is what enables a hit, so it needs one to exist; adding does not
    if (archives && canArchive)
    {
        if constexpr (std::is_same_v<TDescriptor, MTL::RenderPipelineDescriptor>)
            desc->setBinaryArchives(archives);
        else if (__builtin_available(macOS 15.0, *))
            MetalCppCompat::MeshRenderPipelineDescriptorSetBinaryArchives(desc, archives);

        // Verification mode, not the normal path: FailOnBinaryArchiveMiss makes a successful creation
        // mean "the archive served this pipeline", which is the only way a hit is observable (Metal
        // reports nothing otherwise), and the census counts them. It is only probed when the user asked
        // for Metal backend detail (Debug -> Logging) because a miss is documented as an error return
        // while an unexpected MTLReportFailure under API validation would abort the process - so the
        // default path stays the plain attach-and-compile that is known to be safe.
        // With the category off the archive is still attached and still used, just not observed
        if (cemuLog_isLoggingEnabled(LogType::MetalBackend))
        {
            MTL::RenderPipelineState* archived = mtlr->GetDevice()->newRenderPipelineState(desc, MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, nullptr);
            if (archived)
            {
                MetalDiag_Count(MetalDiagEvent::PipelineArchiveHit, "vs {:016x} ps {:016x}",
                    m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0, m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0);
                // Carried into this session's write archive as well. The file is rebuilt from the
                // pipelines this session used, so one served by a hit has to be re-added too - otherwise
                // the next file would hold only the misses and coverage would shrink every cycle.
                // (Without the probe this is not needed: creation without FailOnBinaryArchiveMiss goes
                // through the normal path below and is added there either way)
                const uint64 vertexHash = m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0;
                const uint64 pixelHash = m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0;
                if (pipelineCache)
                {
                    if constexpr (std::is_same_v<TDescriptor, MTL::RenderPipelineDescriptor>)
                        pipelineCache->AddRenderPipelineToArchive(desc, vertexHash, pixelHash);
                    else
                        pipelineCache->AddMeshRenderPipelineToArchive(desc, vertexHash, pixelHash);
                }
                return archived;
            }
            MetalDiag_Count(MetalDiagEvent::PipelineArchiveMiss, "vs {:016x} ps {:016x}",
                m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0, m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0);
        }
    }

    MTL::RenderPipelineState* pipeline = mtlr->GetDevice()->newRenderPipelineState(desc, MTL::PipelineOptionNone, nullptr, error);

    if (pipeline && canArchive && pipelineCache)
    {
        // goes into the cache's fresh write archive (never the loaded one - see the header), so the
        // file written at exit holds exactly this session's pipelines
        const uint64 vertexHash = m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0;
        const uint64 pixelHash = m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0;
        if constexpr (std::is_same_v<TDescriptor, MTL::RenderPipelineDescriptor>)
            pipelineCache->AddRenderPipelineToArchive(desc, vertexHash, pixelHash);
        else
            pipelineCache->AddMeshRenderPipelineToArchive(desc, vertexHash, pixelHash);
    }
    return pipeline;
}

bool MetalPipelineCompiler::Compile(bool forceCompile, bool isRenderThread, bool showInOverlay)
{
    // This runs on the async pipeline-compile threads and on the shader-cache loader thread, and
    // neither installs an ObjC autorelease pool. Every autoreleased object created here would leak
    // without one - notably the pipeline label strings built with ToNSString below, which stopped
    // being debug-only. RendererShaderMtl installs the same pool around MSL compilation for
    // exactly this reason (an imbalance there crashes at thread exit)
    NS_STACK_SCOPED NS::AutoreleasePool* autoreleasePool = NS::AutoreleasePool::alloc()->init();

    // already reported by InitFromState, which tests the same condition on this compiler
    if (m_usesGeometryShader && !m_mtlr->SupportsMeshShaders())
        return false;

    if (forceCompile)
	{
		// if some shader stages are not compiled yet, compile them now
		if (m_vertexShaderMtl && !m_vertexShaderMtl->IsCompiled())
			m_vertexShaderMtl->PreponeCompilation(isRenderThread);
		if (m_geometryShaderMtl && !m_geometryShaderMtl->IsCompiled())
			m_geometryShaderMtl->PreponeCompilation(isRenderThread);
		if (m_pixelShaderMtl && !m_pixelShaderMtl->IsCompiled())
			m_pixelShaderMtl->PreponeCompilation(isRenderThread);
	}
	else
	{
	    // fail early if some shader stages are not compiled
		if (m_vertexShaderMtl && !m_vertexShaderMtl->IsCompiled())
			return false;
		if (m_geometryShaderMtl && !m_geometryShaderMtl->IsCompiled())
			return false;
		if (m_pixelShaderMtl && !m_pixelShaderMtl->IsCompiled())
			return false;
	}

	// fail early if any shader stage failed to compile (permanent: retrying fails identically)
	if ((m_vertexShaderMtl && m_vertexShaderMtl->HasError()) ||
	    (m_geometryShaderMtl && m_geometryShaderMtl->HasError()) ||
	    (m_pixelShaderMtl && m_pixelShaderMtl->HasError()))
	{
		MetalDiag_Count(MetalDiagEvent::PipelineCompileFailedPermanent, "shader stage failed to compile (vs {:016x}, ps {:016x})",
			m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0, m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0);
		m_pipelineObj.compileFailed = true;
		m_pipelineObj.permanentFailure = true;
		return false;
	}

	// fail early if any required shader function is missing (permanent for the same reason)
	if (!m_vertexShaderMtl || !m_vertexShaderMtl->GetFunction() ||
	    (m_usesGeometryShader && (!m_geometryShaderMtl || !m_geometryShaderMtl->GetFunction())) ||
	    (m_rasterizationEnabled && m_pixelShaderMtl && !m_pixelShaderMtl->GetFunction()))
	{
		MetalDiag_Count(MetalDiagEvent::PipelineCompileFailedPermanent, "shader function missing (vs {:016x}, ps {:016x})",
			m_vertexShaderMtl ? m_vertexShaderMtl->GetBaseHash() : 0, m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0);
		m_pipelineObj.compileFailed = true;
		m_pipelineObj.permanentFailure = true;
		return false;
	}

	// Compile
    MTL::RenderPipelineState* pipeline = nullptr;
    NS::Error* error = nullptr;

    auto start = std::chrono::high_resolution_clock::now();
    if (m_usesGeometryShader)
    {
        auto desc = static_cast<MTL::MeshRenderPipelineDescriptor*>(m_pipelineDescriptor);

        // Shaders
        desc->setObjectFunction(m_vertexShaderMtl->GetFunction());
        desc->setMeshFunction(m_geometryShaderMtl->GetFunction());
        if (m_rasterizationEnabled)
            desc->setFragmentFunction(ResolveFragmentFunction());

        // label with the game-side shader identity so GPU traces are searchable
        desc->setLabel(ToNSString(fmt::format("rps mesh vs{:016x} gs{:016x} ps{:016x}",
            m_vertexShaderMtl->GetBaseHash(), m_geometryShaderMtl ? m_geometryShaderMtl->GetBaseHash() : 0,
            m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0)));
       	pipeline = CreatePipelineStateWithArchive(desc, &error);
    }
    else
    {
        auto desc = static_cast<MTL::RenderPipelineDescriptor*>(m_pipelineDescriptor);

        // Shaders
        desc->setVertexFunction(m_vertexShaderMtl->GetFunction());
        if (m_rasterizationEnabled)
            desc->setFragmentFunction(ResolveFragmentFunction());

        // label with the game-side shader identity so GPU traces are searchable
        desc->setLabel(ToNSString(fmt::format("rps vs{:016x} ps{:016x}",
            m_vertexShaderMtl->GetBaseHash(), m_pixelShaderMtl ? m_pixelShaderMtl->GetBaseHash() : 0)));
       	pipeline = CreatePipelineStateWithArchive(desc, &error);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto creationDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

   	if (error)
   	{
       	cemuLog_log(LogType::Force, "error creating render pipeline state: {}", error->localizedDescription()->utf8String());
		// identify the failing pipeline so it can be reproduced: shader hashes, the
		// framebuffer-fetch slots the PS reads (textureRenderTargetIndex) and the formats the
		// pipeline declared
		if (cemuLog_isLoggingEnabled(LogType::MetalBackend))
		{
			std::string fmts;
			for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
				fmts += fmt::format("{}:{:04x} ", i, (uint32)m_activeAttachmentsInfo.colorFormats[i]);
			std::string fetch;
			if (m_pixelShader)
			{
				// textureUnitList holds the actual unit numbers, so iterate over its entries
				for (uint32 i = 0; i < m_pixelShader->textureUnitListCount; i++)
				{
					uint8 u = m_pixelShader->textureUnitList[i];
					if (m_pixelShader->textureRenderTargetIndex[u] != 255)
						fetch += fmt::format("u{}->c{} ", (uint32)u, (uint32)m_pixelShader->textureRenderTargetIndex[u]);
				}
			}
			cemuLog_log(LogType::MetalBackend, "pipeline failure detail: PS {:016x}_{:016x} GS {} outMask {:04x} depth {:04x} stencil {} raster {} fetch[{}] declared[{}]",
				m_pixelShader ? m_pixelShader->baseHash : 0,
				m_pixelShader ? m_pixelShader->auxHash : 0,
				m_usesGeometryShader ? 1 : 0,
				m_pixelShader ? m_pixelShader->pixelColorOutputMask : 0,
				(uint32)m_activeAttachmentsInfo.depthFormat,
				m_activeAttachmentsInfo.hasStencil ? 1 : 0,
				m_rasterizationEnabled ? 1 : 0, fetch, fmts);
		}
		// the out-param error is autoreleased (metal-cpp does not retain it) - do not release it
	}

    // the session totals count every successful creation on any path (see the declarations above) -
    // what the binary archive exists to skip
    if (pipeline)
    {
        g_mtlPipelineCreationTimeNs += creationDuration;
        g_mtlPipelineCreationCount++;
    }

    if (showInOverlay)
	{
		if (isRenderThread)
			g_compiling_pipelines_syncTimeSum += creationDuration;
		else
			g_compiling_pipelines_async++;
		g_compiling_pipelines++;
	}

	if (!pipeline)
	{
		// signal failure so the broken pipeline is evicted from the cache and retried on the next draw
		m_pipelineObj.m_pipeline = nullptr;
		m_pipelineObj.compileFailed = true;
		return false;
	}

	// clear the failure flags BEFORE publishing the pipeline: the cache-eviction path on the
	// render thread evicts (and deletes) this entry whenever it observes compileFailed with a
	// non-null pipeline. The stores are atomic with the default sequential consistency, so this
	// store order is what the reader observes: an early reader can only see "still compiling"
	// (null pipeline + false flags), never a stale failure flag against the fresh pipeline
	m_pipelineObj.compileFailed = false;
	m_pipelineObj.permanentFailure = false;
	m_pipelineObj.m_pipeline = pipeline;

	return true;
}

// Resolves the fragment function for the pipeline, stripping orphaned color outputs when
// the active FBO does not attach all the slots the shader writes. Metal rejects a pipeline
// whose fragment function declares [[color(N)]] without a matching attachment (Vulkan
// silently ignores such outputs), which made depth-only passes with color-writing shaders
// fail pipeline creation forever and dropped their draws every frame
MTL::Function* MetalPipelineCompiler::ResolveFragmentFunction()
{
	// m_pixelShaderMtl is dereferenced before m_pixelShader is checked below, and the early
	// sanity check in Compile() short-circuits past a null m_pixelShaderMtl (it tests
	// "m_pixelShaderMtl && !GetFunction()"), so a null stage would be dereferenced here
	if (!m_pixelShaderMtl)
		return nullptr;
	MTL::Function* fragmentFunction = m_pixelShaderMtl->GetFunction();
	if (!m_pixelShader)
		return fragmentFunction;
	uint32 declaredMask = 0;
	for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
	{
		if (m_activeAttachmentsInfo.colorFormats[i] != Latte::E_GX2SURFFMT::INVALID_FORMAT)
			declaredMask |= (1u << i);
	}
	const uint32 orphanMask = m_pixelShader->pixelColorOutputMask & ~declaredMask;
	// orphan depth output: the shader writes [[depth(any)]] but the active FBO has no depth
	// attachment - Metal rejects the pipeline (Vulkan silently ignores orphan depth exports),
	// which made such passes fail pipeline creation and drop their draws every frame
	const bool orphanDepth = m_pixelShader->depthMask && m_activeAttachmentsInfo.depthFormat == Latte::E_GX2SURFFMT::INVALID_FORMAT;
	if (orphanMask == 0 && !orphanDepth)
		return fragmentFunction;
	// the combined variant removes both the orphan colors and the depth output in one
	// surgery - stripping only one would leave the other orphan output declared
	MTL::Function* stripped = m_pixelShaderMtl->GetStrippedVariant(orphanMask, orphanDepth);
	if (stripped)
	{
		cemuLog_logOnce(LogType::Force, "stripping orphan outputs (colors {:05x} depth {}) from PS {:016x}_{:016x} (depth-only or partial MRT pass)", orphanMask, orphanDepth ? 1 : 0, m_pixelShader->baseHash, m_pixelShader->auxHash);
		return stripped;
	}
	return fragmentFunction;
}

void MetalPipelineCompiler::InitFromStateRender(const LatteFetchShader* fetchShader, const LatteDecompilerShader* vertexShader, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr)
{
	// Render pipeline state
	MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();

    // Vertex descriptor
    if (!fetchShader->mtlFetchVertexManually)
    {
    	NS_STACK_SCOPED MTL::VertexDescriptor* vertexDescriptor = MTL::VertexDescriptor::alloc()->init();
    	for (auto& bufferGroup : fetchShader->bufferGroups)
    	{
    		std::optional<LatteConst::VertexFetchType2> fetchType;

    		uint32 minBufferStride = 0;
    		for (sint32 j = 0; j < bufferGroup.attribCount; ++j)
    		{
    			auto& attr = bufferGroup.attrib[j];

    			uint32 semanticId = vertexShader->resourceMapping.attributeMapping[attr.semanticId];
    			if (semanticId == (uint32)-1)
    				continue; // attribute not used?

    			auto attribute = vertexDescriptor->attributes()->object(semanticId);
    			attribute->setOffset(attr.offset);
    			attribute->setBufferIndex(GET_MTL_VERTEX_BUFFER_INDEX(attr.attributeBufferIndex));
    			attribute->setFormat(GetMtlVertexFormat(attr.format));

    			minBufferStride = std::max(minBufferStride, attr.offset + GetMtlVertexFormatSize(attr.format));

    			if (fetchType.has_value())
    				cemu_assert_debug(fetchType == attr.fetchType);
    			else
    				fetchType = attr.fetchType;

    			if (attr.fetchType == LatteConst::INSTANCE_DATA)
    			{
    				cemu_assert_debug(attr.aluDivisor == 1); // other divisor not yet supported
    			}
    		}

    		uint32 bufferIndex = bufferGroup.attributeBufferIndex;
    		uint32 bufferBaseRegisterIndex = mmSQ_VTX_ATTRIBUTE_BLOCK_START + bufferIndex * 7;
    		uint32 bufferStride = (lcr.GetRawView()[bufferBaseRegisterIndex + 2] >> 11) & 0xFFFF;

    		auto layout = vertexDescriptor->layouts()->object(GET_MTL_VERTEX_BUFFER_INDEX(bufferIndex));
    		if (bufferStride == 0)
    		{
    		    // Buffer stride cannot be zero, let's use the minimum stride
    			bufferStride = minBufferStride;

    			// Additionally, constant vertex function must be used
    			layout->setStepFunction(MTL::VertexStepFunctionConstant);
    			layout->setStepRate(0);
    		}
    		else
    		{
      		if (!fetchType.has_value() || fetchType == LatteConst::VertexFetchType2::VERTEX_DATA)
     			layout->setStepFunction(MTL::VertexStepFunctionPerVertex);
      		else if (fetchType == LatteConst::VertexFetchType2::INSTANCE_DATA)
     			layout->setStepFunction(MTL::VertexStepFunctionPerInstance);
      		else
      		{
      		    cemuLog_log(LogType::Force, "unimplemented vertex fetch type {}", (uint32)fetchType.value());
     			cemu_assert(false);
      		}
    		}
    		bufferStride = Align(bufferStride, 4);
    		layout->setStride(bufferStride);
    	}

    	desc->setVertexDescriptor(vertexDescriptor);
    }

	SetFragmentState(desc, lastUsedAttachmentsInfo, activeAttachmentsInfo, m_rasterizationEnabled, lcr);

	m_pipelineDescriptor = desc;
}

void MetalPipelineCompiler::InitFromStateMesh(const LatteFetchShader* fetchShader, const MetalAttachmentsInfo& lastUsedAttachmentsInfo, const MetalAttachmentsInfo& activeAttachmentsInfo, const LatteContextRegister& lcr)
{
	// Render pipeline state
	MTL::MeshRenderPipelineDescriptor* desc = MTL::MeshRenderPipelineDescriptor::alloc()->init();

	SetFragmentState(desc, lastUsedAttachmentsInfo, activeAttachmentsInfo, m_rasterizationEnabled, lcr);

	m_pipelineDescriptor = desc;
}
