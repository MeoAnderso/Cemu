#include "Cafe/HW/Latte/Renderer/Metal/MetalRenderer.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalMemoryManager.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/RendererShaderMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/CachedFBOMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalOutputShaderCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalPipelineCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalDepthStencilCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalSamplerCache.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureReadbackMtl.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalVoidVertexPipeline.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalQuery.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteToMtl.h"
#include "Cafe/HW/Latte/Renderer/SpirvCompiler.h"
#include "Cafe/HW/Latte/Renderer/Metal/UtilityShaderSource.h"

#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/LatteIndices.h"
#include "Cafe/HW/Latte/Core/LatteBufferCache.h"
#include "CafeSystem.h"
#include "Cemu/Logging/CemuLogging.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/Core/LatteConst.h"
#include "config/CemuConfig.h"

#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cstdio>

#define IMGUI_IMPL_METAL_CPP
#include "imgui/imgui_extension.h"
#include "imgui/imgui_impl_metal.h"

#define EVENT_VALUE_WRAP 4096

extern bool hasValidFramebufferAttached;

float supportBufferData[512 * 4];

// Defined in the Common renderer
void LatteDraw_handleSpecialState8_clearAsDepth();

std::vector<MetalRenderer::DeviceInfo> MetalRenderer::GetDevices()
{
    NS_STACK_SCOPED auto devices = MTL::CopyAllDevices();
    std::vector<MetalRenderer::DeviceInfo> result;
    result.reserve(devices->count());
    for (uint32 i = 0; i < devices->count(); i++)
    {
        MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
        result.push_back({std::string(device->name()->utf8String()), device->registryID()});
    }

    return result;
}

static const char* StageLetter(LatteConst::ShaderType shaderType)
{
	switch (shaderType)
	{
	case LatteConst::ShaderType::Vertex: return "VS";
	case LatteConst::ShaderType::Pixel: return "PS";
	case LatteConst::ShaderType::Geometry: return "GS";
	default: return "??";
	}
}

// One line describing the most recent copy into a texture, for the sampled-mip-chain diagnostics: what
// the copy connected, at what sizes, and whether it stayed inside the texture (the game building its
// own chain) or brought content in from another texture (a preservation copy carrying a previous
// incarnation's content). This is what explains the provenance verdict in the same log line.
//
// Captured as fields rather than a formatted string: the diagnostic call sites sit on a per-draw path
// and are normally reached with the channel off, and a string argument would be built on every call
// because C++ evaluates call arguments before the callee can decline to log
struct LastCopyDescription
{
	bool recorded = false;
	uint32 srcMip = 0, srcWidth = 0, srcHeight = 0;
	uint32 dstMip = 0, dstWidth = 0, dstHeight = 0;
	uint32 copyWidth = 0, copyHeight = 0;
	bool sameTexture = false;
};

template <>
struct fmt::formatter<LastCopyDescription> : fmt::formatter<std::string_view>
{
	template <typename Context>
	auto format(const LastCopyDescription& d, Context& ctx) const
	{
		if (!d.recorded)
			return fmt::format_to(ctx.out(), "none recorded");
		return fmt::format_to(ctx.out(), "src mip{} {}x{} -> dst mip{} {}x{}, region {}x{}, {}",
			d.srcMip, d.srcWidth, d.srcHeight, d.dstMip, d.dstWidth, d.dstHeight,
			d.copyWidth, d.copyHeight, d.sameTexture ? "within the same texture" : "from another texture");
	}
};

static LastCopyDescription DescribeLastCopy(const LatteTextureMtl* texMtl)
{
	const auto& last = texMtl->GetLastCopyInfo();
	LastCopyDescription d;
	d.recorded = last.copyWidth != 0;
	d.srcMip = last.srcMip;
	d.srcWidth = last.srcWidth;
	d.srcHeight = last.srcHeight;
	d.dstMip = last.dstMip;
	d.dstWidth = last.dstWidth;
	d.dstHeight = last.dstHeight;
	d.copyWidth = last.copyWidth;
	d.copyHeight = last.copyHeight;
	d.sameTexture = last.sameTexture;
	return d;
}

void MetalRenderer::CaptureFrame()
{
    m_captureFrame = true;
}

MetalRenderer::MetalRenderer() : Renderer(RendererAPI::Metal)
{
    // initialize glslang for graphic pack shader translation
    SpirvCompiler_EnsureInitialized();

    // frame debuggers / GPU captures don't handle async-compiled (initially skipped) draws well
    // - Xcode's GPU capture sets METAL_CAPTURE_ENABLED in the launched process environment
    m_usingTracingTool = getenv("METAL_CAPTURE_ENABLED") != nullptr;

    // Options

    // Position invariance
    switch (g_current_game_profile->GetPositionInvariance())
    {
    case PositionInvariance::Auto:
        switch (CafeSystem::GetForegroundTitleId())
        {
        // Bayonetta
        case 0x0005000010157F00: // EUR
        case 0x0005000010157E00: // USA
        case 0x000500001014DB00: // JPN
        // Bayonetta 2
        case 0x0005000010172700: // EUR
        case 0x0005000010172600: // USA
        // Disney Planes
        case 0x0005000010136900: // EUR
        case 0x0005000010136A00: // EUR (TODO: check)
        case 0x0005000010136B00: // EUR (TODO: check)
        case 0x000500001011C500: // USA (TODO: check)
        // LEGO STAR WARS: The Force Awakens
        case 0x00050000101DAA00: // EUR
        case 0x00050000101DAB00: // USA
        // Mario Kart 8
        case 0x000500001010ED00: // EUR
        case 0x000500001010EC00: // USA
        case 0x000500001010EB00: // JPN
        case 0x0005000010183A00: // JPN (TODO: check)
        // Minecraft: Story Mode
        case 0x000500001020A300: // EUR
        case 0x00050000101E0100: // USA
        //case 0x000500001020a200: // USA
        // Ninja Gaiden 3: Razor's Edge
        case 0x0005000010110B00: // EUR
        case 0x0005000010139B00: // EUR (TODO: check)
        case 0x0005000010110A00: // USA
        case 0x0005000010110900: // JPN
        // Resident Evil: Revelations
        case 0x000500001012B400: // EUR
        case 0x000500001012CF00: // USA
        // Star Fox Zero
        case 0x00050000101B0500: // EUR
        case 0x0005000010201C00: // EUR (TODO: check)
        case 0x00050000101B0400: // USA
        case 0x0005000010201B00: // USA (TODO: check)
        // The Legend of Zelda: Breath of the Wild
        case 0x00050000101C9500: // EUR
        case 0x00050000101C9400: // USA
        case 0x00050000101C9300: // JPN
        // Wonderful 101
        case 0x0005000010135300: // EUR
        case 0x000500001012DC00: // USA
        case 0x0005000010116300: // JPN
        case 0x0005000010185600: // JPN (TODO: check)
            m_positionInvariance = true;
            break;
        default:
            m_positionInvariance = false;
            break;
        }
        break;
    case PositionInvariance::False:
        m_positionInvariance = false;
        break;
    case PositionInvariance::True:
        m_positionInvariance = true;
        break;
    }

    // Pick a device
    auto& config = GetConfig();
    const bool hasDeviceSet = config.mtl_graphic_device_uuid != 0;

#ifdef CEMU_DEBUG_ASSERT
    // Metal shader validation is controlled by the MTL_SHADER_VALIDATION environment variable,
    // which Metal evaluates when the first device is created - it cannot be toggled per-device
    // after the fact. CEMU_MTL_SHADER_VALIDATION=1 is a convenience alias that sets it early
    // enough. For full runtime checking, additionally enable Metal API Validation (Xcode scheme
    // Diagnostics > Metal > API Validation, or launch with METAL_DEVICE_WRAPPER_TYPE=1).
    // Debug builds only: this is developer tooling and must not mutate the user's environment
    // in release builds
    if (getenv("CEMU_MTL_SHADER_VALIDATION") != nullptr)
        setenv("MTL_SHADER_VALIDATION", "1", 1);
#endif

    // If a device is set, try to find it
    if (hasDeviceSet)
    {
        NS_STACK_SCOPED auto devices = MTL::CopyAllDevices();
        for (uint32 i = 0; i < devices->count(); i++)
        {
            MTL::Device* device = static_cast<MTL::Device*>(devices->object(i));
            if (device->registryID() == config.mtl_graphic_device_uuid)
            {
                m_device = device;
                break;
            }
        }
    }

    if (!m_device)
    {
        if (hasDeviceSet)
        {
            cemuLog_log(LogType::Force, "The selected GPU ({}) could not be found. Using the system default device.", config.mtl_graphic_device_uuid);
            config.mtl_graphic_device_uuid = 0;
        }
        // Use the system default device
        m_device = MTL::CreateSystemDefaultDevice();
    }

    if (getenv("MTL_SHADER_VALIDATION") != nullptr)
        cemuLog_log(LogType::Force, "Metal shader validation is enabled (expect reduced performance)");

    // Vendor
    const char* deviceName = m_device->name()->utf8String();
    if (memcmp(deviceName, "Apple", 5) == 0)
        m_vendor = GfxVendor::Apple;
    else if (memcmp(deviceName, "AMD", 3) == 0)
        m_vendor = GfxVendor::AMD;
    else if (memcmp(deviceName, "Intel", 5) == 0)
        m_vendor = GfxVendor::Intel;
    else if (memcmp(deviceName, "NVIDIA", 6) == 0)
        m_vendor = GfxVendor::Nvidia;
    else
        m_vendor = GfxVendor::Generic;

    m_selectedDeviceName = deviceName;

    // Feature support
    m_isAppleGPU = m_device->supportsFamily(MTL::GPUFamilyApple1);
    m_supportsFramebufferFetch = GetConfig().framebuffer_fetch.GetValue() ? m_device->supportsFamily(MTL::GPUFamilyApple2) : false;
    m_hasUnifiedMemory = m_device->hasUnifiedMemory();
    m_supportsMetal3 = m_device->supportsFamily(MTL::GPUFamilyMetal3);
    m_supportsMeshShaders = (m_supportsMetal3 && (m_vendor != GfxVendor::Intel || GetConfig().force_mesh_shaders.GetValue())); // Intel GPUs have issues with mesh shaders
    m_recommendedMaxVRAMUsage = m_device->recommendedMaxWorkingSetSize();
    m_pixelFormatSupport = MetalPixelFormatSupport(m_device);

    CheckForPixelFormatSupport(m_pixelFormatSupport);

    // One line naming the capabilities the renderer's fallback decisions are made from. Without it a
    // report of "this effect is wrong on Metal" cannot be told apart from "this effect is wrong on
    // this GPU" (framebuffer fetch in particular is device- and config-dependent)
    cemuLog_log(LogType::Force, "Metal backend: device \"{}\", unified memory {}, Metal3 {}, mesh shaders {}, framebuffer fetch {}",
        deviceName, m_hasUnifiedMemory ? "yes" : "no", m_supportsMetal3 ? "yes" : "no",
        m_supportsMeshShaders ? "yes" : "no", m_supportsFramebufferFetch ? "yes" : "no");

    // Command queue
    m_commandQueue = m_device->newCommandQueue();

    // Synchronization resources
    m_event = m_device->newEvent();

    // Resources
    NS_STACK_SCOPED MTL::SamplerDescriptor* samplerDescriptor = MTL::SamplerDescriptor::alloc()->init();
#ifdef CEMU_DEBUG_ASSERT
    samplerDescriptor->setLabel(GetLabel("Nearest sampler state", samplerDescriptor));
#endif
    m_nearestSampler = m_device->newSamplerState(samplerDescriptor);

    samplerDescriptor->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDescriptor->setMagFilter(MTL::SamplerMinMagFilterLinear);
#ifdef CEMU_DEBUG_ASSERT
    samplerDescriptor->setLabel(GetLabel("Linear sampler state", samplerDescriptor));
#endif
    m_linearSampler = m_device->newSamplerState(samplerDescriptor);

    // Null resources
    NS_STACK_SCOPED MTL::TextureDescriptor* textureDescriptor = MTL::TextureDescriptor::alloc()->init();
    textureDescriptor->setTextureType(MTL::TextureType1D);
    textureDescriptor->setWidth(1);
    textureDescriptor->setUsage(MTL::TextureUsageShaderRead);
    m_nullTexture1D = m_device->newTexture(textureDescriptor);
#ifdef CEMU_DEBUG_ASSERT
    m_nullTexture1D->setLabel(GetLabel("Null texture 1D", m_nullTexture1D));
#endif

    textureDescriptor->setTextureType(MTL::TextureType2D);
    textureDescriptor->setHeight(1);
    textureDescriptor->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
    m_nullTexture2D = m_device->newTexture(textureDescriptor);
#ifdef CEMU_DEBUG_ASSERT
    m_nullTexture2D->setLabel(GetLabel("Null texture 2D", m_nullTexture2D));
#endif

    // used when a depth texture cannot be mirrored for depth-as-data sampling (see
    // depthCopy_ensureColorCopy) and the shader expects a 2D-array binding
    textureDescriptor->setTextureType(MTL::TextureType2DArray);
    textureDescriptor->setArrayLength(1);
    m_nullTexture2DArray = m_device->newTexture(textureDescriptor);
#ifdef CEMU_DEBUG_ASSERT
    m_nullTexture2DArray->setLabel(GetLabel("Null texture 2DArray", m_nullTexture2DArray));
#endif

    m_memoryManager = new MetalMemoryManager(this);
    m_outputShaderCache = new MetalOutputShaderCache(this);
    m_pipelineCache = new MetalPipelineCache(this);
    m_depthStencilCache = new MetalDepthStencilCache(this);
    m_samplerCache = new MetalSamplerCache(this);

    // Lower the commit treshold when buffer cache needs reduced latency
    if (m_memoryManager->NeedsReducedLatency())
        m_defaultCommitTreshlod = 64;
    else
        m_defaultCommitTreshlod = 196;

    // Occlusion queries
    m_occlusionQuery.m_resultBuffer = m_device->newBuffer(OCCLUSION_QUERY_POOL_SIZE * sizeof(uint64), MTL::ResourceStorageModeShared);
#ifdef CEMU_DEBUG_ASSERT
    m_occlusionQuery.m_resultBuffer->setLabel(GetLabel("Occlusion query result buffer", m_occlusionQuery.m_resultBuffer));
#endif
    m_occlusionQuery.m_resultsPtr = (uint64*)m_occlusionQuery.m_resultBuffer->contents();

    // Reset vertex and uniform buffers
   	for (uint32 i = 0; i < MAX_MTL_VERTEX_BUFFERS; i++)
        m_state.m_vertexBufferOffsets[i] = INVALID_OFFSET;

   	for (uint32 i = 0; i < METAL_SHADER_TYPE_TOTAL; i++)
    {
        for (uint32 j = 0; j < MAX_MTL_BUFFERS; j++)
            m_state.m_uniformBufferOffsets[i][j] = INVALID_OFFSET;
    }

    // Utility shader library

    // Create the library
    NS::Error* error = nullptr;
	NS_STACK_SCOPED MTL::Library* utilityLibrary = m_device->newLibrary(ToNSString(utilityShaderSource), nullptr, &error);
	if (error)
    {
        cemuLog_log(LogType::Force, "failed to create utility library (error: {})", error->localizedDescription()->utf8String());
    }

    // Pipelines
    NS_STACK_SCOPED MTL::Function* vertexFullscreenFunction = utilityLibrary->newFunction(ToNSString("vertexFullscreen"));
    NS_STACK_SCOPED MTL::Function* fragmentCopyDepthToColorFunction = utilityLibrary->newFunction(ToNSString("fragmentCopyDepthToColor"));
    NS_STACK_SCOPED MTL::Function* fragmentCopyColorToDepthFunction = utilityLibrary->newFunction(ToNSString("fragmentCopyColorToDepth"));

    m_copyDepthToColorDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    m_copyDepthToColorDesc->setVertexFunction(vertexFullscreenFunction);
    m_copyDepthToColorDesc->setFragmentFunction(fragmentCopyDepthToColorFunction);

    m_copyColorToDepthDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    m_copyColorToDepthDesc->setVertexFunction(vertexFullscreenFunction);
    m_copyColorToDepthDesc->setFragmentFunction(fragmentCopyColorToDepthFunction);

    // Void vertex pipelines
    if (m_isAppleGPU)
        m_copyBufferToBufferPipeline = new MetalVoidVertexPipeline(this, utilityLibrary, "vertexCopyBufferToBuffer");

    m_occlusionQuery.m_lastCommandBuffer = nullptr;
}

MetalRenderer::~MetalRenderer()
{
    if (m_isAppleGPU)
        delete m_copyBufferToBufferPipeline;
    //delete m_copyTextureToTexturePipeline;
    //delete m_restrideBufferPipeline;

    m_copyDepthToColorDesc->release();
    m_copyColorToDepthDesc->release();
    for (const auto [key, pipeline] : m_copySurfacePipelines)
        pipeline->release();
    if (m_copyDepthState)
        m_copyDepthState->release();

    delete m_outputShaderCache;
    delete m_pipelineCache;
    delete m_depthStencilCache;
    delete m_samplerCache;
    delete m_memoryManager;

    m_nullTexture1D->release();
    m_nullTexture2D->release();
    m_nullTexture2DArray->release();

    for (auto& [texture, shadowCopy] : m_feedbackShadowCopies)
    {
        if (shadowCopy.texture)
            shadowCopy.texture->release();
    }
    m_feedbackShadowCopies.clear();

    m_nearestSampler->release();
    m_linearSampler->release();

    if (m_readbackBuffer)
        m_readbackBuffer->release();

    if (m_textureCopyStagingBuffer)
        m_textureCopyStagingBuffer->release();

    if (m_meshIndexDummyBuffer)
        m_meshIndexDummyBuffer->release();

    if (m_xfbRingBuffer)
        m_xfbRingBuffer->release();

    m_occlusionQuery.m_resultBuffer->release();

    if (m_depthColorCopyPipeline)
        m_depthColorCopyPipeline->release();

    // created and drained on the Latte thread (SwapBuffers/GetCommandBuffer) - this destructor
    // also runs on the Latte thread (LatteThread_Exit), so the release is thread-affine
    if (m_frameAutoreleasePool)
        m_frameAutoreleasePool->release();

    m_event->release();

    m_commandQueue->release();
    m_device->release();
}

void MetalRenderer::InitializeLayer(const Vector2i& size, bool mainWindow)
{
    auto& layer = GetLayer(mainWindow);
    layer = MetalLayerHandle(m_device, size, mainWindow);
    layer.GetLayer()->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
}

void MetalRenderer::ShutdownLayer(bool mainWindow)
{
    GetLayer(mainWindow) = MetalLayerHandle();
}

void MetalRenderer::ResizeLayer(const Vector2i& size, bool mainWindow)
{
    GetLayer(mainWindow).Resize(size);
}

void MetalRenderer::Initialize()
{
    Renderer::Initialize();
    RendererShaderMtl::Initialize();
}

void MetalRenderer::Shutdown()
{
    // TODO: should shutdown both layers
    ImGui_ImplMetal_Shutdown();
    CommitCommandBuffer();
    Renderer::Shutdown();
    RendererShaderMtl::Shutdown();
}

bool MetalRenderer::IsPadWindowActive()
{
    return (GetLayer(false).GetLayer() != nullptr);
}

bool MetalRenderer::GetVRAMInfo(int& usageInMB, int& totalInMB) const
{
    // Subtract host memory from total VRAM, since it's shared with the CPU
    usageInMB = (m_device->currentAllocatedSize() - m_memoryManager->GetHostAllocationSize()) / 1024 / 1024;
    totalInMB = m_recommendedMaxVRAMUsage / 1024 / 1024;

    return true;
}

void MetalRenderer::ClearColorbuffer(bool padView)
{
    if (!AcquireDrawable(!padView))
        return;

    ClearColorTextureInternal(GetLayer(!padView).GetDrawable()->texture(), 0, 0, 0.0f, 0.0f, 0.0f, 0.0f);
}

void MetalRenderer::DrawEmptyFrame(bool mainWindow)
{
    if (!BeginFrame(mainWindow))
		return;
	SwapBuffers(mainWindow, !mainWindow);
}

void MetalRenderer::SwapBuffers(bool swapTV, bool swapDRC)
{
    if (swapTV)
        SwapBuffer(true);
    if (swapDRC)
        SwapBuffer(false);

    // Reset the command buffers (they are released by TemporaryBufferAllocator)
    CommitCommandBuffer();

    // A screenshot recorded during this frame is read back here: the frame's command buffer is now
    // committed and nothing is encoding, which is the only point where the blit can be committed and
    // waited on without disturbing the frame
    ProcessPendingScreenshot();

    // Debug
    m_performanceMonitor.ResetPerFrameData();

    // GPU capture
    if (m_capturing)
    {
        EndCapture();
    }
    else if (m_captureFrame)
    {
        StartCapture();
        m_captureFrame = false;
    }

    // Drain the frame pool and re-open it for the next frame (see the member comment). Everything
    // the frame autoreleased on this thread - labels, drawable internals, autoreleases inside Metal
    // API calls - is released here, which is the only thing that bounds them
    if (m_frameAutoreleasePool)
    {
        m_frameAutoreleasePool->release();
        m_frameAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    }
}

void MetalRenderer::HandleScreenshotRequest(LatteTextureView* texView, bool padView) {
	if (!m_screenshot_requested && m_screenshot_state == ScreenshotState::None)
		return;

	if (m_mainLayer.GetDrawable())
	{
		// we already took a pad view screenshow and want a main window screenshot
		if (m_screenshot_state == ScreenshotState::Main && padView)
			return;

		if (m_screenshot_state == ScreenshotState::Pad && !padView)
			return;

		// remember which screenshot is left to take
		if (m_screenshot_state == ScreenshotState::None)
			m_screenshot_state = padView ? ScreenshotState::Main : ScreenshotState::Pad;
		else
			m_screenshot_state = ScreenshotState::None;
	}
	else
		m_screenshot_state = ScreenshotState::None;

	auto texMtl = static_cast<LatteTextureMtl*>(texView->baseTexture);

	int width, height;
	texMtl->GetEffectiveSize(width, height, 0);

	// Record only. This runs inside LatteRenderTarget_copyToBackbuffer with the frame's render pass
	// open, so neither the blit nor a wait can happen here - see ProcessPendingScreenshot, which does
	// both at the end of SwapBuffers where the frame's command buffer has just been committed
	m_pendingScreenshot.texMtl = texMtl;
	m_pendingScreenshot.width = (uint32)width;
	m_pendingScreenshot.height = (uint32)height;
	m_pendingScreenshot.pixelFormat = texMtl->GetTexture()->pixelFormat();
	m_pendingScreenshot.padView = padView;
	m_hasPendingScreenshot = true;
}

void MetalRenderer::ProcessPendingScreenshot()
{
	if (!m_hasPendingScreenshot)
		return;
	m_hasPendingScreenshot = false;

	const PendingScreenshot req = m_pendingScreenshot;
	m_pendingScreenshot = {};
	if (!req.texMtl || req.width == 0 || req.height == 0)
		return;

	const uint32 bytesPerRow = GetMtlTextureBytesPerRow(req.texMtl->format, req.texMtl->isDepth, req.width);
	const uint32 size = GetMtlTextureBytesPerImage(req.texMtl->format, req.texMtl->isDepth, req.height, bytesPerRow);

	// Renderer-owned destination, grown as needed: the staging allocator reclaims its buffers the
	// moment the command buffer using them completes, which is precisely when this reads them
	if (!m_screenshotBuffer || m_screenshotBufferSize < size)
	{
		if (m_screenshotBuffer)
			m_screenshotBuffer->release();
		m_screenshotBuffer = m_device->newBuffer(size, MTL::ResourceStorageModeShared);
		m_screenshotBufferSize = size;
		if (!m_screenshotBuffer)
		{
			cemuLog_log(LogType::Force, "screenshot: failed to allocate a {} byte readback buffer", size);
			return;
		}
		m_screenshotBuffer->setLabel(GetLabel("Screenshot readback buffer", m_screenshotBuffer));
	}

	// Its own command buffer, committed and waited on here. SwapBuffers has already committed the
	// frame's, so nothing is open and this neither disturbs the frame nor reads unwritten memory
	NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
	{
		MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
		auto blitCommandEncoder = commandBuffer->blitCommandEncoder();
		blitCommandEncoder->copyFromTexture(req.texMtl->GetTexture(), 0, 0, MTL::Origin(0, 0, 0), MTL::Size(req.width, req.height, 1), m_screenshotBuffer, 0, bytesPerRow, 0);
		blitCommandEncoder->endEncoding();
		commandBuffer->commit();
		commandBuffer->waitUntilCompleted();
	}

	const uint8* memPtr = (const uint8*)m_screenshotBuffer->contents();
	bool formatValid = true;
	std::vector<uint8> rgb_data;
	rgb_data.reserve(3 * (size_t)req.width * req.height);

	// TODO: implement more formats
	switch (req.pixelFormat)
	{
	case MTL::PixelFormatRGBA8Unorm:
		for (const uint8* ptr = memPtr; ptr < memPtr + size; ptr += 4)
		{
			rgb_data.emplace_back(ptr[0]);
			rgb_data.emplace_back(ptr[1]);
			rgb_data.emplace_back(ptr[2]);
		}
		break;
	case MTL::PixelFormatRGBA8Unorm_sRGB:
		for (const uint8* ptr = memPtr; ptr < memPtr + size; ptr += 4)
		{
			rgb_data.emplace_back(SRGBComponentToRGB(ptr[0]));
			rgb_data.emplace_back(SRGBComponentToRGB(ptr[1]));
			rgb_data.emplace_back(SRGBComponentToRGB(ptr[2]));
		}
		break;
	// 10:10:10:2 is the scan buffer format most Wii U titles use, and without these two cases the
	// screenshot is abandoned entirely ("Unsupported screenshot texture pixel format"). Vulkan does
	// not special-case them either, but it converts through an RGBA8 blit - which Metal's blit
	// encoder cannot do (it requires matching formats), so the unpacking happens here instead
	case MTL::PixelFormatRGB10A2Unorm:
	case MTL::PixelFormatBGR10A2Unorm:
		for (const uint8* ptr = memPtr; ptr < memPtr + size; ptr += 4)
		{
			const uint32 packed = (uint32)ptr[0] | ((uint32)ptr[1] << 8) | ((uint32)ptr[2] << 16) | ((uint32)ptr[3] << 24);
			const uint8 lo = (uint8)(((packed & 0x3FF) * 255 + 511) / 1023);
			const uint8 mid = (uint8)((((packed >> 10) & 0x3FF) * 255 + 511) / 1023);
			const uint8 hi = (uint8)((((packed >> 20) & 0x3FF) * 255 + 511) / 1023);
			const bool bgr = req.pixelFormat == MTL::PixelFormatBGR10A2Unorm;
			rgb_data.emplace_back(bgr ? hi : lo);
			rgb_data.emplace_back(mid);
			rgb_data.emplace_back(bgr ? lo : hi);
		}
		break;
	default:
		cemuLog_log(LogType::Force, "Unsupported screenshot texture pixel format {}", req.pixelFormat);
		formatValid = false;
		break;
	}

	pool->release();

	if (formatValid)
		SaveScreenshot(rgb_data, req.width, req.height, !req.padView);
}

void MetalRenderer::DrawBackbufferQuad(LatteTextureView* texView, RendererOutputShader* shader, bool useLinearTexFilter,
								sint32 imageX, sint32 imageY, sint32 imageWidth, sint32 imageHeight,
								bool padView, bool clearBackground)
{
    if (!AcquireDrawable(!padView))
        return;

    MTL::Texture* presentTexture = static_cast<LatteTextureViewMtl*>(texView)->GetRGBAView();

    // Create render pass
    auto& layer = GetLayer(!padView);

    NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
    colorAttachment->setTexture(layer.GetDrawable()->texture());
    colorAttachment->setLoadAction(clearBackground ? MTL::LoadActionClear : MTL::LoadActionLoad);
    colorAttachment->setStoreAction(MTL::StoreActionStore);

    auto renderCommandEncoder = GetTemporaryRenderCommandEncoder(renderPassDescriptor);

    // Get a render pipeline

    // Find out which shader we are using
    uint8 shaderIndex = 255;
    if (shader == RendererOutputShader::s_copy_shader) shaderIndex = 0;
    else if (shader == RendererOutputShader::s_bicubic_shader) shaderIndex = 1;
    else if (shader == RendererOutputShader::s_hermit_shader) shaderIndex = 2;
    else if (shader == RendererOutputShader::s_copy_shader_ud) shaderIndex = 3;
    else if (shader == RendererOutputShader::s_bicubic_shader_ud) shaderIndex = 4;
    else if (shader == RendererOutputShader::s_hermit_shader_ud) shaderIndex = 5;

    uint8 shaderType = shaderIndex % 3;

    // Get the render pipeline state
    auto renderPipelineState = m_outputShaderCache->GetPipeline(shader, shaderIndex, m_state.m_usesSRGB);

    // Draw to Metal layer
    renderCommandEncoder->setRenderPipelineState(renderPipelineState);
    renderCommandEncoder->setFragmentTexture(presentTexture, 0);
    renderCommandEncoder->setFragmentSamplerState((useLinearTexFilter ? m_linearSampler : m_nearestSampler), 0);

    // Set uniforms
    float outputSize[2] = {(float)imageWidth, (float)imageHeight};
    switch (shaderType)
    {
    case 2:
        renderCommandEncoder->setFragmentBytes(outputSize, sizeof(outputSize), 0);
        break;
    default:
        break;
    }

    renderCommandEncoder->setViewport(MTL::Viewport{(double)imageX, (double)imageY, (double)imageWidth, (double)imageHeight, 0.0, 1.0});
    renderCommandEncoder->setScissorRect(MTL::ScissorRect{(uint32)imageX, (uint32)imageY, (uint32)imageWidth, (uint32)imageHeight});

    renderCommandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

    EndEncoding();
}

bool MetalRenderer::BeginFrame(bool mainWindow)
{
    return AcquireDrawable(mainWindow);
}

void MetalRenderer::Flush(bool waitIdle)
{
    if (m_recordedDrawcalls > 0 || waitIdle)
        CommitCommandBuffer();

    if (waitIdle && m_executingCommandBuffers.size() != 0)
        m_executingCommandBuffers.back()->waitUntilCompleted();
}

void MetalRenderer::NotifyLatteCommandProcessorIdle()
{
    //if (m_commitOnIdle)
    //    CommitCommandBuffer();
}

bool MetalRenderer::ImguiBegin(bool mainWindow)
{
    if (!Renderer::ImguiBegin(mainWindow))
		return false;

	if (!AcquireDrawable(mainWindow))
		return false;

	EnsureImGuiBackend();

	// Check if the font texture needs to be built
	ImGuiIO& io = ImGui::GetIO();
    if (!io.Fonts->IsBuilt())
        ImGui_ImplMetal_CreateFontsTexture(m_device);

	auto& layer = GetLayer(mainWindow);

	// Render pass descriptor
	NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
    colorAttachment->setTexture(layer.GetDrawable()->texture());
    colorAttachment->setLoadAction(MTL::LoadActionLoad);
    colorAttachment->setStoreAction(MTL::StoreActionStore);

    // New frame
	ImGui_ImplMetal_NewFrame(renderPassDescriptor);
	ImGui_UpdateWindowInformation(mainWindow);
	ImGui::NewFrame();

	if (m_encoderType != MetalEncoderType::Render)
	    GetTemporaryRenderCommandEncoder(renderPassDescriptor);

	return true;
}

void MetalRenderer::ImguiEnd()
{
    EnsureImGuiBackend();

    if (m_encoderType != MetalEncoderType::Render)
    {
        cemuLog_logOnce(LogType::Force, "no render command encoder, cannot draw ImGui");
        return;
    }

    ImGui::Render();
	ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), GetCurrentCommandBuffer(), (MTL::RenderCommandEncoder*)m_commandEncoder);
	//ImGui::EndFrame();

	EndEncoding();
}

ImTextureID MetalRenderer::GenerateTexture(const std::vector<uint8>& data, const Vector2i& size)
{
    try
	{
		std::vector <uint8> tmp(size.x * size.y * 4);
		for (size_t i = 0; i < data.size() / 3; ++i)
		{
			tmp[(i * 4) + 0] = data[(i * 3) + 0];
			tmp[(i * 4) + 1] = data[(i * 3) + 1];
			tmp[(i * 4) + 2] = data[(i * 3) + 2];
			tmp[(i * 4) + 3] = 0xFF;
		}

		NS_STACK_SCOPED MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
		desc->setTextureType(MTL::TextureType2D);
		desc->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
		desc->setWidth(size.x);
		desc->setHeight(size.y);
		desc->setStorageMode(m_isAppleGPU ? MTL::StorageModeShared : MTL::StorageModeManaged);
		desc->setUsage(MTL::TextureUsageShaderRead);

		MTL::Texture* texture = m_device->newTexture(desc);

		// TODO: do a GPU copy?
		texture->replaceRegion(MTL::Region(0, 0, size.x, size.y), 0, 0, tmp.data(), size.x * 4, 0);

		return (ImTextureID)texture;
	}
	catch (const std::exception& ex)
	{
		cemuLog_log(LogType::Force, "can't generate imgui texture: {}", ex.what());
		return nullptr;
	}
}

void MetalRenderer::DeleteTexture(ImTextureID id)
{
    EnsureImGuiBackend();

    ((MTL::Texture*)id)->release();
}

void MetalRenderer::DeleteFontTextures()
{
    EnsureImGuiBackend();

    ImGui_ImplMetal_DestroyFontsTexture();
}

void MetalRenderer::AppendOverlayDebugInfo()
{
    ImGui::Text("--- GPU info ---");
    ImGui::Text("GPU                        %s", m_device->name()->utf8String());
    ImGui::Text("Is Apple GPU               %s", (m_isAppleGPU ? "yes" : "no"));
    ImGui::Text("Supports framebuffer fetch %s", (m_supportsFramebufferFetch ? "yes" : "no"));
    ImGui::Text("Has unified memory         %s", (m_hasUnifiedMemory ? "yes" : "no"));
    ImGui::Text("Supports Metal3            %s", (m_supportsMetal3 ? "yes" : "no"));

    ImGui::Text("--- Metal info ---");
    ImGui::Text("Render pipeline states     %zu", m_pipelineCache->GetPipelineCacheSize());

    ImGui::Text("--- Metal info (per frame) ---");
    ImGui::Text("Command buffers            %u", m_performanceMonitor.m_commandBuffers);
    ImGui::Text("Render passes              %u", m_performanceMonitor.m_renderPasses);
    ImGui::Text("Clears                     %u", m_performanceMonitor.m_clears);
    ImGui::Text("Manual vertex fetch draws  %u (mesh draws: %u)", m_performanceMonitor.m_manualVertexFetchDraws, m_performanceMonitor.m_meshDraws);
    ImGui::Text("Triangle fans              %u", m_performanceMonitor.m_triangleFans);

    ImGui::Text("--- Cache debug info ---");

	uint32 bufferCacheHeapSize = 0;
	uint32 bufferCacheAllocationSize = 0;
	uint32 bufferCacheNumAllocations = 0;

	LatteBufferCache_getStats(bufferCacheHeapSize, bufferCacheAllocationSize, bufferCacheNumAllocations);

	ImGui::Text("Buffer");
	ImGui::SameLine(60.0f);
	ImGui::Text("%06uKB / %06uKB Allocs: %u", (uint32)(bufferCacheAllocationSize + 1023) / 1024, ((uint32)bufferCacheHeapSize + 1023) / 1024, (uint32)bufferCacheNumAllocations);

	uint32 numBuffers;
	size_t totalSize, freeSize;

	m_memoryManager->GetStagingAllocator().GetStats(numBuffers, totalSize, freeSize);
	ImGui::Text("Staging");
	ImGui::SameLine(60.0f);
	ImGui::Text("%06uKB / %06uKB Buffers: %u", ((uint32)(totalSize - freeSize) + 1023) / 1024, ((uint32)totalSize + 1023) / 1024, (uint32)numBuffers);

	m_memoryManager->GetIndexAllocator().GetStats(numBuffers, totalSize, freeSize);
	ImGui::Text("Index");
	ImGui::SameLine(60.0f);
	ImGui::Text("%06uKB / %06uKB Buffers: %u", ((uint32)(totalSize - freeSize) + 1023) / 1024, ((uint32)totalSize + 1023) / 1024, (uint32)numBuffers);

	// Session fallback census. Shown here so it is visible live: the shutdown summary is the only
	// other channel and it is lost to anything but a graceful quit (SIGTERM is a bare _Exit).
	// Names come from the same descriptors the summary prints, so the two cannot disagree.
	ImGui::Text("--- Backend fallbacks (session) ---");
	{
		uint32 shown = 0;
		for (uint32 i = 0; i < (uint32)MetalDiagEvent::COUNT; i++)
		{
			const uint32 count = MetalDiag_GetCount((MetalDiagEvent)i);
			if (count == 0)
				continue;
			ImGui::Text("%7u  %s", count, MetalDiag_GetName((MetalDiagEvent)i));
			shown++;
		}
		if (shown == 0)
			ImGui::Text("none recorded");
	}
}

void MetalRenderer::LogDiagnosticsSummary()
{
	MetalDiag_LogSummary();
}

void MetalRenderer::renderTarget_setViewport(float x, float y, float width, float height, float nearZ, float farZ, bool halfZ)
{
    // halfZ is handled in the shader

    m_state.m_viewport = MTL::Viewport{x, y, width, height, nearZ, farZ};
}

void MetalRenderer::renderTarget_setScissor(sint32 scissorX, sint32 scissorY, sint32 scissorWidth, sint32 scissorHeight)
{
    m_state.m_scissor = MTL::ScissorRect{(uint32)scissorX, (uint32)scissorY, (uint32)scissorWidth, (uint32)scissorHeight};
}

LatteCachedFBO* MetalRenderer::rendertarget_createCachedFBO(uint64 key)
{
	return new CachedFBOMtl(this, key);
}

void MetalRenderer::rendertarget_deleteCachedFBO(LatteCachedFBO* cfbo)
{
	if (cfbo == (LatteCachedFBO*)m_state.m_activeFBO.m_fbo)
	    m_state.m_activeFBO = {nullptr};
}

void MetalRenderer::rendertarget_bindFramebufferObject(LatteCachedFBO* cfbo)
{
	m_state.m_activeFBO = {(CachedFBOMtl*)cfbo, MetalAttachmentsInfo((CachedFBOMtl*)cfbo)};
	m_state.m_fboChanged = true;
}

void* MetalRenderer::texture_acquireTextureUploadBuffer(uint32 size)
{
    return m_memoryManager->AcquireTextureUploadBuffer(size);
}

void MetalRenderer::texture_releaseTextureUploadBuffer(uint8* mem)
{
    m_memoryManager->ReleaseTextureUploadBuffer(mem);
}

TextureDecoder* MetalRenderer::texture_chooseDecodedFormat(Latte::E_GX2SURFFMT format, bool isDepth, Latte::E_DIM dim, uint32 width, uint32 height)
{
    return GetMtlPixelFormatInfo(format, isDepth).textureDecoder;
}

void MetalRenderer::texture_clearSlice(LatteTexture* hostTexture, sint32 sliceIndex, sint32 mipIndex)
{
    if (hostTexture->isDepth)
    {
        texture_clearDepthSlice(hostTexture, sliceIndex, mipIndex, true, hostTexture->hasStencil, 0.0f, 0);
    }
    else
    {
        texture_clearColorSlice(hostTexture, sliceIndex, mipIndex, 0.0f, 0.0f, 0.0f, 0.0f);
    }
}

MTL::BlitOption GetBlitOptionForTexture(const LatteTextureMtl* texture) {
    switch (texture->GetTexture()->pixelFormat()) {
        case MTL::PixelFormatDepth16Unorm:
        case MTL::PixelFormatDepth32Float:
            return MTL::BlitOptionDepthFromDepthStencil;

        case MTL::PixelFormatStencil8:
            return MTL::BlitOptionStencilFromDepthStencil;

        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
            // Can't copy both in one call — caller must specify.
            // Default to depth
            return MTL::BlitOptionDepthFromDepthStencil;

        default:
            return MTL::BlitOptionNone;
    }
}

bool IsDepthStencilFormat(MTL::PixelFormat format) {
    return format == MTL::PixelFormatDepth24Unorm_Stencil8 ||
           format == MTL::PixelFormatDepth32Float_Stencil8;
}

// TODO: do a cpu copy on Apple Silicon?
void MetalRenderer::texture_loadSlice(LatteTexture* hostTexture, sint32 width, sint32 height, sint32 depth, void* pixelData, sint32 sliceIndex, sint32 mipIndex, uint32 compressedImageSize)
{
    auto textureMtl = (LatteTextureMtl*)hostTexture;

    uint32 offsetZ = 0;
    if (textureMtl->Is3DTexture())
    {
        offsetZ = sliceIndex;
        sliceIndex = 0;
    }

    size_t bytesPerRow = GetMtlTextureBytesPerRow(textureMtl->format, textureMtl->isDepth, width);
    // No need to set bytesPerImage for 3D textures, since we always load just one slice
    //size_t bytesPerImage = GetMtlTextureBytesPerImage(textureMtl->GetFormat(), textureMtl->isDepth, height, bytesPerRow);
    //if (m_isAppleGPU)
    //{
    //    textureMtl->GetTexture()->replaceRegion(MTL::Region(0, 0, offsetZ, width, height, 1), mipIndex, sliceIndex, pixelData, bytesPerRow, 0);
    //}
    //else
    //{
    auto blitCommandEncoder = GetBlitCommandEncoder();

    // Allocate a temporary buffer
    auto& bufferAllocator = m_memoryManager->GetStagingAllocator();
    auto allocation = bufferAllocator.AllocateBufferMemory(compressedImageSize, 1);
    memcpy(allocation.memPtr, pixelData, compressedImageSize);
    bufferAllocator.FlushReservation(allocation);

    // Copy the data from the temporary buffer to the texture
	if (IsDepthStencilFormat(textureMtl->GetTexture()->pixelFormat())) {
        // Metal doesn't allow copying depth and stencil data at the same time, so we need to do two copies for combined depth/stencil formats
        blitCommandEncoder->copyFromBuffer(allocation.mtlBuffer, allocation.bufferOffset, bytesPerRow, 0, MTL::Size(width, height, 1), textureMtl->GetTexture(), sliceIndex, mipIndex, MTL::Origin(0, 0, offsetZ), MTL::BlitOptionDepthFromDepthStencil);
        blitCommandEncoder->copyFromBuffer(allocation.mtlBuffer, allocation.bufferOffset, bytesPerRow, 0, MTL::Size(width, height, 1), textureMtl->GetTexture(), sliceIndex, mipIndex, MTL::Origin(0, 0, offsetZ), MTL::BlitOptionStencilFromDepthStencil);
    } else {
        blitCommandEncoder->copyFromBuffer(allocation.mtlBuffer, allocation.bufferOffset, bytesPerRow, 0, MTL::Size(width, height, 1), textureMtl->GetTexture(), sliceIndex, mipIndex, MTL::Origin(0, 0, offsetZ), GetBlitOptionForTexture(textureMtl));
    }
    //}
}

void MetalRenderer::texture_clearColorSlice(LatteTexture* hostTexture, sint32 sliceIndex, sint32 mipIndex, float r, float g, float b, float a)
{
    if (!FormatIsRenderable(hostTexture->format))
    {
        cemuLog_logOnce(LogType::Force, "cannot clear color texture with format {}, because it's not renderable", hostTexture->format);
        return;
    }

    auto mtlTexture = static_cast<LatteTextureMtl*>(hostTexture)->GetTexture();

    ClearColorTextureInternal(mtlTexture, sliceIndex, mipIndex, r, g, b, a);
}

void MetalRenderer::texture_clearDepthSlice(LatteTexture* hostTexture, uint32 sliceIndex, sint32 mipIndex, bool clearDepth, bool clearStencil, float depthValue, uint32 stencilValue)
{
    clearStencil = (clearStencil && GetMtlPixelFormatInfo(hostTexture->format, true).hasStencil);
    if (!clearDepth && !clearStencil)
    {
        cemuLog_logOnce(LogType::Force, "skipping depth/stencil clear");
        return;
    }

    auto mtlTexture = static_cast<LatteTextureMtl*>(hostTexture)->GetTexture();

    NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    if (clearDepth)
    {
        auto depthAttachment = renderPassDescriptor->depthAttachment();
        depthAttachment->setTexture(mtlTexture);
        depthAttachment->setClearDepth(depthValue);
        depthAttachment->setLoadAction(MTL::LoadActionClear);
        depthAttachment->setStoreAction(MTL::StoreActionStore);
        depthAttachment->setSlice(sliceIndex);
        depthAttachment->setLevel(mipIndex);
    }
    if (clearStencil)
    {
        auto stencilAttachment = renderPassDescriptor->stencilAttachment();
        stencilAttachment->setTexture(mtlTexture);
        stencilAttachment->setClearStencil(stencilValue);
        stencilAttachment->setLoadAction(MTL::LoadActionClear);
        stencilAttachment->setStoreAction(MTL::StoreActionStore);
        stencilAttachment->setSlice(sliceIndex);
        stencilAttachment->setLevel(mipIndex);
    }

    GetTemporaryRenderCommandEncoder(renderPassDescriptor);
    EndEncoding();

    // Debug
    m_performanceMonitor.m_clears++;
}

LatteTexture* MetalRenderer::texture_createTextureEx(Latte::E_DIM dim, MPTR physAddress, MPTR physMipAddress, Latte::E_GX2SURFFMT format, uint32 width, uint32 height, uint32 depth, uint32 pitch, uint32 mipLevels, uint32 swizzle, Latte::E_HWTILEMODE tileMode, bool isDepth)
{
    return new LatteTextureMtl(this, dim, physAddress, physMipAddress, format, width, height, depth, pitch, mipLevels, swizzle, tileMode, isDepth);
}

void MetalRenderer::texture_setLatteTexture(LatteTextureView* textureView, uint32 textureUnit)
{
    m_state.m_textures[textureUnit] = static_cast<LatteTextureViewMtl*>(textureView);
}

void MetalRenderer::texture_copyImageSubData(LatteTexture* src, sint32 srcMip, sint32 effectiveSrcX, sint32 effectiveSrcY, sint32 srcSlice, LatteTexture* dst, sint32 dstMip, sint32 effectiveDstX, sint32 effectiveDstY, sint32 dstSlice, sint32 effectiveCopyWidth, sint32 effectiveCopyHeight, sint32 srcDepth_)
{
    const auto srcMtlFormat = GetMtlPixelFormat(src->format, src->isDepth);
    const auto dstMtlFormat = GetMtlPixelFormat(dst->format, dst->isDepth);
    const auto& srcInfo = GetMtlPixelFormatInfo(src->format, src->isDepth);
    const auto& dstInfo = GetMtlPixelFormatInfo(dst->format, dst->isDepth);

    // Copy provenance (see LatteTextureMtl::MarkCopyMipsWritten). Classify the copy by what it does
    // to the destination level, which is what decides whether that level's content counts as the
    // game's own: the level fully covered by a same-sized source level (carried over 1:1), fully
    // covered from a differently sized source (a deliberate resample - how a game builds its own
    // mip chain by copying a larger level into a smaller one), or only partly covered (a crop
    // pasted into the level, leaving the rest of it as whatever was there before)
    sint32 srcLevelWidth = 0, srcLevelHeight = 0, dstLevelWidth = 0, dstLevelHeight = 0;
    src->GetEffectiveSize(srcLevelWidth, srcLevelHeight, srcMip);
    dst->GetEffectiveSize(dstLevelWidth, dstLevelHeight, dstMip);
    const LatteTextureMtl::CopyProvenance copyProvenance = [&]() {
        using CopyProvenance = LatteTextureMtl::CopyProvenance;
        if (effectiveDstX != 0 || effectiveDstY != 0)
            return CopyProvenance::Partial;
        if (effectiveCopyWidth != dstLevelWidth || effectiveCopyHeight != dstLevelHeight)
            return CopyProvenance::Partial;
        if (effectiveSrcX != 0 || effectiveSrcY != 0)
            return CopyProvenance::Partial;
        if (effectiveCopyWidth != srcLevelWidth || effectiveCopyHeight != srcLevelHeight)
            return CopyProvenance::Resampled;
        return CopyProvenance::Matched;
    }();
    // Source size seems to apply to the destination texture as well, therefore we need to adjust it when block size doesn't match
    Uvec2 srcBlockTexelSize = GetMtlPixelFormatInfo(src->format, src->isDepth).blockTexelSize;
    Uvec2 dstBlockTexelSize = GetMtlPixelFormatInfo(dst->format, dst->isDepth).blockTexelSize;
    if (srcBlockTexelSize.x != dstBlockTexelSize.x || srcBlockTexelSize.y != dstBlockTexelSize.y)
    {
        uint32 multX = (srcBlockTexelSize.x > dstBlockTexelSize.x ? srcBlockTexelSize.x / dstBlockTexelSize.x : dstBlockTexelSize.x / srcBlockTexelSize.x);
        effectiveCopyWidth *= multX;

        uint32 multY = (srcBlockTexelSize.y > dstBlockTexelSize.y ? srcBlockTexelSize.y / dstBlockTexelSize.y : dstBlockTexelSize.y / srcBlockTexelSize.y);
        effectiveCopyHeight *= multY;
    }

    // The block-size adjustment above rescales the extent, but that same extent is also used verbatim
    // as the SOURCE region's size. When the source and destination disagree on block size, the scaled
    // extent can exceed the source mip, so sourceOrigin + sourceSize overruns the texture and Metal
    // aborts the entire blit under API validation (without it the copy merely fails and the
    // destination silently keeps its previous contents). Clamp the region to both mips so the copy
    // degrades to the overlapping area instead - partial copies are already handled (the provenance
    // below records them as Partial, marking the destination chain untrusted). This is a safety net,
    // not a fix for the scaling rule itself
    // The limits come from the MTL resources, not from LatteTexture::GetMipWidth(): Metal validates
    // sourceOrigin + sourceSize against the texture it was handed, and under a graphic-pack
    // resolution overwrite the Metal texture's dimensions differ from the logical ones - clamping
    // against the logical size would leave the abort in place
    MTL::Texture* mtlDimSrc = static_cast<LatteTextureMtl*>(src)->GetTexture();
    MTL::Texture* mtlDimDst = static_cast<LatteTextureMtl*>(dst)->GetTexture();
    // MTL::Texture::width()/height() report the BASE level, but the blits below address srcMip/dstMip
    // and Metal validates the region against the LEVEL's dimensions. Halving gives the level size - the
    // same relationship the Metal mip chain has, including under a resolution overwrite, where the base
    // size differs from the logical one and the LatteTexture accessors cannot be used either
    auto mtlLevelSize = [](MTL::Texture* tex, sint32 mip) -> Uvec2 {
        const uint32 shift = std::min<uint32>((uint32)std::max(mip, 0), 31u);
        return { std::max(1u, (uint32)tex->width() >> shift), std::max(1u, (uint32)tex->height() >> shift) };
    };
    const Uvec2 srcLevelSize = mtlLevelSize(mtlDimSrc, srcMip);
    const Uvec2 dstLevelSize = mtlLevelSize(mtlDimDst, dstMip);
    const sint32 srcMipW = (sint32)srcLevelSize.x;
    const sint32 srcMipH = (sint32)srcLevelSize.y;
    const sint32 dstMipW = (sint32)dstLevelSize.x;
    const sint32 dstMipH = (sint32)dstLevelSize.y;
    {
        if (effectiveSrcX < 0 || effectiveSrcY < 0 || effectiveDstX < 0 || effectiveDstY < 0 ||
            effectiveSrcX + effectiveCopyWidth > srcMipW || effectiveSrcY + effectiveCopyHeight > srcMipH ||
            effectiveDstX + effectiveCopyWidth > dstMipW || effectiveDstY + effectiveCopyHeight > dstMipH)
        {
            const sint32 clampedW = std::min({ effectiveCopyWidth, srcMipW - effectiveSrcX, dstMipW - effectiveDstX });
            const sint32 clampedH = std::min({ effectiveCopyHeight, srcMipH - effectiveSrcY, dstMipH - effectiveDstY });
            cemuLog_logOnce(LogType::Force,
                "texture_copyImageSubData: copy region {}x{} at src ({},{}) / dst ({},{}) exceeds the mips (src {:04x} {}x{}, dst {:04x} {}x{}) - clamping to {}x{}",
                effectiveCopyWidth, effectiveCopyHeight, effectiveSrcX, effectiveSrcY, effectiveDstX, effectiveDstY,
                (uint32)src->format, srcMipW, srcMipH, (uint32)dst->format, dstMipW, dstMipH, clampedW, clampedH);
            if (clampedW <= 0 || clampedH <= 0)
                return;
            effectiveCopyWidth = clampedW;
            effectiveCopyHeight = clampedH;
        }
    }

    auto blitCommandEncoder = GetBlitCommandEncoder();

    auto mtlSrc = static_cast<LatteTextureMtl*>(src)->GetTexture();
    auto mtlDst = static_cast<LatteTextureMtl*>(dst)->GetTexture();

    uint32 srcBaseLayer = 0;
    uint32 dstBaseLayer = 0;
    uint32 srcOffsetZ = 0;
    uint32 dstOffsetZ = 0;
    uint32 srcLayerCount = 1;
    uint32 dstLayerCount = 1;
    uint32 srcDepth = 1;
    uint32 dstDepth = 1;

	if (src->Is3DTexture())
	{
		srcOffsetZ = srcSlice;
		srcDepth = srcDepth_;
	}
	else
	{
		srcBaseLayer = srcSlice;
		srcLayerCount = srcDepth_;
	}

	if (dst->Is3DTexture())
	{
		dstOffsetZ = dstSlice;
		dstDepth = srcDepth_;
	}
	else
	{
		dstBaseLayer = dstSlice;
		dstLayerCount = srcDepth_;
	}

	// number of slices/layers this copy moves. srcLayerCount is only set for the 2D-array case, so a
	// 3D texture needs its depth here - both for the staging buffer size and for the copy loop
	const uint32 copySliceCount = src->Is3DTexture() ? srcDepth : srcLayerCount;

	if (srcMtlFormat != dstMtlFormat)
	{
		// Metal's copyFromTexture requires identical pixel formats. Aliased textures can use
		// different formats with compatible block sizes (e.g. RGBA32Uint <-> BC3, both 16 bytes
		// per block) - Vulkan's vkCmdCopyImage allows those, so route the raw data through a
		// temporary private buffer to keep the memory-level copy semantics
		if (srcInfo.bytesPerBlock != dstInfo.bytesPerBlock)
		{
			cemuLog_logOnce(LogType::Force, "texture_copyImageSubData: cannot copy between formats {:04x} -> {:04x} (different bytes per block)", (uint32)src->format, (uint32)dst->format);
			return;
		}
		if (srcLayerCount != dstLayerCount)
		{
			cemuLog_logOnce(LogType::Force, "texture_copyImageSubData: cannot copy between formats {:04x} -> {:04x} with mismatching layer counts", (uint32)src->format, (uint32)dst->format);
			return;
		}
		// mips smaller than a single block of a compressed side cannot be copied (Metal cannot
		// address them). Matches the Vulkan implementation, which drops these tiny mips as well
		const bool srcIsBlocked = srcInfo.blockTexelSize.x > 1 || srcInfo.blockTexelSize.y > 1;
		const bool dstIsBlocked = dstInfo.blockTexelSize.x > 1 || dstInfo.blockTexelSize.y > 1;
		if (srcIsBlocked && (src->GetMipWidth(srcMip) < (sint32)srcInfo.blockTexelSize.x || src->GetMipHeight(srcMip) < (sint32)srcInfo.blockTexelSize.y))
		{
			cemuLog_logDebug(LogType::Force, "texture_copyImageSubData: skipping copy with compressed source mip smaller than one block");
			return;
		}
		if (dstIsBlocked && (dst->GetMipWidth(dstMip) < (sint32)dstInfo.blockTexelSize.x || dst->GetMipHeight(dstMip) < (sint32)dstInfo.blockTexelSize.y))
		{
			cemuLog_logDebug(LogType::Force, "texture_copyImageSubData: skipping copy with compressed destination mip smaller than one block");
			return;
		}
		// raw region in blocks of the source layout. The block grid maps 1:1 to the destination
		// (equal bytes per block), which means the destination region covers
		// srcBlockCount * dstBlockTexelSize destination texels
		cemu_assert_debug(effectiveCopyWidth % srcInfo.blockTexelSize.x == 0 && effectiveCopyHeight % srcInfo.blockTexelSize.y == 0);
		cemu_assert_debug(effectiveSrcX % srcInfo.blockTexelSize.x == 0 && effectiveSrcY % srcInfo.blockTexelSize.y == 0);
		cemu_assert_debug(effectiveDstX % dstInfo.blockTexelSize.x == 0 && effectiveDstY % dstInfo.blockTexelSize.y == 0);
		// The block grid has to fit BOTH textures. The source read covers
		// srcBlockCount * srcBlockTexelSize source texels, but the destination write covers
		// srcBlockCount * dstBlockTexelSize texels - and when the two block sizes differ those are
		// not the same extent. Clamping the region to the source mip (above) therefore bounds the
		// read but not the write, which Metal aborts the blit for. Bound the grid by the source mip,
		// the destination mip and the staged bytes, then re-derive the extent from the bounded grid
		// so bytesPerRow stays consistent with the size the source read declares
		const uint32 maxSrcBlocksX = (uint32)std::max(srcMipW - effectiveSrcX, 0) / srcInfo.blockTexelSize.x;
		const uint32 maxSrcBlocksY = (uint32)std::max(srcMipH - effectiveSrcY, 0) / srcInfo.blockTexelSize.y;
		const uint32 maxDstBlocksX = (uint32)std::max(dstMipW - effectiveDstX, 0) / dstInfo.blockTexelSize.x;
		const uint32 maxDstBlocksY = (uint32)std::max(dstMipH - effectiveDstY, 0) / dstInfo.blockTexelSize.y;
		const uint32 srcBlockCountX = std::min({ effectiveCopyWidth / srcInfo.blockTexelSize.x, maxSrcBlocksX, maxDstBlocksX });
		const uint32 srcBlockCountY = std::min({ effectiveCopyHeight / srcInfo.blockTexelSize.y, maxSrcBlocksY, maxDstBlocksY });
		if (srcBlockCountX == 0 || srcBlockCountY == 0)
			return;
		effectiveCopyWidth = (sint32)(srcBlockCountX * srcInfo.blockTexelSize.x);
		effectiveCopyHeight = (sint32)(srcBlockCountY * srcInfo.blockTexelSize.y);
		const uint32 bytesPerRow = srcBlockCountX * srcInfo.bytesPerBlock;
		const uint32 bytesPerImage = bytesPerRow * srcBlockCountY;
		const uint64 copyBytes = (uint64)bytesPerImage * copySliceCount;

		// depth-stencil textures require an explicit aspect option for buffer copies; the copy
		// carries the depth aspect (stencil layouts are not portable between depth formats anyway)
		MTL::BlitOption srcBlitOption = MTL::BlitOptionNone;
		MTL::BlitOption dstBlitOption = MTL::BlitOptionNone;
		if (src->isDepth && GetMtlPixelFormatInfo(src->format, true).hasStencil)
			srcBlitOption = MTL::BlitOptionDepthFromDepthStencil;
		if (dst->isDepth && GetMtlPixelFormatInfo(dst->format, true).hasStencil)
			dstBlitOption = MTL::BlitOptionDepthFromDepthStencil;

		if (!m_textureCopyStagingBuffer || m_textureCopyStagingBuffer->length() < copyBytes)
		{
			if (m_textureCopyStagingBuffer)
				m_textureCopyStagingBuffer->release();
			m_textureCopyStagingBuffer = m_device->newBuffer(copyBytes, MTL::ResourceStorageModePrivate);
		}
		if (!m_textureCopyStagingBuffer)
		{
			cemuLog_logOnce(LogType::Force, "texture_copyImageSubData: failed to allocate staging buffer");
			return;
		}

		// 3D textures address their slices through origin.z (Metal ignores the slice argument for
		// TextureType3D) while 2D arrays address them through the slice index, so a 3D copy has to
		// carry the z coordinate - hardcoding it to 0 copied slice 0 regardless of
		// srcOffsetZ/dstOffsetZ. The two dimensions cannot mix here: a 3D source against a 2D
		// destination (or the reverse) leaves the layer counts unequal and returns above
		for (uint32 i = 0; i < copySliceCount; i++)
		{
			const uint32 srcSliceI = src->Is3DTexture() ? 0 : srcBaseLayer + i;
			const uint32 dstSliceI = dst->Is3DTexture() ? 0 : dstBaseLayer + i;
			const uint32 srcZ = src->Is3DTexture() ? srcOffsetZ + i : 0;
			const uint32 dstZ = dst->Is3DTexture() ? dstOffsetZ + i : 0;
			blitCommandEncoder->copyFromTexture(mtlSrc, srcSliceI, srcMip, MTL::Origin(effectiveSrcX, effectiveSrcY, srcZ), MTL::Size(effectiveCopyWidth, effectiveCopyHeight, 1), m_textureCopyStagingBuffer, (uint64)i * bytesPerImage, bytesPerRow, bytesPerImage, srcBlitOption);
			blitCommandEncoder->copyFromBuffer(m_textureCopyStagingBuffer, (uint64)i * bytesPerImage, bytesPerRow, bytesPerImage, MTL::Size(srcBlockCountX * dstInfo.blockTexelSize.x, srcBlockCountY * dstInfo.blockTexelSize.y, 1), mtlDst, dstSliceI, dstMip, MTL::Origin(effectiveDstX, effectiveDstY, dstZ), dstBlitOption);
		}
		// track the copied mip as written (same bookkeeping as render passes) so the sampled-view
		// mip clamping in BindStageResources doesn't collapse views whose upper mips were populated
		// by copies instead of draws. Also bump the write event counter - it has to cover every
		// content-mutating path or EnsureSampledMipContentValid's freshness check is unsound
		static_cast<LatteTextureMtl*>(dst)->MarkRenderMipsWritten(dstMip, 1);
		// the region here is expressed in the source's block grid, so "covers the whole level"
		// above only carries over when both formats share a block size; otherwise the copy is
		// conservatively recorded as partial (such a chain keeps being clamped/regenerated)
		const bool blockGridComparable = srcInfo.blockTexelSize.x == dstInfo.blockTexelSize.x && srcInfo.blockTexelSize.y == dstInfo.blockTexelSize.y;
		auto* dstMtl = static_cast<LatteTextureMtl*>(dst);
		dstMtl->MarkCopyMipsWritten(dstMip, 1, blockGridComparable ? copyProvenance : LatteTextureMtl::CopyProvenance::Partial);
		dstMtl->SetLastCopyInfo({ (uint32)srcMip, (uint32)dstMip, (uint32)srcLevelWidth, (uint32)srcLevelHeight,
			(uint32)dstLevelWidth, (uint32)dstLevelHeight, (uint32)effectiveCopyWidth, (uint32)effectiveCopyHeight, src == dst });
		LatteTexture_TrackTextureGPUWrite(dst, dstSlice, dstMip, LatteTexture_getNextUpdateEventCounter());
		return;
	}

	// If copying whole textures, we can do a more efficient copy
    if (effectiveSrcX == 0 && effectiveSrcY == 0 && effectiveDstX == 0 && effectiveDstY == 0 &&
        srcOffsetZ == 0 && dstOffsetZ == 0 &&
        effectiveCopyWidth == src->GetMipWidth(srcMip) && effectiveCopyHeight == src->GetMipHeight(srcMip) && srcDepth == src->GetMipDepth(srcMip) &&
        effectiveCopyWidth == dst->GetMipWidth(dstMip) && effectiveCopyHeight == dst->GetMipHeight(dstMip) && dstDepth == dst->GetMipDepth(dstMip) &&
        srcLayerCount == dstLayerCount)
    {
        // copySliceCount rather than srcLayerCount: for a 3D texture the slice count is its depth,
        // and srcLayerCount is never set for one - passing it copied a single slice of the volume
        blitCommandEncoder->copyFromTexture(mtlSrc, srcBaseLayer, srcMip, mtlDst, dstBaseLayer, dstMip, copySliceCount, 1);
    }
    else
    {
        if (srcLayerCount == dstLayerCount)
        {
            for (uint32 i = 0; i < srcLayerCount; i++)
            {
                blitCommandEncoder->copyFromTexture(mtlSrc, srcBaseLayer + i, srcMip, MTL::Origin(effectiveSrcX, effectiveSrcY, srcOffsetZ), MTL::Size(effectiveCopyWidth, effectiveCopyHeight, srcDepth), mtlDst, dstBaseLayer + i, dstMip, MTL::Origin(effectiveDstX, effectiveDstY, dstOffsetZ));
            }
        }
        else
        {
            for (uint32 i = 0; i < std::max(srcLayerCount, dstLayerCount); i++)
            {
                if (srcLayerCount == 1)
                    srcOffsetZ++;
                else
                    srcSlice++;

                if (dstLayerCount == 1)
                    dstOffsetZ++;
                else
                    dstSlice++;

                blitCommandEncoder->copyFromTexture(mtlSrc, srcBaseLayer, srcMip, MTL::Origin(effectiveSrcX, effectiveSrcY, srcOffsetZ), MTL::Size(effectiveCopyWidth, effectiveCopyHeight, 1), mtlDst, dstBaseLayer, dstMip, MTL::Origin(effectiveDstX, effectiveDstY, dstOffsetZ));
            }
        }
    }
    // track the copied mip as written (same bookkeeping as render passes) so the sampled-view
    // mip clamping in BindStageResources doesn't collapse views whose upper mips were populated
    // by copies instead of draws. Also bumps the write event counter - same soundness
    // requirement as the staging path above
    static_cast<LatteTextureMtl*>(dst)->MarkRenderMipsWritten(dstMip, 1);
    // copy provenance - see the classification at the top of this function. This is the record the
    // sample-time workarounds consult to decide whether the level's content is the game's own
    auto* dstMtl = static_cast<LatteTextureMtl*>(dst);
    dstMtl->MarkCopyMipsWritten(dstMip, 1, copyProvenance);
    dstMtl->SetLastCopyInfo({ (uint32)srcMip, (uint32)dstMip, (uint32)srcLevelWidth, (uint32)srcLevelHeight,
        (uint32)dstLevelWidth, (uint32)dstLevelHeight, (uint32)effectiveCopyWidth, (uint32)effectiveCopyHeight, src == dst });
    LatteTexture_TrackTextureGPUWrite(dst, dstSlice, dstMip, LatteTexture_getNextUpdateEventCounter());
}

LatteTextureReadbackInfo* MetalRenderer::texture_createReadback(LatteTextureView* textureView)
{
    auto* baseTexture = static_cast<LatteTextureMtl*>(textureView->baseTexture);
    if (baseTexture->m_isAlternateFormat)
    {
        cemuLog_logDebug(LogType::Force, "Metal does not support readback of texture format 0x{:x}", (uint32)baseTexture->format);
        return nullptr;
    }
    size_t uploadSize = baseTexture->GetTexture()->allocatedSize();

    if ((m_readbackBufferWriteOffset + uploadSize) > TEXTURE_READBACK_SIZE)
	{
		m_readbackBufferWriteOffset = 0;
	}

    auto* result = new LatteTextureReadbackInfoMtl(this, textureView, m_readbackBufferWriteOffset);
    m_readbackBufferWriteOffset += uploadSize;

	return result;
}

void MetalRenderer::surfaceCopy_copySurfaceWithFormatConversion(LatteTexture* sourceTexture, sint32 srcMip, sint32 srcSlice, LatteTexture* destinationTexture, sint32 dstMip, sint32 dstSlice, sint32 width, sint32 height)
{
    // scale copy size to effective size
	sint32 effectiveCopyWidth = width;
	sint32 effectiveCopyHeight = height;
	LatteTexture_scaleToEffectiveSize(sourceTexture, &effectiveCopyWidth, &effectiveCopyHeight, 0);

	// check if texture rescale ratios match
	if (!LatteTexture_doesEffectiveRescaleRatioMatch(sourceTexture, srcMip, destinationTexture, dstMip))
	{
		MetalDiag_Count(MetalDiagEvent::SurfaceCopySkipped, "{:016x} -> {:016x}: mismatching effective dimensions (src 0x{:04x}, dst 0x{:04x})",
			sourceTexture->physAddress, destinationTexture->physAddress, (uint32)sourceTexture->format, (uint32)destinationTexture->format);
		return;
	}

	// check if bpp size matches
	if (sourceTexture->GetBPP() != destinationTexture->GetBPP())
	{
		MetalDiag_Count(MetalDiagEvent::SurfaceCopySkipped, "{:016x} -> {:016x}: mismatching BPP (src 0x{:04x} bpp {}, dst 0x{:04x} bpp {})",
			sourceTexture->physAddress, destinationTexture->physAddress, (uint32)sourceTexture->format, sourceTexture->GetBPP(), (uint32)destinationTexture->format, destinationTexture->GetBPP());
		return;
	}

	// copies between two depth or two color textures can use a raw blit
	if (sourceTexture->isDepth == destinationTexture->isDepth)
	{
		// no TrackTextureGPUWrite here: texture_copyImageSubData already does it on both of its
		// success paths, and calling it again with a second counter made the mirror-refresh and
		// mip-regeneration freshness checks see a write that no GPU work produced
		texture_copyImageSubData(sourceTexture, srcMip, 0, 0, srcSlice, destinationTexture, dstMip, 0, 0, dstSlice, effectiveCopyWidth, effectiveCopyHeight, 1);
		return;
	}

	// color <-> depth copies require a draw-based copy
	surfaceCopy_viaDrawcall(sourceTexture, srcMip, srcSlice, destinationTexture, dstMip, dstSlice, effectiveCopyWidth, effectiveCopyHeight);
}

MTL::RenderPipelineState* MetalRenderer::surfaceCopy_getOrCreatePipeline(LatteTexture* destinationTexture)
{
	const bool dstIsDepth = destinationTexture->isDepth;
	const MTL::PixelFormat dstPixelFormat = GetMtlPixelFormat(destinationTexture->format, dstIsDepth);

	const auto key = std::make_pair(dstPixelFormat, dstIsDepth);
	auto itr = m_copySurfacePipelines.find(key);
	if (itr != m_copySurfacePipelines.end())
		return itr->second;

	auto desc = dstIsDepth ? m_copyColorToDepthDesc : m_copyDepthToColorDesc;
	if (dstIsDepth)
	{
		desc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatInvalid);
		desc->setDepthAttachmentPixelFormat(dstPixelFormat);
		if (GetMtlPixelFormatInfo(destinationTexture->format, true).hasStencil)
			desc->setStencilAttachmentPixelFormat(dstPixelFormat);
		else
			desc->setStencilAttachmentPixelFormat(MTL::PixelFormatInvalid);
	}
	else
	{
		desc->colorAttachments()->object(0)->setPixelFormat(dstPixelFormat);
		desc->setDepthAttachmentPixelFormat(MTL::PixelFormatInvalid);
		desc->setStencilAttachmentPixelFormat(MTL::PixelFormatInvalid);
	}

	NS::Error* error = nullptr;
	auto pipeline = m_device->newRenderPipelineState(desc, &error);
	if (error || !pipeline)
	{
		cemuLog_log(LogType::Force, "failed to create surface copy pipeline (error: {})", error ? error->localizedDescription()->utf8String() : "unknown error");
		// the driver may still return a state object alongside the error - don't use it
		if (pipeline)
			pipeline->release();
		// note: the out-param error is autoreleased (metal-cpp does not retain it) - do not
		// release it here, the autorelease pool owns its lifetime
		return nullptr;
	}

	m_copySurfacePipelines.emplace(key, pipeline);

	return pipeline;
}

void MetalRenderer::surfaceCopy_viaDrawcall(LatteTexture* sourceTexture, sint32 srcMip, sint32 srcSlice, LatteTexture* destinationTexture, sint32 dstMip, sint32 dstSlice, sint32 effectiveCopyWidth, sint32 effectiveCopyHeight)
{
	auto mtlSrc = static_cast<LatteTextureMtl*>(sourceTexture)->GetTexture();
	auto mtlDst = static_cast<LatteTextureMtl*>(destinationTexture)->GetTexture();
	const bool dstIsDepth = destinationTexture->isDepth;

	auto pipeline = surfaceCopy_getOrCreatePipeline(destinationTexture);
	if (!pipeline)
		return;

	// neither texture may be attached to an active render pass
	EndEncoding();

	// render pass targeting only the destination
	NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
	// load actions preserve the existing contents: the copy only writes the copy region and the
	// textures are aliased surface copies whose remaining contents must stay intact
	if (dstIsDepth)
	{
		auto depthAttachment = renderPassDescriptor->depthAttachment();
		depthAttachment->setTexture(mtlDst);
		depthAttachment->setLevel(dstMip);
		depthAttachment->setSlice(dstSlice);
		depthAttachment->setLoadAction(MTL::LoadActionLoad);
		depthAttachment->setStoreAction(MTL::StoreActionStore);
		// the copy pipeline declares a stencil attachment for stencil depth formats (required by
		// Metal), so the render pass must declare one too even though the copy never writes stencil
		if (GetMtlPixelFormatInfo(destinationTexture->format, true).hasStencil)
		{
			auto stencilAttachment = renderPassDescriptor->stencilAttachment();
			stencilAttachment->setTexture(mtlDst);
			stencilAttachment->setLevel(dstMip);
			stencilAttachment->setSlice(dstSlice);
			stencilAttachment->setLoadAction(MTL::LoadActionLoad);
			stencilAttachment->setStoreAction(MTL::StoreActionStore);
		}
	}
	else
	{
		auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
		colorAttachment->setTexture(mtlDst);
		colorAttachment->setLevel(dstMip);
		colorAttachment->setSlice(dstSlice);
		colorAttachment->setLoadAction(MTL::LoadActionLoad);
		colorAttachment->setStoreAction(MTL::StoreActionStore);
	}

	auto renderCommandEncoder = GetTemporaryRenderCommandEncoder(renderPassDescriptor);

	// restrict drawing to the copy region
	MTL::Viewport viewport = { 0.0, 0.0, (double)effectiveCopyWidth, (double)effectiveCopyHeight, 0.0, 1.0 };
	renderCommandEncoder->setViewport(viewport);

	renderCommandEncoder->setRenderPipelineState(pipeline);
	m_state.m_encoderState.m_renderPipelineState = pipeline;

	if (dstIsDepth)
	{
		// writing depth requires compare always + write enabled
		if (!m_copyDepthState)
		{
			NS_STACK_SCOPED MTL::DepthStencilDescriptor* depthStencilDescriptor = MTL::DepthStencilDescriptor::alloc()->init();
			depthStencilDescriptor->setDepthCompareFunction(MTL::CompareFunctionAlways);
			depthStencilDescriptor->setDepthWriteEnabled(true);
			m_copyDepthState = m_device->newDepthStencilState(depthStencilDescriptor);
		}
		renderCommandEncoder->setDepthStencilState(m_copyDepthState);
		m_state.m_encoderState.m_depthStencilState = m_copyDepthState;
	}

	// bind the source as a single-mip, single-slice view so texel reads map 1:1 to the copy region
	auto srcView = mtlSrc->newTextureView(GetMtlPixelFormat(sourceTexture->format, sourceTexture->isDepth), MTL::TextureType2D, NS::Range::Make(srcMip, 1), NS::Range::Make(srcSlice, 1));
	SetTexture(renderCommandEncoder, METAL_SHADER_TYPE_FRAGMENT, srcView, GET_HELPER_TEXTURE_BINDING(0));
	// the view is temporary, don't keep it in the encoder state
	m_state.m_encoderState.m_textures[METAL_SHADER_TYPE_FRAGMENT][GET_HELPER_TEXTURE_BINDING(0)] = nullptr;
	srcView->release();

	renderCommandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));

	EndEncoding();

	LatteTexture_TrackTextureGPUWrite(destinationTexture, dstSlice, dstMip, LatteTexture_getNextUpdateEventCounter());
	// track the copied mip as written (same bookkeeping as render passes) - see
	// texture_copyImageSubData
	static_cast<LatteTextureMtl*>(destinationTexture)->MarkRenderMipsWritten(dstMip, 1);
}

MTL::RenderPipelineState* MetalRenderer::depthCopy_getOrCreatePipeline(MTL::PixelFormat mirrorFormat)
{
	if (m_depthColorCopyPipeline && m_depthColorCopyPipelineFormat == mirrorFormat)
		return m_depthColorCopyPipeline;
	if (m_depthColorCopyPipeline)
	{
		m_depthColorCopyPipeline->release();
		m_depthColorCopyPipeline = nullptr;
	}

	// m_copyDepthToColorDesc carries the fullscreen vertex function and the depth-reading
	// fragmentCopyDepthToColor function (set up during initialization)
	m_copyDepthToColorDesc->colorAttachments()->object(0)->setPixelFormat(mirrorFormat);
	m_copyDepthToColorDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatInvalid);
	m_copyDepthToColorDesc->setStencilAttachmentPixelFormat(MTL::PixelFormatInvalid);

	NS::Error* error = nullptr;
	m_depthColorCopyPipeline = m_device->newRenderPipelineState(m_copyDepthToColorDesc, &error);
	if (error || !m_depthColorCopyPipeline)
	{
		cemuLog_log(LogType::Force, "failed to create depth color copy pipeline (error: {})", error ? error->localizedDescription()->utf8String() : "unknown error");
		// the driver may still return a state object alongside the error - don't use it
		if (m_depthColorCopyPipeline)
			m_depthColorCopyPipeline->release();
		m_depthColorCopyPipeline = nullptr;
		// the out-param error is autoreleased (metal-cpp does not retain it) - do not release it
		return nullptr;
	}
	m_depthColorCopyPipelineFormat = mirrorFormat;
	return m_depthColorCopyPipeline;
}

// Validates the depth mirror for the given view, allocates/reallocates it as needed and applies the
// freshness + rate-limit checks. Returns true when a refresh copy must be encoded (with the mirror
// texture and its format in the out params), false when the mirror is up to date or unsupported
bool MetalRenderer::depthCopy_ensureColorCopy(LatteTextureView* textureView, MTL::Texture** colorCopyOut, MTL::PixelFormat* mirrorFormatOut)
{
	auto texMtl = static_cast<LatteTextureMtl*>(textureView->baseTexture);
	auto depthTexture = texMtl->GetTexture();

	// plain 2D and 2D-array depth textures are supported (shadow cascades commonly use arrays; the
	// copy shader reads texels at 1:1 via access::read on a per-layer 2D view at mip 0). Other
	// types (MSAA, 3D) cannot be mirrored - the binding path must never fall back to binding the
	// raw depth view in that case, so it checks GetDepthColorCopy() and binds a null texture
	const MTL::TextureType srcTextureType = depthTexture->textureType();
	if (srcTextureType != MTL::TextureType2D && srcTextureType != MTL::TextureType2DArray)
	{
		MetalDiag_Count(MetalDiagEvent::DepthMirrorUnavailable,
			"depth-as-data sampling of a non-2D/2DArray depth texture is not supported (texture type {}, dim {}, {:016x})",
			(uint32)srcTextureType, (uint32)textureView->dim, textureView->baseTexture->physAddress);
		return false;
	}
	const bool isArray = (srcTextureType == MTL::TextureType2DArray);
	const NS::UInteger arrayLength = isArray ? depthTexture->arrayLength() : 1;

	// R32Float is only filterable on Apple GPUs (needed for generateMipmaps and for sampling the
	// mirror with the game's linear filters); other vendors use R16Float. Depth values are
	// normalized to [0,1], which R16Float represents with adequate precision
	const MTL::PixelFormat mirrorFormat = m_device->supportsFamily(MTL::GPUFamilyApple7) ? MTL::PixelFormatR32Float : MTL::PixelFormatR16Float;

	auto colorCopy = texMtl->GetDepthColorCopy();
	if (colorCopy && (colorCopy->width() != depthTexture->width() || colorCopy->height() != depthTexture->height() ||
		colorCopy->pixelFormat() != mirrorFormat || colorCopy->mipmapLevelCount() != depthTexture->mipmapLevelCount() ||
		colorCopy->textureType() != srcTextureType || colorCopy->arrayLength() != arrayLength))
	{
		texMtl->SetDepthColorCopy(nullptr, 0); // releases the outdated copy
		colorCopy = nullptr;
	}
	if (!colorCopy)
	{
		NS_STACK_SCOPED MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
		desc->setTextureType(srcTextureType);
		desc->setPixelFormat(mirrorFormat);
		desc->setWidth(depthTexture->width());
		desc->setHeight(depthTexture->height());
		desc->setMipmapLevelCount(depthTexture->mipmapLevelCount()); // keep the full mip chain so LOD selection behaves like on Vulkan
		desc->setArrayLength(arrayLength); // mirror the whole array so per-layer sample views work
		// PixelFormatView: swizzled sample views are created from the mirror (same pixel format
		// today, but the swizzle-taking newTextureView variants require the usage flag)
		desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget | MTL::TextureUsagePixelFormatView);
		colorCopy = m_device->newTexture(desc);
		if (!colorCopy)
		{
			MetalDiag_Count(MetalDiagEvent::DepthMirrorUnavailable,
				"failed to allocate the depth color copy ({}x{}, {} layers)", depthTexture->width(), depthTexture->height(), arrayLength);
			return false;
		}
		// labeled so the texture is identifiable in GPU captures
		colorCopy->setLabel(ToNSString("Cemu DepthMirror"));
		texMtl->SetDepthColorCopy(colorCopy, 0);
	}

	// refresh when the depth data changed since the copy was taken. lastWriteEventCounter is
	// bumped after every draw that wrote the texture as a render target
	// (LatteRenderTarget_trackUpdates) and by GPU surface copies, so the mirror stays coherent
	// even when the game rewrites a depth map mid-frame (e.g. shadow clouds)
	if (texMtl->GetDepthColorCopyUpdateCounter() == texMtl->lastWriteEventCounter)
		return false;

	// rate limit: each refresh commits a render pass (plus mipmap regeneration), so cap the
	// number of refreshes per frame. Games alternating between rewriting a depth map and sampling it
	// would otherwise pay that cost for every draw. Once the cap is hit the mirror keeps its last
	// refreshed contents until the next frame (the update counter check above will trigger a refresh).
	// Mirrors without valid contents yet (never refreshed, or re-allocated mid-frame) are exempt -
	// sampling uninitialized mirror data would be worse than the extra refresh
	constexpr uint32 kMaxRefreshesPerFrame = 4;
	const bool mirrorHasContents = colorCopy && texMtl->GetDepthColorCopyUpdateCounter() != 0;
	if (mirrorHasContents && texMtl->IsDepthColorCopyRefreshCapped(LatteGPUState.frameCounter, kMaxRefreshesPerFrame))
	{
		cemuLog_logOnce(LogType::Force, "depth color copy refresh capped for this frame (depth texture written and sampled more than {} times in one frame)", kMaxRefreshesPerFrame);
		return false;
	}
	if (mirrorHasContents)
		texMtl->TrackDepthColorCopyRefresh(LatteGPUState.frameCounter);

	*colorCopyOut = colorCopy;
	*mirrorFormatOut = mirrorFormat;
	return true;
}

// Encodes the depth->mirror copy passes: one render pass per array layer plus mipmap regeneration.
// Must be encoded strictly after the writes that produced the depth data and strictly before the
// draws that sample the mirror (see the two callers for how that ordering is achieved)
void MetalRenderer::depthCopy_encodeCopies(MTL::CommandBuffer* commandBuffer, LatteTextureMtl* texMtl, MTL::Texture* colorCopy, MTL::PixelFormat mirrorFormat)
{
	auto depthTexture = texMtl->GetTexture();
	const NS::UInteger arrayLength = (depthTexture->textureType() == MTL::TextureType2DArray) ? depthTexture->arrayLength() : 1;

	auto pipeline = depthCopy_getOrCreatePipeline(mirrorFormat);
	if (!pipeline)
		return;

	for (NS::UInteger layer = 0; layer < arrayLength; layer++)
	{
		NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
		auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
		colorAttachment->setTexture(colorCopy);
		colorAttachment->setLevel(0);
		colorAttachment->setSlice(layer);
		colorAttachment->setLoadAction(MTL::LoadActionDontCare);
		colorAttachment->setStoreAction(MTL::StoreActionStore);

		auto renderCommandEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor);
		renderCommandEncoder->setRenderPipelineState(pipeline);
		MTL::Viewport viewport = { 0.0, 0.0, (double)depthTexture->width(), (double)depthTexture->height(), 0.0, 1.0 };
		renderCommandEncoder->setViewport(viewport);

		// bind the depth texture as source; fragmentCopyDepthToColor reads texels at 1:1 via access::read
		auto srcView = depthTexture->newTextureView(GetMtlPixelFormat(texMtl->format, true), MTL::TextureType2D, NS::Range::Make(0, 1), NS::Range::Make(layer, 1));
		renderCommandEncoder->setFragmentTexture(srcView, GET_HELPER_TEXTURE_BINDING(0));
		srcView->release();

		renderCommandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
		renderCommandEncoder->endEncoding();
	}

	// regenerate the remaining mip levels from the freshly copied mip 0
	if (colorCopy->mipmapLevelCount() > 1)
	{
		auto blitEncoder = commandBuffer->blitCommandEncoder();
		blitEncoder->generateMipmaps(colorCopy);
		blitEncoder->endEncoding();
	}
}

// The single refresh path for the depth mirror. The copy must be encoded into the main command
// buffer between the previous pass and the draw's fresh pass (PrepareFeedbackLoopShadowCopies calls
// this before the draw's pass is acquired), because the depth writes it captures may have been
// recorded moments earlier into the same, still-uncommitted command buffer. A separate side
// command buffer would commit ahead of the main one (same-queue command buffers execute in commit
// order, and hazard tracking only serializes in that order), so it could read depth as of the last
// commit boundary instead of the current contents
void MetalRenderer::depthCopy_refreshColorCopyBeforeDraw(LatteTextureView* textureView)
{
	MTL::Texture* colorCopy = nullptr;
	MTL::PixelFormat mirrorFormat;
	if (!depthCopy_ensureColorCopy(textureView, &colorCopy, &mirrorFormat))
		return;
	auto texMtl = static_cast<LatteTextureMtl*>(textureView->baseTexture);

	EndEncoding();
	depthCopy_encodeCopies(GetCommandBuffer(), texMtl, colorCopy, mirrorFormat);
	GetRenderCommandEncoder();

	texMtl->SetDepthColorCopy(colorCopy, texMtl->lastWriteEventCounter);
}

// Builds the sample view for a feedback-loop shadow copy, matching the type/mip/slice selection of
// the bound view (mirrors CreateViewInternal, but on the shadow copy)
MTL::Texture* MetalRenderer::CreateFeedbackShadowView(MTL::Texture* shadow, LatteTextureView* textureView, const MTL::TextureSwizzleChannels& swizzle)
{
	MTL::TextureType textureType;
	switch (textureView->dim)
	{
	case Latte::E_DIM::DIM_1D:
		textureType = MTL::TextureType1D;
		break;
	case Latte::E_DIM::DIM_2D:
	case Latte::E_DIM::DIM_2D_MSAA:
		textureType = MTL::TextureType2D;
		break;
	case Latte::E_DIM::DIM_2D_ARRAY:
		textureType = MTL::TextureType2DArray;
		break;
	case Latte::E_DIM::DIM_3D:
		textureType = MTL::TextureType3D;
		break;
	case Latte::E_DIM::DIM_CUBEMAP:
		textureType = MTL::TextureTypeCubeArray;
		break;
	default:
		textureType = MTL::TextureType2D;
		break;
	}

	// the shadow mirrors the full base texture, so clamp the view's mip/slice selection to its extent
	uint32 baseLevel = textureView->firstMip;
	const uint32 shadowMipCount = (uint32)shadow->mipmapLevelCount();
	baseLevel = std::min(baseLevel, shadowMipCount - 1);
	uint32 levelCount = std::min<uint32>(std::max<uint32>(textureView->numMip, 1), shadowMipCount - baseLevel);
	uint32 baseLayer = textureView->firstSlice;
	uint32 layerCount = (textureType == MTL::TextureType2DArray || textureType == MTL::TextureTypeCubeArray) ? textureView->numSlice : 1;
	// the sampled view must use the same pixel format as the game's view of the original texture
	// (view->format can differ from the base texture's format for compatible re-interpretations,
	// e.g. sRGB swaps - using the shadow's own format would misinterpret the data)
	const MTL::PixelFormat viewPixelFormat = GetMtlPixelFormat(textureView->format, static_cast<LatteTextureMtl*>(textureView->baseTexture)->isDepth);
	return shadow->newTextureView(viewPixelFormat, textureType, NS::Range::Make(baseLevel, levelCount), NS::Range::Make(baseLayer, layerCount), swizzle);
}

// Cached sample view of a feedback-loop shadow copy (BindStageResources): one driver allocation
// per distinct view spec instead of per draw. The cache lives on the shadow entry and is
// invalidated when the shadow texture is replaced
MTL::Texture* MetalRenderer::GetFeedbackShadowSampleView(FeedbackShadowCopy& shadowCopy, LatteTextureView* textureView, uint32 gpuSamplerSwizzle)
{
	const LatteMtlSampleViewKey key = {
		.swizzle = gpuSamplerSwizzle & 0x0FFF0000,
		.format = (uint16)textureView->format,
		.dim = (uint8)textureView->dim,
		.firstMip = (uint8)textureView->firstMip,
		.numMip = (uint8)textureView->numMip,
		.firstSlice = (uint16)textureView->firstSlice,
		.numSlice = (uint16)textureView->numSlice,
	};
	auto itr = shadowCopy.sampleViews.find(key);
	if (itr != shadowCopy.sampleViews.end())
		return itr->second;
	auto swizzle = LatteTextureViewMtl::GetSwizzleChannels(textureView->format, gpuSamplerSwizzle);
	MTL::Texture* view = CreateFeedbackShadowView(shadowCopy.texture, textureView, swizzle);
	shadowCopy.sampleViews.emplace(key, view);
	return view;
}


// Attachment feedback loop workaround. Draws that sample a texture which is also an attachment of
// the active FBO (read-modify-write post-processing like DoF/bloom compositing) cannot read it from
// the attachment on Metal - Vulkan serves these reads via VK_EXT_attachment_feedback_loop_layout.
// Called before the render pass for a draw is acquired: if a feedback loop is detected, the current
// pass is ended, the attachment contents are copied into shadow textures (strictly ordered in the
// main command buffer, so the copies capture everything before this draw) and a fresh pass is
// opened; BindStageResources then binds the shadows instead of the attachments. The load/store
// actions of the passes preserve the attachment contents, so the game's read-modify-write semantics
// are unaffected. Shadow copies are reused across draws and re-copied per feedback draw (each draw
// reads the previous draw's output)
//
// This scan is also the single place where depth-as-data mirrors are refreshed: every bound depth
// texture that is sampled as plain data gets its color copy refreshed via a pass break into the
// main command buffer (depthCopy_refreshColorCopyBeforeDraw), so the copy is strictly ordered after
// the depth writes - including writes recorded earlier in the same, still-uncommitted command
// buffer. A side command buffer would commit ahead of the main one (same-queue execution is
// commit-ordered) and could read stale depth. BindStageResources then only binds the mirror.

// Regenerates never-written upper mip levels of sampled effect buffers from their fresh
// render/copy-written mip0. Some games (SM3DW's DoF chain) sample effect buffers through their
// full mip chain with forced LODs while only mip 0 ever receives game content (render passes
// write mip 0; the upper levels of these pack-resized chains are never managed by the game).
// Serving the stale upper levels diverges from Vulkan. Detect such chains here (mip 0 GPU-written,
// the sampled levels not trustworthy - see LatteTextureMtl::RangeContentIsTrustworthy - filterable
// FLOAT format, sampled view reaches mip >= 1) and regenerate the upper levels via the blit
// encoder's generateMipmaps (box filter) in the pass-break window, so the copies are strictly
// ordered before the draw. Regeneration runs once per mip0 content update, tracked via the
// texture's write event counter, which the copy paths feed. Runs BEFORE
// PrepareFeedbackLoopShadowCopies so feedback shadow copies pick up the regenerated levels
void MetalRenderer::EnsureSampledMipContentValid(LatteDecompilerShader* vertexShader, LatteDecompilerShader* geometryShader, LatteDecompilerShader* pixelShader)
{
	// the texture plus the sampling unit that asked for it - the unit is what makes the decision
	// interesting (a unit that pins one LOD above 0 reads a specific level, one with a range reads
	// the pyramid), and it is reported in the diagnostic below
	struct ChainToGenerate
	{
		LatteTextureMtl* texMtl;
		LatteDecompilerShader* samplingShader;
		sint32 textureUnit; // relative to the stage, as the diagnostics for the clamp side report it
	};
	std::vector<ChainToGenerate> toGenerate;
	// scan only the texture units the current draw's shaders actually reference. Slots of units
	// that stopped being sampled are never refreshed by the generic bind flow (which writes
	// texture_setLatteTexture for used units only), so they can hold views deleted by the texture
	// cache long ago - dereferencing them is a use-after-free. The same restriction applies to
	// units handled by framebuffer fetch (they are not bound at all)
	const std::array<std::pair<LatteDecompilerShader*, sint32>, 3> stageBases{ {
		{ vertexShader, LATTE_CEMU_VS_TEX_UNIT_BASE },
		{ geometryShader, LATTE_CEMU_GS_TEX_UNIT_BASE },
		{ pixelShader, LATTE_CEMU_PS_TEX_UNIT_BASE },
	} };
	for (const auto& [shader, stageBase] : stageBases)
	{
		if (!shader)
			continue;
		sint32 textureCount = shader->resourceMapping.getTextureCount();
		for (int i = 0; i < textureCount; ++i)
		{
			const auto relative_textureUnit = shader->resourceMapping.getRelativeTextureUnitFromRelativeBindingPoint(i);
			// don't scan textures that are accessed with a framebuffer fetch (not bound at all)
			if (m_supportsFramebufferFetch && shader->textureRenderTargetIndex[relative_textureUnit] != 255)
				continue;
			auto textureView = m_state.m_textures[relative_textureUnit + stageBase];
			if (!textureView)
				continue;
			LatteTexture* baseTexture = textureView->baseTexture;
			if (baseTexture->isDepth || baseTexture->Is3DTexture())
				continue;
		// the sampled view must be able to reach an upper mip
		if (textureView->firstMip + textureView->numMip <= 1)
			continue;
		auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
		if (baseTexture->mipLevels <= 1)
			continue;
		// Only regenerate when a level this draw samples is NOT trustworthy - the same question the
		// sample-time clamp answers, from the other side (LatteTextureMtl::RangeContentIsTrustworthy).
		// Untrustworthy means a copy left a re-crop of another incarnation's content in that level,
		// or nobody wrote it (the chains this exists for: SM3DW samples a chain it never fills).
		// A trustworthy range is served as the game asked - regenerating it replaces correct content
		// with a box filter of mip 0, the dithering this caused on MK8. The verdict is per sampled
		// view, not per chain, so a chain with one questionable level is not rebuilt for a draw that
		// never reads it. mip 0 must be GPU-written: it is the generation source, and requiring it
		// excludes CPU-uploaded asset textures, whose valid chains must never be replaced
		if ((texMtl->GetRenderWrittenMipMask() & 1u) == 0)
			continue;
		if (texMtl->RangeContentIsTrustworthy(textureView->firstMip, textureView->numMip))
			continue;
		// generateMipmaps needs a filterable format AND color renderability - Metal implements
		// it as render passes, and BC/compressed formats are filterable but NOT renderable
		// (validation aborts the blit). The clamp class in BindStageResources still covers
		// compressed chains safely (single-level views of compressed textures are legal)
		if (GetMtlPixelFormatInfo(baseTexture->format, false).dataType != MetalDataType::FLOAT)
			continue;
		if (!FormatIsRenderable(baseTexture->format))
			continue;
		// Regenerating once per mip 0 content update is required: skipping it loses MK8's bloom and
		// DoF, so the generated upper levels are read even when the written masks say otherwise
		if (texMtl->GetMipGenerationEventCounter() == baseTexture->lastWriteEventCounter)
			continue;
		const bool alreadyQueued = std::any_of(toGenerate.begin(), toGenerate.end(),
			[&](const ChainToGenerate& entry) { return entry.texMtl == texMtl; });
		if (alreadyQueued)
			continue;
		toGenerate.push_back({ texMtl, shader, relative_textureUnit });
		}
	}
	if (toGenerate.empty())
		return;

	{
		// one event per texture, so the session summary says how many distinct sampled textures had
		// their upper mips backend-generated - Vulkan never does this. The sampling unit is included:
		// the same chain can be read by several units with different LOD regimes, and which one asked
		// for the regeneration is part of deciding whether regenerating was the right answer (the
		// sampler's own LOD range is in the clamp line for the same texture, which fires on the same draws)
		for (const auto& entry : toGenerate)
		{
			MetalDiag_CountOncePer(MetalDiagEvent::SampledMipChainRegenerated, entry.texMtl->physAddress,
				"{:016x} ({} mips, format {:04x}, passMask {:x}, renderMask {:x}, matchedCopyMask {:x}, resampledCopyMask {:x}, partialCopyMask {:x}, lastCopy {}, triggered by {} {:016x}_{:016x} unit {})",
				entry.texMtl->physAddress, entry.texMtl->mipLevels, (uint32)entry.texMtl->format,
				entry.texMtl->GetPassWrittenMipMask(), entry.texMtl->GetRenderWrittenMipMask(),
				entry.texMtl->GetMatchedCopyMipMask(), entry.texMtl->GetResampledCopyMipMask(), entry.texMtl->GetPartialCopyMipMask(),
				DescribeLastCopy(entry.texMtl),
				StageLetter(entry.samplingShader->shaderType), entry.samplingShader->baseHash, entry.samplingShader->auxHash,
				entry.textureUnit);
		}
	}

	EndEncoding();
	auto blitEncoder = GetCommandBuffer()->blitCommandEncoder();
	for (const auto& entry : toGenerate)
		blitEncoder->generateMipmaps(entry.texMtl->GetTexture());
	blitEncoder->endEncoding();
	// the fresh render pass for the draw is opened by the caller's GetRenderCommandEncoder()
	for (const auto& entry : toGenerate)
		entry.texMtl->SetMipGenerationEventCounter(entry.texMtl->lastWriteEventCounter);
}

// Is this texture one of the active FBO's color attachments? A draw that samples one is an attachment
// feedback loop (read-modify-write post-processing), so its reads have to come from a shadow copy
bool MetalRenderer::TextureIsActiveColorAttachment(LatteTexture* baseTexture) const
{
	const auto& fbo = m_state.m_activeFBO.m_fbo;
	if (!fbo)
		return false;
	for (int c = 0; c < 8; c++)
	{
		if (fbo->colorBuffer[c].texture && fbo->colorBuffer[c].texture->baseTexture == baseTexture)
			return true;
	}
	return false;
}

void MetalRenderer::PrepareFeedbackLoopShadowCopies(LatteDecompilerShader* vertexShader, LatteDecompilerShader* geometryShader, LatteDecompilerShader* pixelShader)
{
	m_feedbackShadowTextures.clear();
	const auto& fbo = m_state.m_activeFBO.m_fbo;

	std::vector<std::pair<LatteTexture*, LatteTextureView*>> toCopy;
	const auto scanShader = [&](LatteDecompilerShader* shader)
	{
		if (!shader)
			return;
		sint32 textureCount = shader->resourceMapping.getTextureCount();
		for (int i = 0; i < textureCount; ++i)
		{
			const auto relative_textureUnit = shader->resourceMapping.getRelativeTextureUnitFromRelativeBindingPoint(i);
			auto hostTextureUnit = relative_textureUnit;
			// don't scan textures that are accessed with a framebuffer fetch (not bound at all)
			if (m_supportsFramebufferFetch && shader->textureRenderTargetIndex[relative_textureUnit] != 255)
				continue;
			switch (shader->shaderType)
			{
			case LatteConst::ShaderType::Vertex:
				hostTextureUnit += LATTE_CEMU_VS_TEX_UNIT_BASE;
				break;
			case LatteConst::ShaderType::Pixel:
				hostTextureUnit += LATTE_CEMU_PS_TEX_UNIT_BASE;
				break;
			case LatteConst::ShaderType::Geometry:
				hostTextureUnit += LATTE_CEMU_GS_TEX_UNIT_BASE;
				break;
			default:
				continue;
			}

			auto textureView = m_state.m_textures[hostTextureUnit];
			if (!textureView)
				continue;
			LatteTexture* baseTexture = textureView->baseTexture;
			// sampling a depth texture as plain data (e.g. depth-of-field): refresh the color mirror
			// via a pass break so the copy lands strictly before this draw in the main command
			// buffer. This covers both the regular case (depth written by an earlier pass) and the
			// feedback case (sampling the ACTIVE depth attachment). Depth-compare units read the
			// depth view directly via hardware sample_compare and need no mirror
			if (baseTexture->isDepth)
			{
				if (!shader->textureUsesDepthCompare[relative_textureUnit])
					depthCopy_refreshColorCopyBeforeDraw(textureView);
				continue;
			}
			if (!fbo)
				continue; // no active FBO: no color feedback loop possible, depth refresh was the only work
			const bool isAttachment = TextureIsActiveColorAttachment(baseTexture);
			const bool alreadyQueued = std::any_of(toCopy.begin(), toCopy.end(), [baseTexture](const auto& entry) { return entry.first == baseTexture; });
			if (!isAttachment || alreadyQueued)
				continue;

			auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
			MTL::Texture* src = texMtl->GetTexture();
			// only plain 2D and 2D-array color targets are mirrored; anything else falls back to the
			// (broken) attachment read and is logged
			if (src->textureType() != MTL::TextureType2D && src->textureType() != MTL::TextureType2DArray)
			{
				MetalDiag_Count(MetalDiagEvent::FeedbackLoopUnsupported,
					"attachment feedback loop on unsupported texture type {} ({:016x})", (uint32)src->textureType(), baseTexture->physAddress);
				continue;
			}

			auto& shadowEntry = m_feedbackShadowCopies[baseTexture];
			if (!shadowEntry.texture || shadowEntry.pixelFormat != src->pixelFormat() || shadowEntry.textureType != src->textureType() ||
				shadowEntry.width != src->width() || shadowEntry.height != src->height() ||
				shadowEntry.mipLevels != src->mipmapLevelCount() || shadowEntry.arrayLength != src->arrayLength())
			{
				if (shadowEntry.texture)
				{
					// cached sample views reference the old shadow texture - drop them with it
					shadowEntry.releaseSampleViews();
					shadowEntry.texture->release();
				}
				NS_STACK_SCOPED MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
				desc->setTextureType(src->textureType());
				desc->setPixelFormat(src->pixelFormat());
				desc->setWidth(src->width());
				desc->setHeight(src->height());
				desc->setMipmapLevelCount(src->mipmapLevelCount());
				desc->setArrayLength(src->arrayLength());
				// PixelFormatView: CreateFeedbackShadowView creates swizzled views from the shadow
				// (same pixel format today, but the swizzle-taking newTextureView variants
				// require the usage flag)
				desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsagePixelFormatView);
				shadowEntry.texture = m_device->newTexture(desc);
				// labeled so the texture is identifiable in GPU captures
				shadowEntry.texture->setLabel(ToNSString(fmt::format("Cemu Feedback Shadow {:016x}", baseTexture->physAddress)));
				shadowEntry.pixelFormat = src->pixelFormat();
				shadowEntry.textureType = src->textureType();
				shadowEntry.width = src->width();
				shadowEntry.height = src->height();
				shadowEntry.mipLevels = src->mipmapLevelCount();
				shadowEntry.arrayLength = src->arrayLength();
				if (!shadowEntry.texture)
				{
					MetalDiag_Count(MetalDiagEvent::FeedbackLoopUnsupported,
						"failed to allocate the feedback loop shadow copy ({}x{})", src->width(), src->height());
					m_feedbackShadowCopies.erase(baseTexture);
					continue;
				}
			}
			toCopy.emplace_back(baseTexture, textureView);
			m_feedbackShadowTextures[baseTexture] = shadowEntry.texture;
		}
	};
	scanShader(vertexShader);
	scanShader(geometryShader);
	scanShader(pixelShader);
	if (toCopy.empty())
		return;


	static std::atomic<uint32> s_feedbackLoopLogCount{ 0 };
	if (s_feedbackLoopLogCount.fetch_add(1) < 8)
	{
		cemuLog_logDebug(LogType::Force, "MetalRenderer: attachment feedback loop detected (sampling {} active attachment(s), frame {}, PS {:016x}). Serving the reads from shadow copies",
			toCopy.size(), LatteGPUState.frameCounter, pixelShader ? pixelShader->baseHash : 0);
		// geometry diagnostics: a mismatch between the copied extent and the region the render pass
		// actually wrote (e.g. padding bands, resolution overwrite) shows up as blocky post effects
		for (const auto& [baseTexture, textureView] : toCopy)
		{
			auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
			MTL::Texture* src = texMtl->GetTexture();
			cemuLog_logDebug(LogType::Force, "MetalRenderer: feedback copy for texture {:016x}: {}x{} mips {} layers {} fmt {}, FBO render size {}x{}, sampled view mips {}+{} slices {}+{}, base fmt {} vs view fmt {}",
				baseTexture->physAddress, src->width(), src->height(), src->mipmapLevelCount(), (uint32)src->arrayLength(), (uint32)src->pixelFormat(),
				m_state.m_activeFBO.m_fbo->m_size.x, m_state.m_activeFBO.m_fbo->m_size.y,
				textureView->firstMip, textureView->numMip, textureView->firstSlice, textureView->numSlice,
				(uint32)texMtl->format, (uint32)textureView->format);
		}
	}

	// break the pass and copy the attachments before the feedback draw's pass begins - within one
	// command buffer the blit is strictly ordered between the stored contents of the ended pass and
	// the fresh pass (which reloads the attachments, preserving the game's read-modify-write)
	EndEncoding();
	auto commandBuffer = GetCommandBuffer();
	auto blitEncoder = commandBuffer->blitCommandEncoder();
	for (const auto& [baseTexture, textureView] : toCopy)
	{
		auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
		MTL::Texture* src = texMtl->GetTexture();
		MTL::Texture* dst = m_feedbackShadowCopies[baseTexture].texture;
		blitEncoder->copyFromTexture(src, 0, 0, dst, 0, 0, src->arrayLength(), src->mipmapLevelCount());
	}
	blitEncoder->endEncoding();
	// open the fresh pass for the feedback draw (the caller's GetRenderCommandEncoder() will reuse it)
	GetRenderCommandEncoder();
}

void MetalRenderer::bufferCache_init(const sint32 bufferSize)
{
    m_memoryManager->InitBufferCache(bufferSize);
}

void MetalRenderer::bufferCache_upload(uint8* buffer, sint32 size, uint32 bufferOffset)
{
    m_memoryManager->UploadToBufferCache(buffer, bufferOffset, size);
}

void MetalRenderer::bufferCache_copy(uint32 srcOffset, uint32 dstOffset, uint32 size)
{
    m_memoryManager->CopyBufferCache(srcOffset, dstOffset, size);
}

void MetalRenderer::bufferCache_copyStreamoutToMainBuffer(uint32 srcOffset, uint32 dstOffset, uint32 size)
{
    if (m_memoryManager->UseHostMemoryForCache())
        dstOffset -= m_memoryManager->GetImportedMemBaseAddress();

    CopyBufferToBuffer(GetXfbRingBuffer(), srcOffset, m_memoryManager->GetBufferCache(), dstOffset, size, MTL::RenderStageVertex | MTL::RenderStageMesh, ALL_MTL_RENDER_STAGES);
}

void MetalRenderer::buffer_bindVertexBuffer(uint32 bufferIndex, uint32 offset, uint32 size)
{
    cemu_assert_debug(!m_memoryManager->UseHostMemoryForCache());
    cemu_assert_debug(bufferIndex < LATTE_MAX_VERTEX_BUFFERS);

    m_state.m_vertexBufferOffsets[bufferIndex] = offset;
}

void MetalRenderer::buffer_bindUniformBuffer(LatteConst::ShaderType shaderType, uint32 bufferIndex, uint32 offset, uint32 size)
{
    cemu_assert_debug(!m_memoryManager->UseHostMemoryForCache());

    m_state.m_uniformBufferOffsets[GetMtlGeneralShaderType(shaderType)][bufferIndex] = offset;
}

RendererShader* MetalRenderer::shader_create(RendererShader::ShaderType type, uint64 baseHash, uint64 auxHash, const std::string& source, bool isGameShader, bool isGfxPackShader)
{
    return new RendererShaderMtl(this, type, baseHash, auxHash, isGameShader, isGfxPackShader, source);
}

void MetalRenderer::streamout_setupXfbBuffer(uint32 bufferIndex, sint32 ringBufferOffset, uint32 rangeAddr, uint32 rangeSize)
{
    m_state.m_streamoutState.buffers[bufferIndex].enabled = true;
	m_state.m_streamoutState.buffers[bufferIndex].ringBufferOffset = ringBufferOffset;
}

void MetalRenderer::streamout_begin()
{
    // Do nothing
}

void MetalRenderer::streamout_rendererFinishDrawcall()
{
    // Do nothing
}

void MetalRenderer::draw_beginSequence()
{
    m_state.m_skipDrawSequence = false;

    bool streamoutEnable = LatteGPUState.contextRegister[mmVGT_STRMOUT_EN] != 0;

    // update shader state
	LatteSHRC_UpdateActiveShaders();
	if (LatteGPUState.activeShaderHasError)
	{
		cemuLog_logOnce(LogType::Force, "Skipping drawcalls due to shader error\n");
		m_state.m_skipDrawSequence = true;
		cemu_assert_debug(false);
		return;
	}

	// update render target and texture state
	LatteGPUState.requiresTextureBarrier = false;
	while (true)
	{
		LatteGPUState.repeatTextureInitialization = false;
		if (!LatteMRT::UpdateCurrentFBO())
		{
			cemuLog_logOnce(LogType::Force, "Rendertarget invalid\n");
			m_state.m_skipDrawSequence = true;
			return; // no render target
		}

		if (!hasValidFramebufferAttached && !streamoutEnable)
		{
			cemuLog_logOnce(LogType::Force, "Drawcall with no color buffer or depth buffer attached\n");
			m_state.m_skipDrawSequence = true;
			return; // no render target
		}
		LatteTexture_updateTextures();
		if (!LatteGPUState.repeatTextureInitialization)
			break;
	}

	// apply render target
	LatteMRT::ApplyCurrentState();

	// viewport and scissor box
	LatteRenderTarget_updateViewport();
	LatteRenderTarget_updateScissorBox();

	if (!LatteGPUState.contextNew.IsRasterizationEnabled() && !streamoutEnable)
		m_state.m_skipDrawSequence = true;
}

// Identity address for GPU-trace labels: the first color attachment's physAddress (depth
// if the FBO has no color attachments). CachedFBOMtl itself has no single address
static MPTR GetFBOIdentityPhysAddress(const CachedFBOMtl* fbo)
{
	if (!fbo)
		return MPTR_NULL;
	for (int c = 0; c < 8; c++)
		if (fbo->colorBuffer[c].texture)
			return fbo->colorBuffer[c].texture->baseTexture->physAddress;
	if (fbo->depthBuffer.texture)
		return fbo->depthBuffer.texture->baseTexture->physAddress;
	return MPTR_NULL;
}

void MetalRenderer::draw_execute(uint32 baseVertex, uint32 baseInstance, uint32 instanceCount, uint32 count, MPTR indexDataMPTR, Latte::LATTE_VGT_DMA_INDEX_TYPE::E_INDEX_TYPE indexType, const LatteDrawcallContext& drawcallContext)
{
    if (m_state.m_skipDrawSequence)
	{
	    LatteGPUState.drawCallCounter++;
		return;
	}

	// fast clear color as depth
	if (LatteGPUState.contextNew.GetSpecialStateValues()[8] != 0)
	{
		LatteDraw_handleSpecialState8_clearAsDepth();
		LatteGPUState.drawCallCounter++;
		return;
	}
	else if (LatteGPUState.contextNew.GetSpecialStateValues()[5] != 0)
	{
		draw_handleSpecialState5();
		LatteGPUState.drawCallCounter++;
		return;
	}

	auto& encoderState = m_state.m_encoderState;

	// Vulkan/hardware parity: a guest scissor lying entirely outside the render target (or with
	// zero extent) produces zero fragments. Metal validates the scissor to be within the target
	// and non-empty, so a 1-texel clamp as fallback would still execute fragment writes (and any
	// fragment-side effects) on that texel - skip such draws instead. Exception: when streamout
	// is active the draw must still run so the vertex-stage capture happens (Vulkan also keeps
	// the streamout draw alive in this state)
	const auto& guestScissor = m_state.m_scissor;
	bool guestScissorProducesNoFragments = guestScissor.width == 0 || guestScissor.height == 0;
	if (!guestScissorProducesNoFragments)
	{
		const sint32 scissorPassWidth = std::max(m_state.m_activeFBO.m_fbo->m_size.x, 1);
		const sint32 scissorPassHeight = std::max(m_state.m_activeFBO.m_fbo->m_size.y, 1);
		guestScissorProducesNoFragments = guestScissor.x >= (uint32)scissorPassWidth || guestScissor.y >= (uint32)scissorPassHeight;
	}
	if (guestScissorProducesNoFragments && LatteGPUState.contextRegister[mmVGT_STRMOUT_EN] == 0)
	{
		LatteGPUState.drawCallCounter++;
		return;
	}

    // Shaders
    LatteDecompilerShader* vertexShader = LatteSHRC_GetActiveVertexShader();
    LatteDecompilerShader* geometryShader = LatteSHRC_GetActiveGeometryShader();
    LatteDecompilerShader* pixelShader = LatteSHRC_GetActivePixelShader();
    const auto fetchShader = LatteSHRC_GetActiveFetchShader();

    /*
    bool neverSkipAccurateBarrier = false;

    // "Accurate barriers" is usually enabled globally but since the CPU cost is substantial we allow users to disable it (debug -> 'Accurate barriers' option)
	// We always force accurate barriers for known problematic shaders
	if (pixelShader)
	{
		if (pixelShader->baseHash == 0x6f6f6e7b9aae57af && pixelShader->auxHash == 0x00078787f9249249) // BotW lava
			neverSkipAccurateBarrier = true;
		if (pixelShader->baseHash == 0x4c0bd596e3aef4a6 && pixelShader->auxHash == 0x003c3c3fc9269249) // BotW foam layer for water on the bottom of waterfalls
			neverSkipAccurateBarrier = true;
	}

	// Check if we need to end the render pass
	if (!m_state.m_isFirstDrawInRenderPass && (GetConfig().vk_accurate_barriers || neverSkipAccurateBarrier))
	{
    	// Fragment shader is most likely to require a render pass flush, so check for it first
    	bool endRenderPass = CheckIfRenderPassNeedsFlush(pixelShader);
    	if (!endRenderPass)
    	    endRenderPass = CheckIfRenderPassNeedsFlush(vertexShader);
    	if (!endRenderPass && geometryShader)
    	    endRenderPass = CheckIfRenderPassNeedsFlush(geometryShader);

    	if (endRenderPass)
        {
    	    EndEncoding();
            // TODO: only log in debug?
            cemuLog_logOnce(LogType::Force, "Ending render pass due to render target self-dependency\n");
        }
	}
	*/

    // Primitive type
    const LattePrimitiveMode primitiveMode = LatteGPUState.contextNew.VGT_PRIMITIVE_TYPE.get_PRIMITIVE_MODE();
    auto mtlPrimitiveType = GetMtlPrimitiveType(primitiveMode);

    bool usesGeometryShader = UseGeometryShader(LatteGPUState.contextNew, geometryShader != nullptr);
    if (usesGeometryShader && !m_supportsMeshShaders)
        return;

    bool fetchVertexManually = (usesGeometryShader || fetchShader->mtlFetchVertexManually);

	// Index buffer
	Renderer::INDEX_TYPE hostIndexType;
	uint32 hostIndexCount;
	uint32 indexMax = 0;
	Renderer::IndexAllocation indexAllocation;
	LatteIndices_decode(memory_getPointerFromVirtualOffset(indexDataMPTR), indexType, count, primitiveMode, indexMax, hostIndexType, hostIndexCount, indexAllocation);
	auto indexAllocationMtl = static_cast<MetalSynchronizedHeapAllocator::AllocatorReservation*>(indexAllocation.rendererInternal);

	// Buffer cache
	if (m_memoryManager->UseHostMemoryForCache())
	{
		// direct memory access (Wii U memory space imported as a buffer), update buffer bindings
		draw_updateVertexBuffersDirectAccess();
		if (vertexShader)
			draw_updateUniformBuffersDirectAccess(vertexShader, mmSQ_VTX_UNIFORM_BLOCK_START);
		if (geometryShader)
			draw_updateUniformBuffersDirectAccess(geometryShader, mmSQ_GS_UNIFORM_BLOCK_START);
		if (pixelShader)
			draw_updateUniformBuffersDirectAccess(pixelShader, mmSQ_PS_UNIFORM_BLOCK_START);
	}
	else
	{
    	// synchronize vertex and uniform cache and update buffer bindings
    	// We need to call this before getting the render command encoder, since it can cause buffer copies
		uint8 stageUniformModifiedMask = 0;
    	LatteBufferCache_Sync(indexMax + baseVertex, baseInstance, instanceCount, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, stageUniformModifiedMask);
	}

	// Attachment feedback loops: detect before the pass for this draw is acquired, so the current
	// pass can be broken and the sampled attachments copied (see PrepareFeedbackLoopShadowCopies)
	EnsureSampledMipContentValid(vertexShader, geometryShader, pixelShader);
	PrepareFeedbackLoopShadowCopies(vertexShader, geometryShader, pixelShader);

	// Render pass
	auto renderCommandEncoder = GetRenderCommandEncoder();

    // Render pipeline state
    PipelineObject* pipelineObj = m_pipelineCache->GetRenderPipelineState(fetchShader, vertexShader, geometryShader, pixelShader, m_state.m_lastUsedFBO.m_attachmentsInfo, m_state.m_activeFBO.m_attachmentsInfo, m_state.m_activeFBO.m_fbo->m_size, count, LatteGPUState.contextNew);
    if (!pipelineObj->m_pipeline)
        return;

    if (pipelineObj->m_pipeline != encoderState.m_renderPipelineState)
   	{
        renderCommandEncoder->setRenderPipelineState(pipelineObj->m_pipeline);
  		encoderState.m_renderPipelineState = pipelineObj->m_pipeline;
   	}

	// Depth stencil state

	// Disable depth write when there is no depth attachment
	auto& depthControl = LatteGPUState.contextNew.DB_DEPTH_CONTROL;
	bool depthWriteEnable = depthControl.get_Z_WRITE_ENABLE();
	if (!m_state.m_activeFBO.m_fbo->depthBuffer.texture)
	    depthControl.set_Z_WRITE_ENABLE(false);

	MTL::DepthStencilState* depthStencilState = m_depthStencilCache->GetDepthStencilState(LatteGPUState.contextNew);
	if (depthStencilState != encoderState.m_depthStencilState)
	{
	    renderCommandEncoder->setDepthStencilState(depthStencilState);
		encoderState.m_depthStencilState = depthStencilState;
	}

	// Restore the original depth write state
	depthControl.set_Z_WRITE_ENABLE(depthWriteEnable);

	// Stencil reference
	bool stencilEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ENABLE();
	if (stencilEnable)
	{
	    bool backStencilEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_BACK_STENCIL_ENABLE();
		uint32 stencilRefFront = LatteGPUState.contextNew.DB_STENCILREFMASK.get_STENCILREF_F();
    	uint32 stencilRefBack;
        if (backStencilEnable)
            stencilRefBack = LatteGPUState.contextNew.DB_STENCILREFMASK_BF.get_STENCILREF_B();
        else
            stencilRefBack = stencilRefFront;

	    if (stencilRefFront != encoderState.m_stencilRefFront || stencilRefBack != encoderState.m_stencilRefBack)
		{
		    renderCommandEncoder->setStencilReferenceValues(stencilRefFront, stencilRefBack);

			encoderState.m_stencilRefFront = stencilRefFront;
			encoderState.m_stencilRefBack = stencilRefBack;
		}
	}

	// Blend color
	uint32* blendColorConstantU32 = LatteGPUState.contextRegister + Latte::REGADDR::CB_BLEND_RED;

	if (blendColorConstantU32[0] != encoderState.m_blendColor[0] || blendColorConstantU32[1] != encoderState.m_blendColor[1] || blendColorConstantU32[2] != encoderState.m_blendColor[2] || blendColorConstantU32[3] != encoderState.m_blendColor[3])
	{
    	float* blendColorConstant = (float*)LatteGPUState.contextRegister + Latte::REGADDR::CB_BLEND_RED;
    	renderCommandEncoder->setBlendColor(blendColorConstant[0], blendColorConstant[1], blendColorConstant[2], blendColorConstant[3]);

        encoderState.m_blendColor[0] = blendColorConstantU32[0];
        encoderState.m_blendColor[1] = blendColorConstantU32[1];
        encoderState.m_blendColor[2] = blendColorConstantU32[2];
        encoderState.m_blendColor[3] = blendColorConstantU32[3];
	}

	// polygon control
	const auto& polygonControlReg = LatteGPUState.contextNew.PA_SU_SC_MODE_CNTL;
	const auto frontFace = polygonControlReg.get_FRONT_FACE();
	uint32 cullFront = polygonControlReg.get_CULL_FRONT();
	uint32 cullBack = polygonControlReg.get_CULL_BACK();
	uint32 polyOffsetFrontEnable = polygonControlReg.get_OFFSET_FRONT_ENABLED();

	if (polyOffsetFrontEnable)
	{
    	uint32 frontScaleU32 = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_FRONT_SCALE.getRawValue();
    	uint32 frontOffsetU32 = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_FRONT_OFFSET.getRawValue();
    	uint32 offsetClampU32 = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_CLAMP.getRawValue();

        if (frontOffsetU32 != encoderState.m_depthBias || frontScaleU32 != encoderState.m_depthSlope || offsetClampU32 != encoderState.m_depthClamp)
        {
           	float frontScale = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_FRONT_SCALE.get_SCALE();
           	float frontOffset = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_FRONT_OFFSET.get_OFFSET();
           	float offsetClamp = LatteGPUState.contextNew.PA_SU_POLY_OFFSET_CLAMP.get_CLAMP();

           	frontScale /= 16.0f;

            renderCommandEncoder->setDepthBias(frontOffset, frontScale, offsetClamp);

            encoderState.m_depthBias = frontOffsetU32;
            encoderState.m_depthSlope = frontScaleU32;
            encoderState.m_depthClamp = offsetClampU32;
        }
	}
	else
	{
	    if (0 != encoderState.m_depthBias || 0 != encoderState.m_depthSlope || 0 != encoderState.m_depthClamp)
		{
	        renderCommandEncoder->setDepthBias(0.0f, 0.0f, 0.0f);

			encoderState.m_depthBias = 0;
			encoderState.m_depthSlope = 0;
			encoderState.m_depthClamp = 0;
		}
	}

	// Depth clip mode
	cemu_assert_debug(LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_NEAR_DISABLE() == LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_FAR_DISABLE()); // near or far clipping can be disabled individually
	bool zClipEnable = LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_FAR_DISABLE() == false;

	if (zClipEnable != encoderState.m_depthClipEnable)
	{
	    renderCommandEncoder->setDepthClipMode(zClipEnable ? MTL::DepthClipModeClip : MTL::DepthClipModeClamp);
        encoderState.m_depthClipEnable = zClipEnable;
	}

	// Visibility result mode. Metal keeps counting into the last query's slots until the mode is
	// explicitly disabled, so draws outside a query must not be left counting (they would corrupt
	// the finished query's result)
	if (m_occlusionQuery.m_active)
	{
	    renderCommandEncoder->setVisibilityResultMode(MTL::VisibilityResultModeCounting, m_occlusionQuery.m_currentIndex * sizeof(uint64));
		encoderState.m_visibilityResultCounting = true;
	}
	else if (encoderState.m_visibilityResultCounting)
	{
	    renderCommandEncoder->setVisibilityResultMode(MTL::VisibilityResultModeDisabled, 0);
		encoderState.m_visibilityResultCounting = false;
	}

	// todo - how does culling behave with rects?
	// right now we just assume that their winding is always CW
	if (primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::RECTS)
	{
		if (frontFace == Latte::LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CW)
			cullFront = cullBack;
		else
			cullBack = cullFront;
	}

	// Cull mode

	// Cull front and back is handled by disabling rasterization
	if (!(cullFront && cullBack))
	{
        MTL::CullMode cullMode;
       	if (cullFront)
      		cullMode = MTL::CullModeFront;
       	else if (cullBack)
      		cullMode = MTL::CullModeBack;
       	else
      		cullMode = MTL::CullModeNone;

        if (cullMode != encoderState.m_cullMode)
       	{
       	    renderCommandEncoder->setCullMode(cullMode);
      		encoderState.m_cullMode = cullMode;
       	}
	}

	// Front face
	MTL::Winding frontFaceWinding;
	if (frontFace == Latte::LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CCW)
		frontFaceWinding = MTL::WindingCounterClockwise;
	else
		frontFaceWinding = MTL::WindingClockwise;

    if (frontFaceWinding != encoderState.m_frontFaceWinding)
   	{
   	    renderCommandEncoder->setFrontFacingWinding(frontFaceWinding);
  		encoderState.m_frontFaceWinding = frontFaceWinding;
   	}

    // Viewport
    if (m_state.m_viewport.originX != encoderState.m_viewport.originX ||
        m_state.m_viewport.originY != encoderState.m_viewport.originY ||
        m_state.m_viewport.width != encoderState.m_viewport.width ||
        m_state.m_viewport.height != encoderState.m_viewport.height ||
        m_state.m_viewport.znear != encoderState.m_viewport.znear ||
        m_state.m_viewport.zfar != encoderState.m_viewport.zfar)
    {
        renderCommandEncoder->setViewport(m_state.m_viewport);

        encoderState.m_viewport = m_state.m_viewport;
    }

    // Scissor
    if (!encoderState.m_scissorSet ||
        m_state.m_scissor.x != encoderState.m_scissor.x ||
        m_state.m_scissor.y != encoderState.m_scissor.y ||
        m_state.m_scissor.width != encoderState.m_scissor.width ||
        m_state.m_scissor.height != encoderState.m_scissor.height)
    {
        // clamp the scissor to the render pass dimensions (Metal validates the scissor rect
        // against the render target size; attachment-less passes render into a 1x1 dummy texture)
        uint32 passWidth;
        uint32 passHeight;
        if (m_state.m_activeFBO.m_fbo->colorBuffer[0].texture == nullptr && m_state.m_activeFBO.m_fbo->depthBuffer.texture == nullptr)
        {
            passWidth = 1;
            passHeight = 1;
        }
        else
        {
            passWidth = (uint32)std::max(m_state.m_activeFBO.m_fbo->m_size.x, 1);
            passHeight = (uint32)std::max(m_state.m_activeFBO.m_fbo->m_size.y, 1);
        }
        MTL::ScissorRect scissor = m_state.m_scissor;
        scissor.x = std::min<uint32>(scissor.x, passWidth - 1);
        scissor.y = std::min<uint32>(scissor.y, passHeight - 1);
        scissor.width = std::min<uint32>(scissor.width, passWidth - scissor.x);
        scissor.height = std::min<uint32>(scissor.height, passHeight - scissor.y);
        // Metal requires non-empty scissors within the target. Zero-fragment scissors (guest
        // fully outside / zero extent) are skipped upstream in draw_execute unless streamout is
        // active; only those reach this point, so clamp to a single texel at the target edge
        // (Vulkan-parity residual: that texel's fragment still executes, but the streamout
        // capture we keep the draw alive for is vertex-side and unaffected)
        scissor.width = std::max<uint32>(scissor.width, 1);
        scissor.height = std::max<uint32>(scissor.height, 1);

        // store the RAW guest scissor for the comparison above - storing the clamped rect would
        // compare unequal against the still-unclamped state and re-issue setScissorRect every
        // draw whenever the clamp engaged
        encoderState.m_scissor = m_state.m_scissor;
        encoderState.m_scissorSet = true;
        renderCommandEncoder->setScissorRect(scissor);
    }

	// Resources

	// Vertex buffers
    for (uint8 i = 0; i < MAX_MTL_VERTEX_BUFFERS; i++)
    {
        size_t offset = m_state.m_vertexBufferOffsets[i];
        if (offset != INVALID_OFFSET)
        {
            // Bind
            SetBuffer(renderCommandEncoder, GetMtlShaderType(vertexShader->shaderType, usesGeometryShader), m_memoryManager->GetBufferCache(), offset, GET_MTL_VERTEX_BUFFER_INDEX(i));
        }
    }

	// Prepare streamout
	m_state.m_streamoutState.verticesPerInstance = count;
	LatteStreamout_PrepareDrawcall(count, instanceCount);

	// Uniform buffers, textures and samplers
	BindStageResources(renderCommandEncoder, vertexShader, usesGeometryShader);
	if (usesGeometryShader && geometryShader)
	    BindStageResources(renderCommandEncoder, geometryShader, usesGeometryShader);
	BindStageResources(renderCommandEncoder, pixelShader, usesGeometryShader);


	// Draw
	if (usesGeometryShader)
	{
	    if (hostIndexType != INDEX_TYPE::NONE)
		    SetBuffer(renderCommandEncoder, METAL_SHADER_TYPE_OBJECT, indexAllocationMtl->mtlBuffer, indexAllocationMtl->bufferOffset, vertexShader->resourceMapping.indexBufferBinding);
		else
		{
			// the object shader always declares indexBuffer (see emitInputs in the MSL header), so a
			// dummy buffer must be bound for non-indexed draws. fetchVertex never reads it there;
			// leaving the binding unset fails Metal validation (null buffers read as zero otherwise)
			if (!m_meshIndexDummyBuffer)
				m_meshIndexDummyBuffer = m_device->newBuffer(sizeof(uint32), MTL::ResourceStorageModePrivate);
			SetBuffer(renderCommandEncoder, METAL_SHADER_TYPE_OBJECT, m_meshIndexDummyBuffer, 0, vertexShader->resourceMapping.indexBufferBinding);
		}

		uint8 hostIndexTypeU8 = (uint8)hostIndexType;
		renderCommandEncoder->setObjectBytes(&hostIndexTypeU8, sizeof(hostIndexTypeU8), vertexShader->resourceMapping.indexTypeBinding);
        encoderState.m_buffers[METAL_SHADER_TYPE_OBJECT][vertexShader->resourceMapping.indexTypeBinding] = {nullptr};

		// mesh-path vertex shaders get the vertex count via a dedicated binding: the SupportBuffer
		// stays byte-identical to the Vulkan ufBlock, which only carries verticesPerInstance for
		// the SSBO streamout path
		sint32 meshVerticesPerInstance = (sint32)count;
		renderCommandEncoder->setObjectBytes(&meshVerticesPerInstance, sizeof(meshVerticesPerInstance), vertexShader->resourceMapping.verticesPerInstanceBinding);
		encoderState.m_buffers[METAL_SHADER_TYPE_OBJECT][vertexShader->resourceMapping.verticesPerInstanceBinding] = {nullptr};

		// one object threadgroup per input primitive. The per-primitive vertex count and the
		// threadgroup count must be derived from the data the index decoder produced (QUADS and
		// QUAD_STRIP arrive as triangle lists, TRIANGLE_FAN as a triangle strip, LINE_LOOP as a
		// reconnecting line strip) - a single generic formula cannot express all of these
		uint32 verticesPerPrimitive = GetVerticesPerPrimitive(primitiveMode);
		uint32 threadgroupCount;
		switch (primitiveMode)
		{
		case LattePrimitiveMode::POINTS:
			threadgroupCount = count;
			break;
		case LattePrimitiveMode::LINES:
			threadgroupCount = count / 2;
			break;
		case LattePrimitiveMode::LINE_STRIP:
			threadgroupCount = (count >= 2) ? count - 1 : 0;
			break;
		case LattePrimitiveMode::LINE_LOOP:
			// rewritten into a line strip with one extra connecting vertex
			threadgroupCount = (hostIndexCount >= 2) ? hostIndexCount - 1 : 0;
			break;
		case LattePrimitiveMode::TRIANGLES:
		case LattePrimitiveMode::RECTS:
			threadgroupCount = count / 3;
			break;
		case LattePrimitiveMode::TRIANGLE_STRIP:
		case LattePrimitiveMode::TRIANGLE_FAN:
			threadgroupCount = (count >= 3) ? count - 2 : 0;
			break;
		case LattePrimitiveMode::QUADS:
		case LattePrimitiveMode::QUAD_STRIP:
			// rewritten into triangle lists (6 indices per quad)
			threadgroupCount = hostIndexCount / 3;
			break;
		default:
			cemuLog_logOnce(LogType::Force, "Unimplemented primitive type {} for mesh draws, draw skipped", primitiveMode);
			threadgroupCount = 0;
			break;
		}
		threadgroupCount *= instanceCount;

	// Metal rejects zero-size mesh grids - skip only the draw call, the post-draw bookkeeping
	// below must still run
	if (threadgroupCount > 0)
		renderCommandEncoder->drawMeshThreadgroups(MTL::Size(threadgroupCount, 1, 1), MTL::Size(verticesPerPrimitive, 1, 1), MTL::Size(1, 1, 1));
	}
	else
	{
        if (hostIndexType != INDEX_TYPE::NONE)
       	{
       	    auto mtlIndexType = GetMtlIndexType(hostIndexType);
      		renderCommandEncoder->drawIndexedPrimitives(mtlPrimitiveType, hostIndexCount, mtlIndexType, indexAllocationMtl->mtlBuffer, indexAllocationMtl->bufferOffset, instanceCount, baseVertex, baseInstance);
       	}
       	else
       	{
      		renderCommandEncoder->drawPrimitives(mtlPrimitiveType, baseVertex, count, instanceCount, baseInstance);
       	}
	}

	m_state.m_isFirstDrawInRenderPass = false;

	// Occlusion queries
	if (m_occlusionQuery.m_active)
	    m_occlusionQuery.m_currentIndex = (m_occlusionQuery.m_currentIndex + 1) % OCCLUSION_QUERY_POOL_SIZE;

	// Streamout
	LatteStreamout_FinishDrawcall(m_memoryManager->UseHostMemoryForCache());

	// Debug
	if (fetchVertexManually)
	    m_performanceMonitor.m_manualVertexFetchDraws++;
	if (usesGeometryShader)
	    m_performanceMonitor.m_meshDraws++;
	if (primitiveMode == LattePrimitiveMode::TRIANGLE_FAN)
	    m_performanceMonitor.m_triangleFans++;

	LatteGPUState.drawCallCounter++;
}

void MetalRenderer::draw_endSequence()
{
    LatteDecompilerShader* pixelShader = LatteSHRC_GetActivePixelShader();
	// post-drawcall logic
	if (pixelShader)
		LatteRenderTarget_trackUpdates();
	bool hasReadback = LatteTextureReadback_Update();
	m_recordedDrawcalls++;
	// The number of draw calls needs to twice as big, since we are interrupting the render pass
	// TODO: ucomment?
	if (m_recordedDrawcalls >= m_commitTreshold * 2/* || hasReadback*/)
	{
		CommitCommandBuffer();

        // TODO: where should this be called?
        LatteTextureReadback_UpdateFinishedTransfers(false);
	}
}

void MetalRenderer::draw_updateVertexBuffersDirectAccess()
{
	LatteFetchShader* parsedFetchShader = LatteSHRC_GetActiveFetchShader();
	if (!parsedFetchShader)
		return;

	for (auto& bufferGroup : parsedFetchShader->bufferGroups)
	{
		uint32 bufferIndex = bufferGroup.attributeBufferIndex;
		uint32 bufferBaseRegisterIndex = mmSQ_VTX_ATTRIBUTE_BLOCK_START + bufferIndex * 7;
		MPTR bufferAddress = LatteGPUState.contextRegister[bufferBaseRegisterIndex + 0];

		if (bufferAddress == MPTR_NULL) [[unlikely]]
			bufferAddress = m_memoryManager->GetImportedMemBaseAddress();

		m_state.m_vertexBufferOffsets[bufferIndex] = bufferAddress - m_memoryManager->GetImportedMemBaseAddress();
	}
}

void MetalRenderer::draw_updateUniformBuffersDirectAccess(LatteDecompilerShader* shader, const uint32 uniformBufferRegOffset)
{
	if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CBANK)
	{
		for (const auto& buf : shader->list_quickBufferList)
		{
			sint32 i = buf.index;
			MPTR physicalAddr = LatteGPUState.contextRegister[uniformBufferRegOffset + i * 7 + 0];
			uint32 uniformSize = LatteGPUState.contextRegister[uniformBufferRegOffset + i * 7 + 1] + 1;

			if (physicalAddr == MPTR_NULL) [[unlikely]]
			{
				cemu_assert_unimplemented();
				continue;
			}
			uniformSize = std::min<uint32>(uniformSize, buf.size);

			cemu_assert_debug(physicalAddr < 0x50000000);

			uint32 bufferIndex = i;
			cemu_assert_debug(bufferIndex < 16);

			m_state.m_uniformBufferOffsets[GetMtlGeneralShaderType(shader->shaderType)][bufferIndex] = physicalAddr - m_memoryManager->GetImportedMemBaseAddress();
		}
	}
}

void MetalRenderer::draw_handleSpecialState5()
{
    LatteMRT::UpdateCurrentFBO();
	LatteRenderTarget_updateViewport();

	LatteTextureView* colorBuffer = LatteMRT::GetColorAttachment(0);
	LatteTextureView* depthBuffer = LatteMRT::GetDepthAttachment();

	if (!colorBuffer || !depthBuffer)
	{
		cemuLog_logDebug(LogType::Force, "draw_handleSpecialState5(): missing color or depth attachment");
		return;
	}

	sint32 vpWidth, vpHeight;
	LatteMRT::GetVirtualViewportDimensions(vpWidth, vpHeight);

	// transfer depth buffer data to color buffer
	surfaceCopy_copySurfaceWithFormatConversion(
		depthBuffer->baseTexture, depthBuffer->firstMip, depthBuffer->firstSlice,
		colorBuffer->baseTexture, colorBuffer->firstMip, colorBuffer->firstSlice,
		vpWidth, vpHeight);
}

Renderer::IndexAllocation MetalRenderer::indexData_reserveIndexMemory(uint32 size)
{
    auto allocation = m_memoryManager->GetIndexAllocator().AllocateBufferMemory(size, 128);

    return {allocation->memPtr, allocation};
}

void MetalRenderer::indexData_releaseIndexMemory(IndexAllocation& allocation)
{
    m_memoryManager->GetIndexAllocator().FreeReservation(static_cast<MetalSynchronizedHeapAllocator::AllocatorReservation*>(allocation.rendererInternal));
}

void MetalRenderer::indexData_uploadIndexMemory(IndexAllocation& allocation)
{
    m_memoryManager->GetIndexAllocator().FlushReservation(static_cast<MetalSynchronizedHeapAllocator::AllocatorReservation*>(allocation.rendererInternal));
}

LatteQueryObject* MetalRenderer::occlusionQuery_create() {
	return new LatteQueryObjectMtl(this);
}

void MetalRenderer::occlusionQuery_destroy(LatteQueryObject* queryObj) {
    auto queryObjMtl = static_cast<LatteQueryObjectMtl*>(queryObj);
    delete queryObjMtl;
}

void MetalRenderer::occlusionQuery_flush() {
    if (m_occlusionQuery.m_lastCommandBuffer)
        m_occlusionQuery.m_lastCommandBuffer->waitUntilCompleted();
}

void MetalRenderer::occlusionQuery_updateState() {
    ProcessFinishedCommandBuffers();
}

void MetalRenderer::SetBuffer(MTL::RenderCommandEncoder* renderCommandEncoder, MetalShaderType shaderType, MTL::Buffer* buffer, size_t offset, uint32 index)
{
    if (offset >= buffer->length())
    {
        // empty range at the end of the buffer (e.g. a zero-sized uniform buffer bound at the end
        // of the buffer cache). Vulkan accepts offset == length with an empty range, Metal rejects
        // it - skip the bind, which is equivalent (the shader cannot read data from an empty range).
        // Clear the state-cache entry so the stale binding from a previous draw isn't kept alive
        // in the state tracking (and would never be re-bound when the same buffer+offset recurs)
        auto& boundBuffer = m_state.m_encoderState.m_buffers[shaderType][index];
        boundBuffer.m_buffer = nullptr;
        boundBuffer.m_offset = 0;
        return;
    }

    auto& boundBuffer = m_state.m_encoderState.m_buffers[shaderType][index];
    if (buffer == boundBuffer.m_buffer && offset == boundBuffer.m_offset)
        return;

    if (buffer == boundBuffer.m_buffer)
    {
        // Update just the offset
        boundBuffer.m_offset = offset;

        switch (shaderType)
        {
        case METAL_SHADER_TYPE_VERTEX:
            renderCommandEncoder->setVertexBufferOffset(offset, index);
            break;
        case METAL_SHADER_TYPE_OBJECT:
            renderCommandEncoder->setObjectBufferOffset(offset, index);
            break;
        case METAL_SHADER_TYPE_MESH:
            renderCommandEncoder->setMeshBufferOffset(offset, index);
            break;
        case METAL_SHADER_TYPE_FRAGMENT:
            renderCommandEncoder->setFragmentBufferOffset(offset, index);
            break;
        }

        return;
    }

    boundBuffer = {buffer, offset};

    switch (shaderType)
    {
    case METAL_SHADER_TYPE_VERTEX:
        renderCommandEncoder->setVertexBuffer(buffer, offset, index);
        break;
    case METAL_SHADER_TYPE_OBJECT:
        renderCommandEncoder->setObjectBuffer(buffer, offset, index);
        break;
    case METAL_SHADER_TYPE_MESH:
        renderCommandEncoder->setMeshBuffer(buffer, offset, index);
        break;
    case METAL_SHADER_TYPE_FRAGMENT:
        renderCommandEncoder->setFragmentBuffer(buffer, offset, index);
        break;
    }
}

void MetalRenderer::SetTexture(MTL::RenderCommandEncoder* renderCommandEncoder, MetalShaderType shaderType, MTL::Texture* texture, uint32 index)
{
    auto& boundTexture = m_state.m_encoderState.m_textures[shaderType][index];
    if (texture == boundTexture)
        return;

    boundTexture = texture;

    switch (shaderType)
    {
    case METAL_SHADER_TYPE_VERTEX:
        renderCommandEncoder->setVertexTexture(texture, index);
        break;
    case METAL_SHADER_TYPE_OBJECT:
        renderCommandEncoder->setObjectTexture(texture, index);
        break;
    case METAL_SHADER_TYPE_MESH:
        renderCommandEncoder->setMeshTexture(texture, index);
        break;
    case METAL_SHADER_TYPE_FRAGMENT:
        renderCommandEncoder->setFragmentTexture(texture, index);
        break;
    }
}

void MetalRenderer::SetSamplerState(MTL::RenderCommandEncoder* renderCommandEncoder, MetalShaderType shaderType, MTL::SamplerState* samplerState, uint32 index)
{
    auto& boundSamplerState = m_state.m_encoderState.m_samplers[shaderType][index];
    if (samplerState == boundSamplerState)
        return;

    boundSamplerState = samplerState;

    switch (shaderType)
    {
    case METAL_SHADER_TYPE_VERTEX:
        renderCommandEncoder->setVertexSamplerState(samplerState, index);
        break;
    case METAL_SHADER_TYPE_OBJECT:
        renderCommandEncoder->setObjectSamplerState(samplerState, index);
        break;
    case METAL_SHADER_TYPE_MESH:
        renderCommandEncoder->setMeshSamplerState(samplerState, index);
        break;
    case METAL_SHADER_TYPE_FRAGMENT:
        renderCommandEncoder->setFragmentSamplerState(samplerState, index);
        break;
    }
}

MTL::CommandBuffer* MetalRenderer::GetCommandBuffer()
{
    // the frame pool is drained + recreated at the end of every SwapBuffers; that leaves the
    // first frame uncovered, so create it lazily here (this runs on the Latte thread before any
    // per-frame work). Without it the frame-1 autoreleases (labels, drawable internals) leak
    if (!m_frameAutoreleasePool)
        m_frameAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    bool needsNewCommandBuffer = (!m_currentCommandBuffer.m_commandBuffer || m_currentCommandBuffer.m_commited);
    if (needsNewCommandBuffer)
	{
        // Debug
        //m_commandQueue->insertDebugCaptureBoundary();

        auto pool = NS::AutoreleasePool::alloc()->init();
	    MTL::CommandBuffer* mtlCommandBuffer = m_commandQueue->commandBuffer()->retain();
		pool->release();
		// frame identity so GPU traces read in game terms
		mtlCommandBuffer->setLabel(ToNSString(fmt::format("cb frame {} submit {}", LatteGPUState.frameCounter, m_performanceMonitor.m_commandBuffers)));
		m_currentCommandBuffer = {mtlCommandBuffer};

		// Wait for the previous command buffer
		if (m_eventValue != -1)
		    mtlCommandBuffer->encodeWait(m_event, m_eventValue);

		m_recordedDrawcalls = 0;
		m_commitTreshold = m_defaultCommitTreshlod;

        // Debug
        m_performanceMonitor.m_commandBuffers++;

		return mtlCommandBuffer;
	}
	else
	{
	    return m_currentCommandBuffer.m_commandBuffer;
	}
}

MTL::RenderCommandEncoder* MetalRenderer::GetTemporaryRenderCommandEncoder(MTL::RenderPassDescriptor* renderPassDescriptor)
{
    EndEncoding();

    auto commandBuffer = GetCommandBuffer();

    auto pool = NS::AutoreleasePool::alloc()->init();
    auto renderCommandEncoder = commandBuffer->renderCommandEncoder(renderPassDescriptor)->retain();
    pool->release();
#ifdef CEMU_DEBUG_ASSERT
    renderCommandEncoder->setLabel(GetLabel("Temporary render command encoder", renderCommandEncoder));
#endif
    m_commandEncoder = renderCommandEncoder;
    m_encoderType = MetalEncoderType::Render;

    // Debug
    m_performanceMonitor.m_renderPasses++;

    return renderCommandEncoder;
}

// Some render passes clear the attachments, forceRecreate is supposed to be used in those cases
MTL::RenderCommandEncoder* MetalRenderer::GetRenderCommandEncoder(bool forceRecreate)
{
    bool fboChanged = m_state.m_fboChanged;
    m_state.m_fboChanged = false;

    // Check if we need to begin a new render pass
    if (m_commandEncoder)
    {
        if (!forceRecreate)
        {
            if (m_encoderType == MetalEncoderType::Render)
            {
                bool needsNewRenderPass = false;
                if (fboChanged)
                {
                    needsNewRenderPass = (m_state.m_lastUsedFBO.m_fbo == nullptr);
                    if (!needsNewRenderPass)
                    {
                        for (uint8 i = 0; i < 8; i++)
                        {
                            // compare in both directions: an attachment added to OR removed from the
                            // active FBO requires a new render pass (the pass descriptor must match
                            // the FBO state, and the pipelines are built from it)
                            if (m_state.m_activeFBO.m_fbo->colorBuffer[i].texture != m_state.m_lastUsedFBO.m_fbo->colorBuffer[i].texture)
                            {
                                needsNewRenderPass = true;
                                break;
                            }
                        }
                    }

                    if (!needsNewRenderPass)
                    {
                        if (m_state.m_activeFBO.m_fbo->depthBuffer.texture != m_state.m_lastUsedFBO.m_fbo->depthBuffer.texture ||
                            (m_state.m_activeFBO.m_fbo->depthBuffer.hasStencil && !m_state.m_lastUsedFBO.m_fbo->depthBuffer.hasStencil))
                        {
                            needsNewRenderPass = true;
                        }
                    }
                }

                if (!needsNewRenderPass)
                {
                    return (MTL::RenderCommandEncoder*)m_commandEncoder;
                }
            }
        }

        EndEncoding();
    }

    auto commandBuffer = GetCommandBuffer();

    auto pool = NS::AutoreleasePool::alloc()->init();
    auto renderCommandEncoder = commandBuffer->renderCommandEncoder(m_state.m_activeFBO.m_fbo->GetRenderPassDescriptor())->retain();
    pool->release();
    // label with the pass identity so GPU traces show which game surface each pass targets
    renderCommandEncoder->setLabel(ToNSString(fmt::format("pass fbo {:07x} draw mask {:x}",
        GetFBOIdentityPhysAddress(m_state.m_activeFBO.m_fbo), (uint32)m_state.m_activeFBO.m_fbo->drawBuffersMask)));
    m_commandEncoder = renderCommandEncoder;
    m_encoderType = MetalEncoderType::Render;

    // this pass writes its attachments at their attached mip LEVEL only (the pass descriptor
    // attaches a single level per attachment; the view's remaining mip range is not touched) -
    // record just that level as render-written content (see BindStageResources' view clamping)
    auto fboMtl = m_state.m_activeFBO.m_fbo;
    for (uint8 i = 0; i < 8; i++)
    {
        if (fboMtl->colorBuffer[i].texture)
        {
            auto texMtl = static_cast<LatteTextureMtl*>(fboMtl->colorBuffer[i].texture->baseTexture);
            texMtl->MarkRenderMipsWritten(fboMtl->colorBuffer[i].texture->firstMip, 1);
            texMtl->MarkPassMipsWritten(fboMtl->colorBuffer[i].texture->firstMip, 1);
        }
    }
    if (fboMtl->depthBuffer.texture)
    {
        auto texMtl = static_cast<LatteTextureMtl*>(fboMtl->depthBuffer.texture->baseTexture);
        texMtl->MarkRenderMipsWritten(fboMtl->depthBuffer.texture->firstMip, 1);
        texMtl->MarkPassMipsWritten(fboMtl->depthBuffer.texture->firstMip, 1);
    }

    // Update state
    m_state.m_lastUsedFBO = m_state.m_activeFBO;
    m_state.m_isFirstDrawInRenderPass = true;

    ResetEncoderState();

    // Debug
    m_performanceMonitor.m_renderPasses++;

    return renderCommandEncoder;
}

MTL::ComputeCommandEncoder* MetalRenderer::GetComputeCommandEncoder()
{
    if (m_commandEncoder)
    {
        if (m_encoderType == MetalEncoderType::Compute)
        {
            return (MTL::ComputeCommandEncoder*)m_commandEncoder;
        }

        EndEncoding();
    }

    auto commandBuffer = GetCommandBuffer();

    auto pool = NS::AutoreleasePool::alloc()->init();
    auto computeCommandEncoder = commandBuffer->computeCommandEncoder()->retain();
    pool->release();
    m_commandEncoder = computeCommandEncoder;
    m_encoderType = MetalEncoderType::Compute;

    ResetEncoderState();

    return computeCommandEncoder;
}

MTL::BlitCommandEncoder* MetalRenderer::GetBlitCommandEncoder()
{
    if (m_commandEncoder)
    {
        if (m_encoderType == MetalEncoderType::Blit)
        {
            return (MTL::BlitCommandEncoder*)m_commandEncoder;
        }

        EndEncoding();
    }

    auto commandBuffer = GetCommandBuffer();

    auto pool = NS::AutoreleasePool::alloc()->init();
    auto blitCommandEncoder = commandBuffer->blitCommandEncoder()->retain();
    pool->release();
    m_commandEncoder = blitCommandEncoder;
    m_encoderType = MetalEncoderType::Blit;

    ResetEncoderState();

    return blitCommandEncoder;
}

void MetalRenderer::EndEncoding()
{
    if (m_commandEncoder)
    {
        m_commandEncoder->endEncoding();
        m_commandEncoder->release();
        m_commandEncoder = nullptr;
        m_encoderType = MetalEncoderType::None;

        // Commit the command buffer if enough draw calls have been recorded
        if (m_recordedDrawcalls >= m_commitTreshold)
            CommitCommandBuffer();
    }
}

void MetalRenderer::CommitCommandBuffer()
{
    if (!m_currentCommandBuffer.m_commandBuffer)
        return;

    EndEncoding();

    ProcessFinishedCommandBuffers();

    // Commit the command buffer
    if (!m_currentCommandBuffer.m_commited)
    {
        // Handled differently, since it seems like Metal doesn't always call the completion handler
        //commandBuffer.m_commandBuffer->addCompletedHandler(^(MTL::CommandBuffer*) {
        //    m_memoryManager->GetTemporaryBufferAllocator().CommandBufferFinished(commandBuffer.m_commandBuffer);
        //});

        // Signal event
        m_eventValue = (m_eventValue + 1) % EVENT_VALUE_WRAP;
        auto mtlCommandBuffer = m_currentCommandBuffer.m_commandBuffer;
        mtlCommandBuffer->encodeSignalEvent(m_event, m_eventValue);

        mtlCommandBuffer->commit();
        m_currentCommandBuffer.m_commited = true;

        m_executingCommandBuffers.push_back(mtlCommandBuffer);

        // Debug
        //m_commandQueue->insertDebugCaptureBoundary();
    }
}

void MetalRenderer::ProcessFinishedCommandBuffers()
{
    // Check for finished command buffers
    for (auto it = m_executingCommandBuffers.begin(); it != m_executingCommandBuffers.end();)
    {
        auto commandBuffer = *it;
        if (CommandBufferCompleted(commandBuffer))
        {
            if (commandBuffer->status() == MTL::CommandBufferStatusError)
            {
                auto error = commandBuffer->error();
                cemuLog_log(LogType::Force, "Metal command buffer failed: {}",
                            error ? error->localizedDescription()->utf8String() : "unknown error");
            }
            m_memoryManager->CleanupBuffers(commandBuffer);
            commandBuffer->release();
            it = m_executingCommandBuffers.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool MetalRenderer::AcquireDrawable(bool mainWindow)
{
    auto& layer = GetLayer(mainWindow);
    if (!layer.GetLayer())
        return false;

    const bool latteBufferUsesSRGB = mainWindow ? LatteGPUState.tvBufferUsesSRGB : LatteGPUState.drcBufferUsesSRGB;
    if (latteBufferUsesSRGB != m_state.m_usesSRGB)
    {
        layer.GetLayer()->setPixelFormat(latteBufferUsesSRGB ? MTL::PixelFormatBGRA8Unorm_sRGB : MTL::PixelFormatBGRA8Unorm);
        m_state.m_usesSRGB = latteBufferUsesSRGB;
    }

    return layer.AcquireDrawable();
}

/*
bool MetalRenderer::CheckIfRenderPassNeedsFlush(LatteDecompilerShader* shader)
{
    sint32 textureCount = shader->resourceMapping.getTextureCount();
	for (int i = 0; i < textureCount; ++i)
	{
		const auto relative_textureUnit = shader->resourceMapping.getTextureUnitFromBindingPoint(i);
		auto hostTextureUnit = relative_textureUnit;
		auto textureDim = shader->textureUnitDim[relative_textureUnit];

		// Texture is accessed as a framebuffer fetch, therefore there is no need to flush it
		if (shader->textureRenderTargetIndex[relative_textureUnit] != 255)
		    continue;

		auto texUnitRegIndex = hostTextureUnit * 7;
		switch (shader->shaderType)
		{
		case LatteConst::ShaderType::Vertex:
			hostTextureUnit += LATTE_CEMU_VS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS;
			break;
		case LatteConst::ShaderType::Pixel:
			hostTextureUnit += LATTE_CEMU_PS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS;
			break;
		case LatteConst::ShaderType::Geometry:
			hostTextureUnit += LATTE_CEMU_GS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_GS;
			break;
		default:
			UNREACHABLE;
		}

		auto textureView = m_state.m_textures[hostTextureUnit];
		if (!textureView)
            continue;

		LatteTexture* baseTexture = textureView->baseTexture;

	    // If the texture is also used in the current render pass, we need to end the render pass to "flush" the texture
		for (uint8 i = 0; i < LATTE_NUM_COLOR_TARGET; i++)
		{
		    auto colorTarget = m_state.m_activeFBO.m_fbo->colorBuffer[i].texture;
			if (colorTarget && colorTarget->baseTexture == baseTexture)
			    return true;
		}
	}

	return false;
}
*/

void MetalRenderer::BindStageResources(MTL::RenderCommandEncoder* renderCommandEncoder, LatteDecompilerShader* shader, bool usesGeometryShader)
{
    auto mtlShaderType = GetMtlShaderType(shader->shaderType, usesGeometryShader);

    sint32 textureCount = shader->resourceMapping.getTextureCount();
	for (int i = 0; i < textureCount; ++i)
	{
		const auto relative_textureUnit = shader->resourceMapping.getRelativeTextureUnitFromRelativeBindingPoint(i);
		auto hostTextureUnit = relative_textureUnit;

		// Don't bind textures that are accessed with a framebuffer fetch
		if (m_supportsFramebufferFetch && shader->textureRenderTargetIndex[relative_textureUnit] != 255)
            continue;

		auto textureDim = shader->textureUnitDim[relative_textureUnit];
		auto texUnitRegIndex = hostTextureUnit * 7;
		switch (shader->shaderType)
		{
		case LatteConst::ShaderType::Vertex:
			hostTextureUnit += LATTE_CEMU_VS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_VS;
			break;
		case LatteConst::ShaderType::Pixel:
			hostTextureUnit += LATTE_CEMU_PS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_PS;
			break;
		case LatteConst::ShaderType::Geometry:
			hostTextureUnit += LATTE_CEMU_GS_TEX_UNIT_BASE;
			texUnitRegIndex += Latte::REGADDR::SQ_TEX_RESOURCE_WORD0_N_GS;
			break;
		default:
			UNREACHABLE;
		}

		uint32 binding = (uint32)shader->resourceMapping.textureUnitToBindingPoint[relative_textureUnit];
		// the analyzer assigns consecutive texture bindings, so this is equivalent to
		// base + i - assert to catch any future change in the binding assignment
		cemu_assert_debug(binding == shader->resourceMapping.getTextureBaseBindingPoint() + i);
		if (binding >= MAX_MTL_TEXTURES)
		{
		    cemuLog_logOnce(LogType::Force, "invalid texture binding {}", binding);
            continue;
		}

		auto textureView = m_state.m_textures[hostTextureUnit];

		if (!textureView)
		{
            if (textureDim == Latte::E_DIM::DIM_1D)
                SetTexture(renderCommandEncoder, mtlShaderType, m_nullTexture1D, binding);
           	else
                SetTexture(renderCommandEncoder, mtlShaderType, m_nullTexture2D, binding);
            SetSamplerState(renderCommandEncoder, mtlShaderType, m_nearestSampler, binding);
            continue;
		}

		if (textureDim == Latte::E_DIM::DIM_1D && (textureView->dim != Latte::E_DIM::DIM_1D))
		{
			// the bound view doesn't match the dimension declared by the shader - bind the null
			// texture of the declared type so the binding stays valid (stale bindings from a
			// previous draw would otherwise survive at this index)
		    SetTexture(renderCommandEncoder, mtlShaderType, m_nullTexture1D, binding);
			SetSamplerState(renderCommandEncoder, mtlShaderType, m_nearestSampler, binding);
			continue;
		}
		else if (textureDim == Latte::E_DIM::DIM_2D && (textureView->dim != Latte::E_DIM::DIM_2D && textureView->dim != Latte::E_DIM::DIM_2D_MSAA))
		{
		    SetTexture(renderCommandEncoder, mtlShaderType, m_nullTexture2D, binding);
			SetSamplerState(renderCommandEncoder, mtlShaderType, m_nearestSampler, binding);
			continue;
		}

		LatteTexture* baseTexture = textureView->baseTexture;

		uint32 stageSamplerIndex = shader->textureUnitSamplerAssignment[relative_textureUnit];
		MTL::SamplerState* sampler;
		// How this unit will sample the chain, carried out of the branch below for the sampled-mip-chain
		// diagnostics. It is the consumer half of the decision: a unit that pins a single LOD above 0
		// reads one specific level, while one with a range reads the pyramid at whatever level the
		// derivative picks - the same chain can need opposite handling for those two regimes
		float unitLodMin = 0.0f;
		float unitLodMax = 0.0f;
		float unitLodBias = 0.0f;
		uint32 unitMipFilter = 0;
		if (stageSamplerIndex != LATTE_DECOMPILER_SAMPLER_NONE)
		{
		    uint32 samplerIndex = stageSamplerIndex + LatteDecompiler_getTextureSamplerBaseIndex(shader->shaderType);
			// apply the overwrites to a local copy of the sampler words - GL and Vulkan do the same.
			// Writing through to the context registers would make the overwrite sticky for every
			// other texture bound to the same sampler index and the relative lod bias would
			// compound on every draw
			_LatteRegisterSetSampler samplerWords = LatteGPUState.contextNew.SQ_TEX_SAMPLER[samplerIndex];

			// Lod bias
            if (baseTexture->overwriteInfo.hasLodBias)
                samplerWords.WORD1.set_LOD_BIAS(baseTexture->overwriteInfo.lodBias);
            else if (baseTexture->overwriteInfo.hasRelativeLodBias)
                samplerWords.WORD1.set_LOD_BIAS(samplerWords.WORD1.get_LOD_BIAS() + baseTexture->overwriteInfo.relativeLodBias);

            // Max anisotropy
            if (baseTexture->overwriteInfo.anisotropicLevel >= 0)
                samplerWords.WORD0.set_MAX_ANISO_RATIO(baseTexture->overwriteInfo.anisotropicLevel);

    		sampler = m_samplerCache->GetSamplerState(LatteGPUState.contextNew, shader->shaderType, stageSamplerIndex, &samplerWords, shader->textureUsesDepthCompare[relative_textureUnit] != 0);

			// LOD fields are in 1/64 units (see MetalSamplerCache and _getSamplerLodBiasMSL), and the
			// sampler words here already include any texture overwrite. MIP_FILTER NONE means the
			// sampler pins itself to level 0 (the sampler cache clamps lodMax to 0.25), so such a unit
			// never reads an upper level at all
			unitLodMin = (float)samplerWords.WORD1.get_MIN_LOD() / 64.0f;
			unitLodMax = (float)samplerWords.WORD1.get_MAX_LOD() / 64.0f;
			unitLodBias = (float)samplerWords.WORD1.get_LOD_BIAS() / 64.0f;
			unitMipFilter = (uint32)samplerWords.WORD0.get_MIP_FILTER();
		}
		else
		{
		    sampler = m_nearestSampler;
		}
        SetSamplerState(renderCommandEncoder, mtlShaderType, sampler, binding);

		// get texture register word 0
		uint32 word4 = LatteGPUState.contextRegister[texUnitRegIndex + 4];

		// Clamp for resolution-overwritten effect surfaces: a view over a chain whose content is
		// not the game's own is served mip 0 only - serving its copy-written upper levels instead
		// brings back the MK8 dithering
		MTL::Texture* clampedView = nullptr; // cache-owned (see LatteTextureViewMtl::GetSwizzledViewWithMipCount) - do not release
		{
			auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
			const uint32 passMask = texMtl->GetPassWrittenMipMask();
			const uint32 renderMask = texMtl->GetRenderWrittenMipMask();
			const bool clampClass = !baseTexture->isDepth && !baseTexture->Is3DTexture() && baseTexture->mipLevels > 1
				&& textureView->numMip > 1 && textureView->firstMip < 32
				&& (textureView->firstMip + textureView->numMip) <= 32
				&& GetMtlPixelFormatInfo(baseTexture->format, false).dataType == MetalDataType::FLOAT;
			if (clampClass)
			{
				// only GPU-written surfaces qualify: CPU-uploaded asset textures (renderMask 0 -
				// content arrives via LatteTextureLoader) must keep their full mip chains, and
				// surfaces that never received GPU content have nothing to fall back to anyway
				const bool gpuWrittenMip0 = (renderMask & 1u) != 0;
				// the discriminator: is the content of the levels this view samples this chain's
				// own? The written masks cannot tell - a faithfully preserved chain and a
				// re-cropped one both look copy-written - so the verdict comes from the copy
				// provenance recorded in the copy paths (LatteTextureMtl::MarkCopyMipsWritten)
				// Only the game's own upper mip levels may be served: the backend's regenerated
				// downsample is no better than clamping to mip 0, because both replace the levels the
				// game's own passes wrote (serving them loses DoF and bloom)
				const bool rangeTrustworthy = texMtl->RangeContentIsTrustworthy(textureView->firstMip, textureView->numMip);
				if (rangeTrustworthy)
				{
					// served exactly as the game asked for it. The per-texture latch lives inside
					// CountOncePer: the decision is per texture, but this is reached once per draw
					// that samples it
					if (rangeTrustworthy)
						MetalDiag_CountFirstSighting(MetalDiagEvent::SampledMipChainTrusted, texMtl->physAddress,
							"{:016x} view mips {}+{} ({} mips total, format {:04x}, guest mips {}, passMask {:x}, renderMask {:x}, matchedCopyMask {:x}, resampledCopyMask {:x}, partialCopyMask {:x}, lastCopy {}, sampled by {} {:016x}_{:016x} unit {} (lod {:.2f}..{:.2f} bias {:+.2f} mipfilter {}))",
							texMtl->physAddress, textureView->firstMip, textureView->numMip, texMtl->mipLevels, (uint32)texMtl->format,
							baseTexture->mipLevels, passMask, renderMask,
							texMtl->GetMatchedCopyMipMask(), texMtl->GetResampledCopyMipMask(), texMtl->GetPartialCopyMipMask(),
							DescribeLastCopy(texMtl),
							StageLetter(shader->shaderType), shader->baseHash, shader->auxHash, relative_textureUnit,
							unitLodMin, unitLodMax, unitLodBias, unitMipFilter);
				}
				else if (gpuWrittenMip0)
				{
					MetalDiag_CountOncePer(MetalDiagEvent::SampledMipChainClamped, texMtl->physAddress,
						"{:016x} view mips {}+{} ({} mips total, format {:04x}, guest mips {}, passMask {:x}, renderMask {:x}, matchedCopyMask {:x}, resampledCopyMask {:x}, partialCopyMask {:x}, lastCopy {}, sampled by {} {:016x}_{:016x} unit {} (lod {:.2f}..{:.2f} bias {:+.2f} mipfilter {}))",
						texMtl->physAddress, textureView->firstMip, textureView->numMip, texMtl->mipLevels, (uint32)texMtl->format,
						baseTexture->mipLevels, passMask, renderMask,
						texMtl->GetMatchedCopyMipMask(), texMtl->GetResampledCopyMipMask(), texMtl->GetPartialCopyMipMask(),
						DescribeLastCopy(texMtl),
						StageLetter(shader->shaderType), shader->baseHash, shader->auxHash, relative_textureUnit,
						unitLodMin, unitLodMax, unitLodBias, unitMipFilter);
					clampedView = textureView->GetSwizzledViewWithMipCount(word4, 1);
				}
			}
		}

		// attachment feedback loop: this draw samples a texture that is an attachment of the active
		// render pass. PrepareFeedbackLoopShadowCopies ended the previous pass and copied the
		// attachment contents into shadow textures before this pass began - bind the shadow copy
		// instead (Metal cannot read a texture attached to the current encoder; Vulkan serves these
		// reads via VK_EXT_attachment_feedback_loop_layout)
		auto shadowItr = m_feedbackShadowTextures.find(baseTexture);
		if (shadowItr != m_feedbackShadowTextures.end())
		{
			auto shadowCopyItr = m_feedbackShadowCopies.find(baseTexture);
			cemu_assert_debug(shadowCopyItr != m_feedbackShadowCopies.end());
			// cached view of the shadow copy (invalidated with the shadow texture) - don't keep
			// it in the encoder state
			MTL::Texture* shadowView = GetFeedbackShadowSampleView(shadowCopyItr->second, textureView, word4);
			SetTexture(renderCommandEncoder, mtlShaderType, shadowView, binding);
			m_state.m_encoderState.m_textures[mtlShaderType][binding] = nullptr;
			continue;
		}

		MTL::Texture* mtlTexture = clampedView ? clampedView : textureView->GetSwizzledView(word4);
		// sampling a depth texture as plain data (e.g. depth-of-field) requires a color copy:
		// Metal rejects depth-format textures on texture2d bindings and depth textures have no
		// compatible color views. Depth-compare sampling is unaffected: the MSL emitter declares
		// depth2d for those units and hardware sample_compare reads the depth view directly
		if (baseTexture->isDepth && !shader->textureUsesDepthCompare[relative_textureUnit])
		{
			// the mirror was refreshed before this draw's pass by PrepareFeedbackLoopShadowCopies
			// (depthCopy_refreshColorCopyBeforeDraw) - only bind it here
			auto texMtl = static_cast<LatteTextureMtl*>(baseTexture);
			auto colorCopy = texMtl->GetDepthColorCopy();
			if (colorCopy)
			{
				// the mirror was refreshed before this draw's pass by PrepareFeedbackLoopShadowCopies
				// (depthCopy_refreshColorCopyBeforeDraw) - only bind it here. The sample view is
				// cached on the base texture (invalidated when the mirror is replaced; see
				// LatteTextureMtl::GetDepthMirrorSampleView for the view semantics) - don't keep
				// it in the encoder state
				MTL::Texture* mirrorView = texMtl->GetDepthMirrorSampleView(textureView, word4);
				SetTexture(renderCommandEncoder, mtlShaderType, mirrorView, binding);
				m_state.m_encoderState.m_textures[mtlShaderType][binding] = nullptr;
				continue;
			}
			// no mirror possible (unsupported texture type, see depthCopy_ensureColorCopy): binding
			// the raw depth view into a regular texture2d slot is rejected by Metal - bind a null
			// texture of the expected type instead so the draw stays valid (the effect will be
			// missing; the log in depthCopy_ensureColorCopy identifies the case)
			SetTexture(renderCommandEncoder, mtlShaderType, textureView->dim == Latte::E_DIM::DIM_2D_ARRAY ? m_nullTexture2DArray : m_nullTexture2D, binding);
			continue;
		}
		SetTexture(renderCommandEncoder, mtlShaderType, mtlTexture, binding);
		if (clampedView)
		{
			// cached single-level view - don't keep it in the encoder state
			m_state.m_encoderState.m_textures[mtlShaderType][binding] = nullptr;
		}
	}

	// Support buffer
	auto GET_UNIFORM_DATA_PTR = [&](size_t index) { return supportBufferData + (index / 4); };

	sint32 shaderAluConst;
	sint32 shaderUniformRegisterOffset;

	switch (shader->shaderType)
	{
	case LatteConst::ShaderType::Vertex:
		shaderAluConst = 0x400;
		shaderUniformRegisterOffset = mmSQ_VTX_UNIFORM_BLOCK_START;
		break;
	case LatteConst::ShaderType::Pixel:
		shaderAluConst = 0;
		shaderUniformRegisterOffset = mmSQ_PS_UNIFORM_BLOCK_START;
		break;
	case LatteConst::ShaderType::Geometry:
		shaderAluConst = 0; // geometry shader has no ALU const
		shaderUniformRegisterOffset = mmSQ_GS_UNIFORM_BLOCK_START;
		break;
	default:
		UNREACHABLE;
	}

	if (shader->resourceMapping.uniformVarsBufferBindingPoint >= 0)
	{
		if (shader->uniform.list_ufTexRescale.empty() == false)
		{
			for (auto& entry : shader->uniform.list_ufTexRescale)
			{
				float* xyScale = LatteTexture_getEffectiveTextureScale(shader->shaderType, entry.texUnit);
				memcpy(entry.currentValue, xyScale, sizeof(float) * 2);
				memcpy(GET_UNIFORM_DATA_PTR(entry.uniformLocation), xyScale, sizeof(float) * 2);
			}
		}
		if (shader->uniform.loc_alphaTestRef >= 0)
		{
			*GET_UNIFORM_DATA_PTR(shader->uniform.loc_alphaTestRef) = LatteGPUState.contextNew.SX_ALPHA_REF.get_ALPHA_TEST_REF();
		}
		if (shader->uniform.loc_pointSize >= 0)
		{
			const auto& pointSizeReg = LatteGPUState.contextNew.PA_SU_POINT_SIZE;
			float pointWidth = (float)pointSizeReg.get_WIDTH() / 8.0f;
			if (pointWidth == 0.0f)
				pointWidth = 1.0f / 8.0f; // minimum size
			*GET_UNIFORM_DATA_PTR(shader->uniform.loc_pointSize) = pointWidth;
		}
		if (shader->uniform.loc_remapped >= 0)
		{
			LatteBufferCache_LoadRemappedUniforms(shader, GET_UNIFORM_DATA_PTR(shader->uniform.loc_remapped), true, (1<<LATTE_NUM_MAX_UNIFORM_BUFFERS)-1);
		}
		if (shader->uniform.loc_uniformRegister >= 0)
		{
			uint32* uniformRegData = (uint32*)(LatteGPUState.contextRegister + mmSQ_ALU_CONSTANT0_0 + shaderAluConst);
			memcpy(GET_UNIFORM_DATA_PTR(shader->uniform.loc_uniformRegister), uniformRegData, shader->uniform.count_uniformRegister * 16);
		}
		if (shader->uniform.loc_windowSpaceToClipSpaceTransform >= 0)
		{
			sint32 viewportWidth;
			sint32 viewportHeight;
			LatteRenderTarget_GetCurrentVirtualViewportSize(&viewportWidth, &viewportHeight); // always call after _updateViewport()
			float* v = GET_UNIFORM_DATA_PTR(shader->uniform.loc_windowSpaceToClipSpaceTransform);
			v[0] = 2.0f / (float)viewportWidth;
			v[1] = 2.0f / (float)viewportHeight;
		}
		if (shader->uniform.loc_fragCoordScale >= 0)
		{
			LatteMRT::GetCurrentFragCoordScale(GET_UNIFORM_DATA_PTR(shader->uniform.loc_fragCoordScale));
		}
		if (shader->uniform.loc_verticesPerInstance >= 0)
		{
			*(int*)(supportBufferData + ((size_t)shader->uniform.loc_verticesPerInstance / 4)) = m_state.m_streamoutState.verticesPerInstance;
			for (sint32 b = 0; b < LATTE_NUM_STREAMOUT_BUFFER; b++)
			{
				if (shader->uniform.loc_streamoutBufferBase[b] >= 0)
				{
					*(uint32*)GET_UNIFORM_DATA_PTR(shader->uniform.loc_streamoutBufferBase[b]) = m_state.m_streamoutState.buffers[b].ringBufferOffset;
				}
			}
		}

		size_t size = shader->uniform.uniformRangeSize;
		auto& bufferAllocator = m_memoryManager->GetStagingAllocator();
		auto allocation = bufferAllocator.AllocateBufferMemory(size, 1);
		memcpy(allocation.memPtr, supportBufferData, size);
		bufferAllocator.FlushReservation(allocation);

		SetBuffer(renderCommandEncoder, mtlShaderType, allocation.mtlBuffer, allocation.bufferOffset, shader->resourceMapping.uniformVarsBufferBindingPoint);
	}

	// Uniform buffers
	for (sint32 i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
	{
		if (shader->resourceMapping.uniformBuffersBindingPoint[i] >= 0)
		{
    		uint32 binding = shader->resourceMapping.uniformBuffersBindingPoint[i];
    		if (binding >= MAX_MTL_BUFFERS)
    		{
    		    cemuLog_logOnce(LogType::Force, "invalid buffer binding {}", binding);
    			continue;
    		}

    		size_t offset = m_state.m_uniformBufferOffsets[GetMtlGeneralShaderType(shader->shaderType)][i];
    		if (offset == INVALID_OFFSET)
                continue;

            SetBuffer(renderCommandEncoder, mtlShaderType, m_memoryManager->GetBufferCache(), offset, binding);
		}
	}

	// Storage buffer
	if (shader->resourceMapping.tfStorageBindingPoint >= 0)
	{
        SetBuffer(renderCommandEncoder, mtlShaderType, GetXfbRingBuffer(), 0, shader->resourceMapping.tfStorageBindingPoint);
	}
}

void MetalRenderer::ClearColorTextureInternal(MTL::Texture* mtlTexture, sint32 sliceIndex, sint32 mipIndex, float r, float g, float b, float a)
{
    NS_STACK_SCOPED MTL::RenderPassDescriptor* renderPassDescriptor = MTL::RenderPassDescriptor::alloc()->init();
    auto colorAttachment = renderPassDescriptor->colorAttachments()->object(0);
    colorAttachment->setTexture(mtlTexture);
    colorAttachment->setClearColor(MTL::ClearColor(r, g, b, a));
    colorAttachment->setLoadAction(MTL::LoadActionClear);
    colorAttachment->setStoreAction(MTL::StoreActionStore);
    colorAttachment->setSlice(sliceIndex);
    colorAttachment->setLevel(mipIndex);

    GetTemporaryRenderCommandEncoder(renderPassDescriptor);
    EndEncoding();

    // Debug
    m_performanceMonitor.m_clears++;
}

void MetalRenderer::CopyBufferToBuffer(MTL::Buffer* src, uint32 srcOffset, MTL::Buffer* dst, uint32 dstOffset, uint32 size, MTL::RenderStages after, MTL::RenderStages before)
{
    // TODO: uncomment and fix performance issues
    // Do the copy in a vertex shader on Apple GPUs
    /*
    if (m_isAppleGPU && m_encoderType == MetalEncoderType::Render)
    {
        auto renderCommandEncoder = static_cast<MTL::RenderCommandEncoder*>(m_commandEncoder);

        MTL::Resource* barrierBuffers[] = {src};
        renderCommandEncoder->memoryBarrier(barrierBuffers, 1, after, after | MTL::RenderStageVertex);

		renderCommandEncoder->setRenderPipelineState(m_copyBufferToBufferPipeline->GetRenderPipelineState());
		m_state.m_encoderState.m_renderPipelineState = m_copyBufferToBufferPipeline->GetRenderPipelineState();

		SetBuffer(renderCommandEncoder, METAL_SHADER_TYPE_VERTEX, src, srcOffset, GET_HELPER_BUFFER_BINDING(0));
		SetBuffer(renderCommandEncoder, METAL_SHADER_TYPE_VERTEX, dst, dstOffset, GET_HELPER_BUFFER_BINDING(1));

		renderCommandEncoder->drawPrimitives(MTL::PrimitiveTypePoint, NS::UInteger(0), NS::UInteger(size));

		barrierBuffers[0] = dst;
        renderCommandEncoder->memoryBarrier(barrierBuffers, 1, before | MTL::RenderStageVertex, before);
    }
    else
    {
    */
        auto blitCommandEncoder = GetBlitCommandEncoder();

        blitCommandEncoder->copyFromBuffer(src, srcOffset, dst, dstOffset, size);
    //}
}

void MetalRenderer::SwapBuffer(bool mainWindow)
{
    if (!AcquireDrawable(mainWindow))
        return;

    auto commandBuffer = GetCommandBuffer();
    GetLayer(mainWindow).PresentDrawable(commandBuffer);
}

void MetalRenderer::EnsureImGuiBackend()
{
    if (!ImGui::GetIO().BackendRendererUserData)
    {
        ImGui_ImplMetal_Init(m_device);
        //ImGui_ImplMetal_CreateFontsTexture(m_device);
    }
}

// Output path of the capture currently being written, for the completion log line
static std::string s_gputracePath;

void MetalRenderer::StartCapture()
{
    auto captureManager = MTL::CaptureManager::sharedCaptureManager();
    auto desc = MTL::CaptureDescriptor::alloc()->init();
    desc->setCaptureObject(m_device);

    std::string captureDir = GetConfig().gpu_capture_dir.GetValue();

    // Check if a debugger with support for GPU capture is attached
    if (captureManager->supportsDestination(MTL::CaptureDestinationDeveloperTools))
    {
        desc->setDestination(MTL::CaptureDestinationDeveloperTools);
    }
    else
    {
        if (captureDir.empty())
        {
            cemuLog_log(LogType::Force, "No GPU capture directory specified, cannot do a GPU capture");
            return;
        }

        // Check if the GPU trace document destination is available. Only a warning:
        // supportsDestination has been observed returning false on this OS even when the
        // capture works, so startCapture below gets the final say (its error is logged)
        if (!captureManager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument))
        {
            cemuLog_log(LogType::Force, "GPU trace document destination reported unavailable; attempting the capture anyway");
        }

        // Get current date and time as a string
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S");
        std::string now_str = oss.str();

        std::string capturePath = fmt::format("{}/cemu_{}.gputrace", captureDir, now_str);
        desc->setDestination(MTL::CaptureDestinationGPUTraceDocument);
        desc->setOutputURL(ToNSURL(capturePath));
        cemuLog_log(LogType::Force, "GPU capture: writing {}", capturePath);
        s_gputracePath = capturePath;
    }

    NS::Error* error = nullptr;
    captureManager->startCapture(desc, &error);
    if (error)
    {
        cemuLog_log(LogType::Force, "Failed to start GPU capture: {}", error->localizedDescription()->utf8String());
        // no capture will run: cancel the done-marker so automation doesn't wait on a
        // trace file that will never exist
        s_gputracePath.clear();
    }

    m_capturing = true;
}

void MetalRenderer::EndCapture()
{
    auto captureManager = MTL::CaptureManager::sharedCaptureManager();
    captureManager->stopCapture();

    m_capturing = false;

    if (!s_gputracePath.empty())
    {
        cemuLog_log(LogType::Force, "GPU capture finished: {}", s_gputracePath);
        s_gputracePath.clear();
    }
}
