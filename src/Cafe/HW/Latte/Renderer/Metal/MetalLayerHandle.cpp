#include "Cafe/HW/Latte/Renderer/Metal/MetalLayerHandle.h"
#include "Cafe/HW/Latte/Renderer/Metal/MetalLayer.h"

#include "gui/interface/WindowSystem.h"

MetalLayerHandle::MetalLayerHandle(MTL::Device* device, const Vector2i& size, bool mainWindow)
{
    const auto& windowInfo = (mainWindow ? WindowSystem::GetWindowInfo().window_main : WindowSystem::GetWindowInfo().window_pad);

    m_layer = (CA::MetalLayer*)CreateMetalLayer(windowInfo.surface, m_layerScaleX, m_layerScaleY, &m_childView);
    m_layer->setDevice(device);
    m_layer->setDrawableSize(CGSize{(float)size.x * m_layerScaleX, (float)size.y * m_layerScaleY});
    m_layer->setFramebufferOnly(true);
}

MetalLayerHandle::~MetalLayerHandle()
{
    ReleaseAll();
}

// drops every owned reference and the view. Used by the destructor and by move assignment
void MetalLayerHandle::ReleaseAll()
{
    if (m_drawable)
    {
        m_drawable->release();
        m_drawable = nullptr;
    }
    // remove the view first: it holds its own reference to the layer, and the layer must not outlive
    // the window it is attached to (AppKit messages it during window teardown otherwise)
    if (m_childView)
    {
        DestroyMetalLayer(m_childView);
        m_childView = nullptr;
    }
    if (m_layer)
    {
        m_layer->release();
        m_layer = nullptr;
    }
}

void MetalLayerHandle::moveFrom(MetalLayerHandle& other)
{
    m_layer = other.m_layer;
    other.m_layer = nullptr;
    m_childView = other.m_childView;
    other.m_childView = nullptr;
    m_drawable = other.m_drawable;
    other.m_drawable = nullptr;
    m_layerScaleX = other.m_layerScaleX;
    m_layerScaleY = other.m_layerScaleY;
    m_usesSRGB = other.m_usesSRGB;
    other.m_usesSRGB = false;
}

// Applies the layer's pixel format. Must run before AcquireDrawable - a drawable takes the format the
// layer had when it was created, so changing it afterwards leaves the held drawable on the old format
void MetalLayerHandle::SetUsesSRGB(bool usesSRGB)
{
    if (!m_layer || m_usesSRGB == usesSRGB)
        return;
    m_layer->setPixelFormat(usesSRGB ? MTL::PixelFormatBGRA8Unorm_sRGB : MTL::PixelFormatBGRA8Unorm);
    m_usesSRGB = usesSRGB;
}

void MetalLayerHandle::Resize(const Vector2i& size)
{
    if (!m_layer)
        return;
    // re-read the scale: the window may have moved to a display with a different scale factor since
    // the handle was created (MetalView updates its contentsScale in that case, this did not)
    if (m_childView)
        GetMetalViewScale(m_childView, m_layerScaleX, m_layerScaleY);
    m_layer->setDrawableSize(CGSize{(float)size.x * m_layerScaleX, (float)size.y * m_layerScaleY});
}

bool MetalLayerHandle::AcquireDrawable()
{
    if (m_drawable)
        return true;
    if (!m_layer)
        return false;

    m_drawable = m_layer->nextDrawable();
    if (!m_drawable)
    {
        // not Force: a minimised or occluded window fails this every frame, which would flood the log
        cemuLog_logDebug(LogType::Force, "layer {} failed to acquire next drawable", (void*)this);
        return false;
    }

    // nextDrawable returns an autoreleased object - hold an explicit reference so the drawable
    // survives the render thread's per-frame autorelease pool drain (drained after present)
    m_drawable->retain();
    return true;
}

void MetalLayerHandle::PresentDrawable(MTL::CommandBuffer* commandBuffer)
{
    if (!m_drawable)
        return;
    commandBuffer->presentDrawable(m_drawable);
    m_drawable->release();
    m_drawable = nullptr;
}
