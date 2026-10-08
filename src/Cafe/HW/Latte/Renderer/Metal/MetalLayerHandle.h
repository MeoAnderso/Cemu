#pragma once

#include <QuartzCore/QuartzCore.hpp>

#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "util/math/vector2.h"

// Owns the CAMetalLayer of one window plus the drawable currently acquired from it. Both are manual
// (non-ARC) ObjC references released by the destructor.
//
// Copies are deleted and moves transfer ownership, so `handle = MetalLayerHandle(...)` hands the
// references over instead of aliasing them into two owners.
class MetalLayerHandle
{
public:
    MetalLayerHandle() = default;
    MetalLayerHandle(MTL::Device* device, const Vector2i& size, bool mainWindow);

    ~MetalLayerHandle();

    MetalLayerHandle(const MetalLayerHandle&) = delete;
    MetalLayerHandle& operator=(const MetalLayerHandle&) = delete;
    MetalLayerHandle(MetalLayerHandle&& other) noexcept { moveFrom(other); }
    MetalLayerHandle& operator=(MetalLayerHandle&& other) noexcept
    {
        if (this != &other)
        {
            ReleaseAll();
            moveFrom(other);
        }
        return *this;
    }

    void Resize(const Vector2i& size);

    // Applies the layer's pixel format (linear or sRGB). Tracked per layer rather than through one
    // shared flag because the TV and DRC scan buffers can disagree, and the format has to be set
    // before the drawable is acquired for it to take effect
    void SetUsesSRGB(bool usesSRGB);

    bool AcquireDrawable();

    void PresentDrawable(MTL::CommandBuffer* commandBuffer);

    CA::MetalLayer* GetLayer() const { return m_layer; }

    CA::MetalDrawable* GetDrawable() const { return m_drawable; }

private:
    void ReleaseAll();
    void moveFrom(MetalLayerHandle& other);

    CA::MetalLayer* m_layer = nullptr;
    // the NSView created for this layer, owned by its superview; removed in ReleaseAll
    void* m_childView = nullptr;
    // scale the drawable is sized with. Re-read on Resize, so a window moved between displays of
    // different scale factors does not keep the old factor for the rest of the session
    float m_layerScaleX = 1.0f;
    float m_layerScaleY = 1.0f;

    CA::MetalDrawable* m_drawable = nullptr;
    // pixel format currently applied to m_layer; false matches the BGRA8Unorm set at creation
    bool m_usesSRGB = false;
};
