#pragma once

// Creates the MetalView + CAMetalLayer for the given window handle and adds the view to it.
// childViewOut receives the created NSView (a +0 reference owned by its superview), which the caller
// passes to DestroyMetalLayer when the layer is torn down. The returned layer is a +1 reference.
void* CreateMetalLayer(void* handle, float& scaleX, float& scaleY, void** childViewOut);

// Removes the view created by CreateMetalLayer from its superview and releases it. The superview
// outlives game sessions (the pad window's frame especially), so without this every session that
// opens the DRC window leaves an NSView and a CAMetalLayer behind
void DestroyMetalLayer(void* childView);

// Current backing scale of the given view. Read on resize rather than cached, so moving a window to a
// display with a different scale factor gives a correctly sized drawable
void GetMetalViewScale(void* childView, float& scaleX, float& scaleY);
