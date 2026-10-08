#include "Cafe/HW/Latte/Renderer/Metal/MetalLayer.h"

#include "Cafe/HW/Latte/Renderer/MetalView.h"

void* CreateMetalLayer(void* handle, float& scaleX, float& scaleY, void** childViewOut)
{
	NSView* view = (NSView*)handle;

	MetalView* childView = [[MetalView alloc] initWithFrame:view.bounds];
	childView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
	childView.wantsLayer = YES;

	[view addSubview:childView];

	// the superview owns the view from here on, but the caller keeps the raw pointer so it can remove
	// it again (DestroyMetalLayer) - it outlives game sessions otherwise
	if (childViewOut)
		*childViewOut = childView;

	GetMetalViewScale(childView, scaleX, scaleY);

	// Non-ARC: the caller (MetalLayerHandle) releases the returned layer in its destructor, so hand
	// back a +1 - childView.layer is a +0 reference the view still uses. Our own reference to
	// childView is dropped now that the superview holds it
	void* layer = (void*)[childView.layer retain];
	[childView release];
	return layer;
}

void DestroyMetalLayer(void* childView)
{
	if (!childView)
		return;
	NSView* view = (NSView*)childView;
	if ([NSThread isMainThread])
	{
		[view removeFromSuperview];
		return;
	}
	// The renderer can be destroyed on the Latte thread (ShutdownTitle joins it before the canvas is
	// destroyed). AppKit is main-thread only, so hand the removal over - asynchronously, because the
	// main thread may be blocked joining this thread
	[view retain];
	dispatch_async(dispatch_get_main_queue(), ^{
		[view removeFromSuperview];
		[view release];
	});
}

void GetMetalViewScale(void* childView, float& scaleX, float& scaleY)
{
	NSView* view = (NSView*)childView;
	const NSRect points = [view frame];
	const NSRect pixels = [view convertRectToBacking:points];
	scaleX = (points.size.width > 0.0) ? (float)(pixels.size.width / points.size.width) : 1.0f;
	scaleY = (points.size.height > 0.0) ? (float)(pixels.size.height / points.size.height) : 1.0f;
}
