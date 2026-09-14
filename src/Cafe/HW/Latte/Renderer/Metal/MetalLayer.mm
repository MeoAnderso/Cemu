#include "Cafe/HW/Latte/Renderer/Metal/MetalLayer.h"

#include "Cafe/HW/Latte/Renderer/MetalView.h"

void* CreateMetalLayer(void* handle, float& scaleX, float& scaleY)
{
	NSView* view = (NSView*)handle;

	MetalView* childView = [[MetalView alloc] initWithFrame:view.bounds];
	childView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
	childView.wantsLayer = YES;

	[view addSubview:childView];

	const NSRect points = [childView frame];
    const NSRect pixels = [childView convertRectToBacking:points];

	scaleX = (float)(pixels.size.width / points.size.width);
    scaleY = (float)(pixels.size.height / points.size.height);

	// These files are built without ARC, so ownership is manual and the caller
	// (MetalLayerHandle) releases the returned layer in its destructor. childView.layer is a +0
	// reference owned by the view, so returning it as-is made that release one too many: it freed
	// the layer while childView still pointed at it, and AppKit then messaged the freed layer when
	// the window was torn down (EXC_BAD_ACCESS in objc_msgSend via -[NSView _setHidden:], on both
	// game stop and app quit). Hand back a +1 instead, and drop our reference to childView now
	// that the superview owns it - previously that +1 was never released either, so the view leaked.
	void* layer = (void*)[childView.layer retain];
	[childView release];
	return layer;
}
