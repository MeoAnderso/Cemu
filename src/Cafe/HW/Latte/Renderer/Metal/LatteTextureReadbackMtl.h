#pragma once

#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"
#include "Cafe/HW/Latte/Core/LatteTextureReadbackInfo.h"

class LatteTextureReadbackInfoMtl : public LatteTextureReadbackInfo
{
public:
	// m_firstMip is the mip level the backend transfers, and the consumer writes the result back at
	// that same level - so it has to come from the view, not default to 0. Vulkan passes it the same
	// way (TextureReadbackVk.cpp); Metal used to default it, which told the consumer every readback
	// was mip 0 no matter which mip the view asked for
	LatteTextureReadbackInfoMtl(class MetalRenderer* mtlRenderer, LatteTextureView* textureView, uint32 bufferOffset) : LatteTextureReadbackInfo(textureView, textureView->firstMip), m_mtlr{mtlRenderer}, m_bufferOffset{bufferOffset} {}
	~LatteTextureReadbackInfoMtl();

	void StartTransfer() override;

	bool IsFinished() override;
	void ForceFinish() override;

	uint8* GetData() override;

private:
	class MetalRenderer* m_mtlr;

	MTL::CommandBuffer* m_commandBuffer = nullptr;

	uint32 m_bufferOffset = 0;
};
