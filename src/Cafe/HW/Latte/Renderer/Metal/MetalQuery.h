#pragma once

#include "Cafe/HW/Latte/Core/LatteQueryObject.h"

#include "Cafe/HW/Latte/Renderer/Metal/MetalCommon.h"

class LatteQueryObjectMtl : public LatteQueryObject
{
public:
	LatteQueryObjectMtl(class MetalRenderer* mtlRenderer) : m_mtlr{mtlRenderer} {}
	~LatteQueryObjectMtl();

	bool getResult(uint64& numSamplesPassed) override;
	void begin() override;
	void end() override;

private:
	class MetalRenderer* m_mtlr;

	MetalQueryRange m_range = {INVALID_UINT32, INVALID_UINT32};
	// TODO: make this a list of command buffers?
	MTL::CommandBuffer* m_commandBuffer = nullptr;
	// whether end() ran: a result polled before the query was ended is not ready, rather than an
	// empty range that reports zero samples (Vulkan returns false until the query ended too)
	bool m_ended = false;
	// slots this query's draws consumed. The pool is OCCLUSION_QUERY_POOL_SIZE entries and a slot is
	// consumed per counted draw, so a query spanning more of them than the pool has had its earlier
	// slots overwritten by its own later draws - the sum below then double-counts and cannot be
	// recovered. Recorded so the case is reported instead of silently returning a wrong number
	uint32 m_slotsConsumed = 0;
	uint64 m_drawCountAtBegin = 0;
};
