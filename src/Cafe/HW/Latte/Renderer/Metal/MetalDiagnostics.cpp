#include "Cafe/HW/Latte/Renderer/Metal/MetalDiagnostics.h"

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace
{
	struct Descriptor
	{
		const char* name;
		// What the user loses when this happens, shown in the first-occurrence line. nullptr marks an
		// informational event (counted for the summary, but not worth a Force-level line of its own).
		const char* consequence;
	};

	// Order must match MetalDiagEvent. A static_assert below keeps them in step.
	constexpr Descriptor kDescriptors[] =
	{
		{ "graphic pack shaders translated", nullptr },

		{ "graphic pack shader not applied: declared resource has no Metal binding",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: texture is a render target on Metal (framebuffer fetch)",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: geometry shader / mesh path",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: manual vertex fetch",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: depth-compare unit declared as a regular sampler",
		  "this graphic pack shader is ignored on Metal - declare it as sampler2DShadow" },
		{ "graphic pack shader not applied: resource maps outside the Metal binding range",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: two resources map to the same Metal binding",
		  "this graphic pack shader is ignored on Metal" },
		{ "graphic pack shader not applied: GLSL front-end failure",
		  "this graphic pack shader is ignored on Metal - the pack's GLSL does not compile" },
		{ "graphic pack shader not applied: SPIR-V to MSL translation failure",
		  "this graphic pack shader is ignored on Metal" },

		{ "render pipeline compilation failed permanently",
		  "draws using this pipeline are skipped for the rest of the session" },

		{ "depth-as-data reads served without a depth mirror",
		  "effects reading depth may be wrong or blank" },
		{ "feedback loop unsupported for this attachment",
		  "effects sampling a texture they also render to may be wrong" },
		{ "sampled mip chain clamped to mip 0 (upper levels are game-unmanaged)",
		  "this texture is sampled at its base level only on Metal, while Vulkan samples the full requested range" },
		{ "sampled mip chain regenerated from mip 0 by the backend",
		  "upper mip levels hold backend-generated content on Metal instead of the game's - a filtered effect using this texture can differ from Vulkan" },
		{ "sampled mip chain served with the game's own upper mip levels", nullptr },
		{ "graphic pack shader outputs cannot be stripped for this pass (translated shader)",
		  "this draw uses the untranslated-output function, which the pipeline may reject - the draw is then missing" },
		{ "surface copies skipped (incompatible dimensions or bytes-per-pixel)",
		  "the copied surface keeps its previous contents" },
	};

	static_assert(std::size(kDescriptors) == (size_t)MetalDiagEvent::COUNT,
		"kDescriptors must have one row per MetalDiagEvent - update both when adding an event");

	std::atomic<uint32> g_eventCounts[(size_t)MetalDiagEvent::COUNT];

	// (event, key) pairs already reported through MetalDiag_CountOncePer. An ordered set
	// rather than a hash: the population is one entry per sampled texture - hundreds in a
	// session - so lookups are O(log n) and there is no collision behavior to reason about
	std::mutex g_firstSightingMutex;
	std::set<std::pair<uint32, uintptr_t>> g_firstSightings;
}

uint32 MetalDiag_Record(MetalDiagEvent event)
{
	const size_t index = (size_t)event;
	if (index >= (size_t)MetalDiagEvent::COUNT)
		return 0;

	const uint32 previousCount = g_eventCounts[index].fetch_add(1, std::memory_order_relaxed);
	if (previousCount == 0 && kDescriptors[index].consequence)
	{
		cemuLog_log(LogType::Force, "MetalDiagnostics: {} - {} (subsequent occurrences are only counted, see the session summary)",
			kDescriptors[index].name, kDescriptors[index].consequence);
	}
	return previousCount;
}

void MetalDiag_Count(MetalDiagEvent event)
{
	MetalDiag_Record(event);
}

const char* MetalDiag_GetName(MetalDiagEvent event)
{
	const size_t index = (size_t)event;
	if (index >= (size_t)MetalDiagEvent::COUNT)
		return "unknown";
	return kDescriptors[index].name;
}

uint32 MetalDiag_GetCount(MetalDiagEvent event)
{
	const size_t index = (size_t)event;
	if (index >= (size_t)MetalDiagEvent::COUNT)
		return 0;
	return g_eventCounts[index].load(std::memory_order_relaxed);
}

bool MetalDiag_MarkFirstSighting(MetalDiagEvent event, uintptr_t key)
{
	std::lock_guard<std::mutex> lock(g_firstSightingMutex);
	return g_firstSightings.emplace((uint32)event, key).second;
}

void MetalDiag_LogSummary()
{
	std::string rows;
	uint32 totalOccurrences = 0;
	uint32 affectingOutput = 0;

	for (size_t i = 0; i < (size_t)MetalDiagEvent::COUNT; i++)
	{
		const uint32 count = g_eventCounts[i].load(std::memory_order_relaxed);
		if (count == 0)
			continue;

		totalOccurrences += count;
		if (kDescriptors[i].consequence)
			affectingOutput += count;
		rows += fmt::format("\n  {:>7}  {}", count, kDescriptors[i].name);
	}

	if (rows.empty())
		return; // nothing happened, stay silent

	cemuLog_log(LogType::Force, "MetalDiagnostics: session summary - {} events recorded, {} of them degrade rendering{}",
		totalOccurrences, affectingOutput, rows);
}
