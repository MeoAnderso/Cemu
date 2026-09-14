#pragma once

#include "Common/precompiled.h"
#include "Cemu/Logging/CemuLogging.h"

// Accounting for renderer features that cannot serve a guest feature on the active backend and fall
// back to something simpler - an unsupported depth format served without a mirror, a graphic pack
// shader that cannot be translated, a pipeline that will never compile.
//
// Why this exists, rather than more logOnce calls: those fire once per call site, so anything that
// happens inside a loop over shaders reports only the first occurrence and the rest are invisible.
// These are also usually the paths that silently change what the user sees, and "silently" is the
// problem - a fallback that is not counted cannot be told apart from a pack shader that simply does
// not apply.
//
// Deliberately NOT behind a compile-time macro. A macro that only exists in Debug builds is dead in
// the builds users actually run and report from; LogType categories are toggled at runtime under
// Debug -> Logging, so the verbose detail works everywhere.
//
// Hot-path cost is one relaxed atomic increment. Messages are formatted only when they will actually
// be emitted (cemuLog_log early-outs on a disabled category).
//
// To add an event: append it before COUNT, add a matching row to kDescriptors in the .cpp, and add
// an entry to the Debug -> Logging menu if it should be user-toggleable.

enum class MetalDiagEvent : uint8
{
	// Graphic pack shader translation (MetalShaderTranslator). "Fallback" means the pack shader was
	// not applied and the decompiled shader is used instead, so the pack has no effect on this draw.
	PackShaderTranslated,
	PackShaderFallbackResource,      // a resource the shader declares has no Metal binding
	PackShaderFallbackFramebuffer,   // the texture unit is a render target on Metal (framebuffer fetch)
	PackShaderFallbackMeshPath,      // geometry shader / mesh path
	PackShaderFallbackVertexFetch,   // manual vertex fetch
	PackShaderFallbackDepthCompare,  // depth-compare unit declared as a regular sampler
	PackShaderFallbackBinding,       // resource maps outside the Metal binding range
	PackShaderFallbackCollision,     // two resources remap to the same Metal index
	PackShaderFallbackFrontend,      // GLSL -> SPIR-V failed
	PackShaderFallbackCross,         // SPIR-V -> MSL failed

	// Pipeline state
	PipelineCompileFailedPermanent,  // compilation cannot succeed, draws are skipped from now on

	// Draw availability. The mesh path is Metal's only geometry-shader emulation, and RECTS rides
	// the same path (UseGeometryShader is hasGeometryShader || UseRectEmulation), so a device with
	// no mesh shader support has no route for these draws at all and they are dropped.
	GeometryShaderUnsupported,       // geometry shader / RECTS draw skipped - no mesh shader support

	// Binding limits. Metal's sampler table is the smaller of the two argument tables (16 against
	// 31 textures) and Latte has 18 texture units. A shader declaring more than 16 textures - and
	// so more than 16 sampler arguments - has no valid encoding: Metal rejects the over-range
	// declaration, the pipeline fails, and the draws using it are skipped.
	SamplerBindingOverflow,          // shader declares more textures/samplers than Metal allows

	// Texture and surface handling
	// The readback blit can only serve a 2D slice of a level the resource actually has, so a
	// request outside that is served as the nearest expressible one. Reported because the failure
	// mode is stale or wrong pixels the game goes on to use - a compiled-out assert used to hide it
	ReadbackShapeUnsupported,        // readback of a 3D texture, or of a mip the texture lacks
	DepthMirrorUnavailable,          // depth-as-data read has no mirror on this backend
	FeedbackLoopUnsupported,         // texture sampled while attached, no shadow copy possible
	// The next three are Metal's handling of sampled mip chains. The first two are the backend's
	// deliberate deviations from Vulkan, and the only two places where Metal hands a draw a mip
	// range the game did not ask for: Vulkan binds the view as requested, while Metal clamps or
	// regenerates when it cannot serve the chain as written. The third is the report that it could,
	// and is what to look for when auditing why a chain was or was not treated as unmanaged. If a
	// filtered effect (DoF, bloom) differs between backends, one of the first two fired on it.
	SampledMipChainClamped,          // sampled view clamped to mip 0
	SampledMipChainRegenerated,      // upper mips regenerated from mip 0 by the backend
	SampledMipChainTrusted,          // upper mips served as written (no clamp, no regeneration)
	// A graphic-pack shader (translated, SPIRV-Cross output) has outputs the active pass has no
	// attachment for, and the stripper cannot rewrite it - so the draw uses the unstripped function,
	// which Metal may reject at pipeline creation
	OrphanOutputStripUnsupported,
	SurfaceCopySkipped,              // a surface copy was requested but could not be performed

	COUNT
};

// Records one occurrence and returns the count observed *before* the increment (0 == first).
// The first occurrence emits a single Force-level line naming the consequence - what the user loses.
// Later occurrences are only counted, so a hot path never turns into log spam.
uint32 MetalDiag_Record(MetalDiagEvent event);

const char* MetalDiag_GetName(MetalDiagEvent event);

// Occurrences recorded for one event since startup. Lets the OSD panel show the census live, so the
// shutdown summary stops being the only way to see it - that one is lost to anything but a graceful
// quit, since Cemu's SIGTERM handler is a bare _Exit.
uint32 MetalDiag_GetCount(MetalDiagEvent event);

// Records one occurrence, attaching `detail` (shader hash, texture address, format...) to the
// verbose channel. Prefer this overload where something identifying is at hand: the summary says
// which class of fallback is happening, the detail says for which shader or texture.
//
// This overload logs *every* occurrence, including the first, so a call site reached once per draw
// emits one line per draw. Such a call site has to latch its own detail - use
// MetalDiag_CountOncePer, the one idiom for that. Counts accumulate either way, so the summary
// stays exact.
template <typename... TArgs>
void MetalDiag_Count(MetalDiagEvent event, fmt::format_string<TArgs...> detail, TArgs&&... args)
{
	MetalDiag_Record(event);
	if (cemuLog_isLoggingEnabled(LogType::MetalBackend))
		cemuLog_log(LogType::MetalBackend, "{}: {}", MetalDiag_GetName(event), fmt::format(detail, std::forward<TArgs>(args)...));
}

// Same, without detail.
void MetalDiag_Count(MetalDiagEvent event);

// True the first time this (event, key) pair is seen, false on every later call. Thread-safe.
//
// `key` identifies the OBJECT the report is about, and callers pass a stable identity - for a
// texture, its guest physical address - rather than its host pointer. A LatteTextureMtl* was the
// obvious choice and is wrong: textures are evicted and freed continuously (LatteTextureCache.cpp
// deletes up to 10 per frame) and the allocator hands the same address back, so a brand new
// texture inherits the dead one's "already reported" state and never logs its own detail line
// while the summary count still increments - the diagnostic going quietly dark exactly where it
// matters. A uintptr_t also avoids ordering unrelated pointers with the built-in operator<.
bool MetalDiag_MarkFirstSighting(MetalDiagEvent event, uintptr_t key);

// Records one occurrence, logging `detail` only the first time this (event, key) pair is seen.
//
// This is the one idiom for "report this once per object". The latch is per key rather than per
// event: a global per-event cap gets spent by the first hot chain, leaving the rest of the
// session unreported. `key` is an identity - the texture object - not a value; two textures with
// equal contents are still two keys. Counts accumulate on every call, so the session summary
// stays exact.
template <typename... TArgs>
void MetalDiag_CountOncePer(MetalDiagEvent event, uintptr_t key,
	fmt::format_string<TArgs...> detail, TArgs&&... args)
{
	MetalDiag_Record(event);
	if (!MetalDiag_MarkFirstSighting(event, key))
		return;
	if (cemuLog_isLoggingEnabled(LogType::MetalBackend))
		cemuLog_log(LogType::MetalBackend, "{}: {}", MetalDiag_GetName(event),
			fmt::format(detail, std::forward<TArgs>(args)...));
}

// Same, but counted once per object too: later sightings of the same key neither log nor count.
//
// Use this where the question is per object rather than per occurrence: "how many times was a
// chain clamped" is a per-draw question (MetalDiag_CountOncePer), "how many chains were
// served as written" is a per-chain one
template <typename... TArgs>
void MetalDiag_CountFirstSighting(MetalDiagEvent event, uintptr_t key,
	fmt::format_string<TArgs...> detail, TArgs&&... args)
{
	if (!MetalDiag_MarkFirstSighting(event, key))
		return;
	MetalDiag_Record(event);
	if (cemuLog_isLoggingEnabled(LogType::MetalBackend))
		cemuLog_log(LogType::MetalBackend, "{}: {}", MetalDiag_GetName(event),
			fmt::format(detail, std::forward<TArgs>(args)...));
}

// Logs the session summary at shutdown. Prints non-zero rows only, and stays silent entirely when
// nothing was recorded, so an uneventful session does not append noise to the log.
void MetalDiag_LogSummary();
