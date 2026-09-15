#pragma once

#include <Metal/Metal.hpp>
#include <atomic>
#include <cstdint>
#include <unordered_map>

#include "Cafe/HW/Latte/Core/LatteTexture.h"
#include "Cafe/HW/Latte/Renderer/Metal/LatteTextureViewMtl.h"
#include "HW/Latte/ISA/LatteReg.h"
#include "util/ChunkedHeap/ChunkedHeap.h"

class LatteTextureMtl : public LatteTexture
{
public:
	LatteTextureMtl(class MetalRenderer* mtlRenderer, Latte::E_DIM dim, MPTR physAddress, MPTR physMipAddress, Latte::E_GX2SURFFMT format, uint32 width, uint32 height, uint32 depth, uint32 pitch, uint32 mipLevels,
		uint32 swizzle, Latte::E_HWTILEMODE tileMode, bool isDepth);
	~LatteTextureMtl();

	MTL::Texture* GetTexture() const {
	    return m_texture;
	}

	// Color-format copy of the depth data (depth value stored in the red channel), used to sample depth
	// textures as regular textures. Metal requires depth2d declarations for depth-format textures, which
	// the decompiled MSL never emits (the same texture unit may sample a color or a depth texture
	// depending on what the game binds), so depth-as-data sampling goes through this copy instead.
	// Mirrors GL/Vulkan semantics, where sampling a depth texture via a regular sampler returns the
	// depth value in the red channel. Managed by MetalRenderer::depthCopy_refreshColorCopy
	MTL::Texture* GetDepthColorCopy() const {
		return m_depthColorCopy;
	}

	// updateEventCounter tracks the data version the copy was taken at (the depth texture's
	// lastWriteEventCounter), so MetalRenderer::depthCopy_refreshColorCopy can detect writes
	// that happen mid-frame (e.g. shadow maps being rewritten between sampling draws)
	void SetDepthColorCopy(MTL::Texture* colorCopy, uint64 updateEventCounter) {
		if (m_depthColorCopy && m_depthColorCopy != colorCopy)
		{
			// the mirror is being replaced: cached sample views built from it are invalid
			for (auto& [key, view] : m_depthMirrorSampleViews)
				view->release();
			m_depthMirrorSampleViews.clear();
			m_depthColorCopy->release();
		}
		m_depthColorCopy = colorCopy;
		m_depthColorCopyUpdateCounter = updateEventCounter;
	}

	uint64 GetDepthColorCopyUpdateCounter() const {
		return m_depthColorCopyUpdateCounter;
	}

	// Cached sample view of the depth color-copy mirror for the given bound view + sampler swizzle
	// (created + released per draw before - one driver allocation per draw). The cache owns the
	// returned view (do not release); it is invalidated when the mirror is replaced
	MTL::Texture* GetDepthMirrorSampleView(LatteTextureView* textureView, uint32 gpuSamplerSwizzle);

	// rate limiting for the refresh (see depthCopy_refreshColorCopy): refreshes within the same
	// frame are capped so a game alternating depth writes and depth-as-data sampling every draw
	// cannot trigger a side command buffer per draw. Mid-frame rewrites within the cap are still
	// refreshed (keeps shadow clouds coherent)
	void TrackDepthColorCopyRefresh(uint32 frameCounter) {
		if (m_depthColorCopyLastRefreshFrame != frameCounter)
		{
			m_depthColorCopyLastRefreshFrame = frameCounter;
			m_depthColorCopyRefreshesThisFrame = 0;
		}
		m_depthColorCopyRefreshesThisFrame++;
	}

	bool IsDepthColorCopyRefreshCapped(uint32 frameCounter, uint32 cap) const {
		return m_depthColorCopyLastRefreshFrame == frameCounter && m_depthColorCopyRefreshesThisFrame >= cap;
	}

	// Mip-content tracking: bit i is set when a render pass or copy wrote mip level i. Games
	// like SM3DW sample effect buffers through their full mip chain with forced LODs while only
	// mip 0 ever receives content from the game. MetalRenderer::EnsureSampledMipContentValid
	// uses the mask to detect these never-managed chains and regenerate the upper levels from
	// the fresh render-written mip0
	uint32 GetRenderWrittenMipMask() const {
		return m_renderWrittenMipMask;
	}

	void MarkRenderMipsWritten(uint32 firstMip, uint32 numMips) {
		if (firstMip >= 32)
			return;
		m_renderWrittenMipMask |= (numMips >= 32 ? 0xFFFFFFFFu : ((1u << numMips) - 1u) << firstMip);
	}

	// Mips written by game render passes only. Cemu's own copy paths (alias preservation,
	// surface copies) also mark the combined mask above, but their byte-offset-matched writes
	// land old incarnation content into the new chain's upper mips (e.g. a 2x-zoomed crop into
	// mip1) - content the game never samples intentionally. The regeneration gate must ignore
	// those, or one preservation copy permanently disables the chain's regeneration
	uint32 GetPassWrittenMipMask() const {
		return m_passWrittenMipMask;
	}

	void MarkPassMipsWritten(uint32 firstMip, uint32 numMips) {
		if (firstMip >= 32)
			return;
		m_passWrittenMipMask |= (numMips >= 32 ? 0xFFFFFFFFu : ((1u << numMips) - 1u) << firstMip);
		// a render pass supersedes whatever a copy or a clear put into these levels, so their
		// provenance is dropped here (see MarkCopyMipsWritten / MarkLevelCleared)
		const uint32 range = MipRangeMask(firstMip, numMips);
		m_clearedMipMask &= ~range;
		m_matchedCopyMipMask &= ~range;
		m_resampledCopyMipMask &= ~range;
		m_partialCopyMipMask &= ~range;
	}

	// What the last copy into a mip level did to that level. Three outcomes, because they are the
	// three things a copy can be, and the question the sample-time workarounds ask - is this
	// level's content the game's own? - turns on which one it was:
	enum class CopyProvenance : uint8
	{
		// the copy covered the whole destination level with a whole source level of the same size:
		// the level's content was carried over 1:1
		Matched,
		// the copy covered the whole destination level from a source level of a different size:
		// the level was written deliberately and completely, but resampled (a game building its own
		// mip chain by copying a larger level into a smaller one does this)
		Resampled,
		// the copy covered only part of the destination level (an offset sub-rect, or a region
		// sized for a differently scaled source): the rest of the level keeps whatever was there
		// before, so what the level holds is a crop pasted into other content
		Partial
	};

	// Copy provenance per mip level. The written masks cannot answer whether a chain's content is
	// the game's own - a chain whose levels were faithfully preserved, one built by deliberate
	// resampling, and one full of pasted-in crops all look identically "copy-written" to them. What
	// differs is what each copy did, which is what these masks record. Each level is owned by at
	// most one of them; the last write wins, so a partial copy invalidates an earlier complete one.
	uint32 GetMatchedCopyMipMask() const {
		return m_matchedCopyMipMask;
	}

	uint32 GetResampledCopyMipMask() const {
		return m_resampledCopyMipMask;
	}

	uint32 GetPartialCopyMipMask() const {
		return m_partialCopyMipMask;
	}

	// Levels the game filled with a clear. Not a copy, so it has no CopyProvenance: a clear writes
	// the whole level with a value the game chose, which puts it in the same standing as a render
	// pass - the level holds what the game wants there. It needs recording because without it a
	// cleared level looks like one nobody ever wrote, and a chain the game clears and then samples
	// through a multi-mip view gets clamped to mip 0
	uint32 GetClearedMipMask() const {
		return m_clearedMipMask;
	}

	// Records that a clear wrote level `mip` in full. The last write wins, exactly as for copies: a
	// clear supersedes any copy provenance for that level, and MarkCopyMipsWritten/MarkPassMipsWritten
	// supersede a clear. A level is owned by at most one provenance
	void MarkLevelCleared(uint32 mip) {
		const uint32 range = MipRangeMask(mip, 1);
		if (range == 0)
			return;
		m_clearedMipMask = (m_clearedMipMask & ~range) | range;
		m_matchedCopyMipMask &= ~range;
		m_resampledCopyMipMask &= ~range;
		m_partialCopyMipMask &= ~range;
	}

	// Records what the copy that wrote the levels [firstMip, firstMip + numMips) did. Ranges that do
	// not fit the 32-bit mask are dropped; RangeContentIsTrustworthy treats an unrepresentable
	// range as untrustworthy, so dropping them cannot make a chain look better than it is
	void MarkCopyMipsWritten(uint32 firstMip, uint32 numMips, CopyProvenance provenance) {
		const uint32 range = MipRangeMask(firstMip, numMips);
		if (range == 0)
			return;
		m_clearedMipMask &= ~range;
		m_matchedCopyMipMask = (m_matchedCopyMipMask & ~range) | (provenance == CopyProvenance::Matched ? range : 0u);
		m_resampledCopyMipMask = (m_resampledCopyMipMask & ~range) | (provenance == CopyProvenance::Resampled ? range : 0u);
		m_partialCopyMipMask = (m_partialCopyMipMask & ~range) | (provenance == CopyProvenance::Partial ? range : 0u);
	}

	// Bitmask of the mip levels [firstMip, firstMip + numMips), or 0 when the range is empty or
	// does not fit in 32 levels
	static uint32 MipRangeMask(uint32 firstMip, uint32 numMips) {
		if (numMips == 0 || firstMip >= 32 || (firstMip + numMips) > 32)
			return 0;
		return ((1u << numMips) - 1u) << firstMip;
	}

	// Whether mip level `mip` holds content this chain's own writes produced: a render pass wrote it,
	// or a copy wrote the WHOLE level. Partial is the only copy provenance that leaves part of the
	// level holding whatever was there before, so it is the only one rejected.
	//
	// Resampled is accepted. It means a whole level was written from a source level of a different
	// size, which happens two ways: the game building its own mip chain (a blur pyramid does exactly
	// this - 512x256 -> 256x128 -> 128x64 - by copying each level down into the next), or a previous,
	// differently scaled incarnation of the surface being pasted in. The earlier rule rejected both,
	// on the grounds that they cannot be told apart from the copy alone. That was too strict, and the
	// strictness is not symmetric: a falsely-trusted level serves the game's content, which is what
	// Vulkan serves, while a falsely-rejected one arms the sample-time clamp and the regeneration and
	// hands the draw a different mip than the game asked for. The trust rule is answering "is this
	// the game's content", and a full-level resample is. GetResampledCopyMipMask is still recorded so
	// the log can tell the two apart.
	bool LevelContentIsTrustworthy(uint32 mip) const {
		if (mip >= 32)
			return false;
		return ((m_passWrittenMipMask | m_matchedCopyMipMask | m_resampledCopyMipMask | m_clearedMipMask) & (1u << mip)) != 0;
	}

	// Whether the whole sampled range [firstMip, firstMip + numMips) can be served exactly as the
	// game asked for it. This is the question both sample-time workarounds approximate, and the one
	// the written masks cannot answer on their own.
	//
	// Every level in the range must be trustworthy, including firstMip: a view that samples a
	// single upper level directly has no upper levels of its own, yet the level it reads may still
	// be one the chain never wrote.
	//
	// Callers additionally require the chain to have GPU-written content at mip 0 - the level they
	// fall back to, and the source regeneration generates from - which also excludes CPU-uploaded
	// asset textures, whose presumably valid chains must never be treated as unmanaged
	bool RangeContentIsTrustworthy(uint32 firstMip, uint32 numMips) const {
		if (numMips == 0 || MipRangeMask(firstMip, numMips) == 0)
			return false; // unrepresentable range - never trust it
		for (uint32 mip = firstMip; mip < firstMip + numMips; mip++)
		{
			if (!LevelContentIsTrustworthy(mip))
				return false;
		}
		return true;
	}

	// Shape of the most recent copy into this texture, recorded for the diagnostics: which levels it
	// connected and at what sizes, and whether it stayed inside this texture (the game building its
	// own chain) or brought content in from another texture (a preservation copy carrying a
	// previous incarnation's content over). This is what explains a level's verdict in the log
	struct LastCopyInfo
	{
		uint32 srcMip{};
		uint32 dstMip{};
		uint32 srcWidth{};
		uint32 srcHeight{};
		uint32 dstWidth{};
		uint32 dstHeight{};
		uint32 copyWidth{};
		uint32 copyHeight{};
		bool sameTexture{};
	};

	void SetLastCopyInfo(const LastCopyInfo& info) {
		m_lastCopy = info;
	}

	const LastCopyInfo& GetLastCopyInfo() const {
		return m_lastCopy;
	}

	// The trust/clamp/regeneration diagnostics latch through MetalDiag_CountOncePer /
	// MetalDiag_CountFirstSighting, keyed per texture - see MetalDiagnostics.h for the policy

	// event counter of the lastWriteEventCounter value at which generateMipmaps last ran for
	// this texture (0 = never). EnsuredMipContentIsCurrent(lastWriteEventCounter) is false when
	// the mip0 content changed since the last regeneration
	uint64 GetMipGenerationEventCounter() const {
		return m_mipGenerationEventCounter;
	}

	void SetMipGenerationEventCounter(uint64 eventCounter) {
		m_mipGenerationEventCounter = eventCounter;
	}

	void AllocateOnHost() override;

	bool m_isAlternateFormat{}; // true if Metal pixel format does not 1:1 match Latte format

protected:
	LatteTextureView* CreateView(Latte::E_DIM dim, Latte::E_GX2SURFFMT format, sint32 firstMip, sint32 mipCount, sint32 firstSlice, sint32 sliceCount) override;

private:
	class MetalRenderer* m_mtlr;

	MTL::Texture* m_texture;

	MTL::Texture* m_depthColorCopy{ nullptr };
	uint64 m_depthColorCopyUpdateCounter{ 0 };
	// owned sample views of m_depthColorCopy - see GetDepthMirrorSampleView
	std::unordered_map<LatteMtlSampleViewKey, MTL::Texture*> m_depthMirrorSampleViews;
	uint32 m_depthColorCopyLastRefreshFrame{ 0xFFFFFFFF };
	uint32 m_depthColorCopyRefreshesThisFrame{ 0 };

	uint32 m_renderWrittenMipMask{ 0 };
	uint32 m_passWrittenMipMask{ 0 };
	// copy provenance, see MarkCopyMipsWritten / MarkLevelCleared. Mutually exclusive per level
	uint32 m_matchedCopyMipMask{ 0 };
	uint32 m_resampledCopyMipMask{ 0 };
	uint32 m_partialCopyMipMask{ 0 };
	uint32 m_clearedMipMask{ 0 };
	LastCopyInfo m_lastCopy;
	uint64 m_mipGenerationEventCounter{ 0 };
};
