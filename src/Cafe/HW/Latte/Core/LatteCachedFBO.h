#pragma once

#include "Cafe/HW/Latte/Core/LatteTexture.h"
#include "util/math/vector2.h"

class LatteCachedFBO
{
public:
	LatteCachedFBO(uint64 key);

	uint32 calculateNumColorBuffers()
	{
		uint32 n = 0;
		for (sint32 i = 0; i < 8; i++)
			if (colorBuffer[i].texture)
				n++;
		return n;
	}

	bool hasDepthBuffer()
	{
		return depthBuffer.texture;
	}

	std::vector<LatteTexture*>& GetTextures()
	{
		return m_referencedTextures;
	}

	virtual ~LatteCachedFBO() {};

private:
	void calculateEffectiveRenderAreaSize()
	{
		Vector2i rtEffectiveSize;
		rtEffectiveSize.x = 0;
		rtEffectiveSize.y = 0;
		sint32 numViews = 0;
		// derive extent from color buffers
		for (sint32 i = 0; i < 8; i++)
		{
			if(colorBuffer[i].texture == nullptr)
				continue;
			sint32 effectiveWidth, effectiveHeight;
			colorBuffer[i].texture->baseTexture->GetEffectiveSize(effectiveWidth, effectiveHeight, colorBuffer[i].texture->firstMip);
			if (rtEffectiveSize.x == 0 && rtEffectiveSize.y == 0)
			{
				rtEffectiveSize.x = effectiveWidth;
				rtEffectiveSize.y = effectiveHeight;
			}
			if (effectiveWidth < rtEffectiveSize.x)
			{
				cemuLog_logDebug(LogType::Force, "Framebuffer has color texture with smaller effective width ({} -> {})", rtEffectiveSize.x, effectiveWidth);
				rtEffectiveSize.x = effectiveWidth;
			}
			if (effectiveHeight < rtEffectiveSize.y)
			{
				cemuLog_logDebug(LogType::Force, "Framebuffer has color texture with smaller effective height ({} -> {})", rtEffectiveSize.y, effectiveHeight);
				rtEffectiveSize.y = effectiveHeight;
			}
			numViews++;
		}
		// derive extent from depth buffer
		if (depthBuffer.texture)
		{
			sint32 effectiveWidth, effectiveHeight;
			depthBuffer.texture->baseTexture->GetEffectiveSize(effectiveWidth, effectiveHeight, depthBuffer.texture->firstMip);
			if (rtEffectiveSize.x == 0 && rtEffectiveSize.y == 0)
			{
				rtEffectiveSize.x = effectiveWidth;
				rtEffectiveSize.y = effectiveHeight;
			}
			if (effectiveWidth < rtEffectiveSize.x)
			{
				cemuLog_logDebug(LogType::Force, "Framebuffer has depth texture with smaller effective width ({} -> {})", rtEffectiveSize.x, effectiveWidth);
				rtEffectiveSize.x = effectiveWidth;
			}
			if (effectiveHeight < rtEffectiveSize.y)
			{
				cemuLog_logDebug(LogType::Force, "Framebuffer has depth texture with smaller effective height ({} -> {})", rtEffectiveSize.y, effectiveHeight);
				rtEffectiveSize.y = effectiveHeight;
			}
			numViews++;
		}
		if (numViews == 0)
		{
			// empty FBO
			m_size = Vector2i(32, 32);
			return;
		}
		cemu_assert_debug(rtEffectiveSize.x != 0);
		cemu_assert_debug(rtEffectiveSize.y != 0);
		m_size = rtEffectiveSize;
	}

public:
	uint64 key;
	Vector2i m_size;

	struct
	{
		LatteTextureView* texture{};
	}colorBuffer[8]{};
	struct
	{
		LatteTextureView* texture{};
		bool hasStencil{};
	}depthBuffer{};
	uint32 drawBuffersMask{};

	std::vector<LatteTexture*> m_referencedTextures; // color and depth views combined
};

class LatteMRT
{
public:
	static void NotifyTextureDeletion(LatteTexture* texture);

	// Address/dimension decoding of one color buffer register set, with no side effects: no view is
	// looked up or created. GetColorAttachmentTexture uses it, and so does the Metal backend's
	// read-only framebuffer-fetch check (LatteShader.cpp) - the two used to carry separate copies of
	// this derivation, which could drift apart silently
	struct ColorBufferViewDecode
	{
		MPTR physAddress;
		uint32 width;
		uint32 height;
		uint32 pitch;
		uint32 firstSlice;
		uint32 numSlices;
		uint32 swizzle;
		Latte::E_HWTILEMODE tileMode;
		Latte::E_GX2SURFFMT format;
	};
	static ColorBufferViewDecode DecodeColorBufferView(uint32 index, const uint32* contextRegisters, const LatteContextRegister& lcr);

	// GPU state
	static LatteTextureView* GetColorAttachmentTexture(uint32 index, bool createNew, bool checkForTextureChanges);
	static uint8 GetActiveColorBufferMask(const LatteDecompilerShader* pixelShader, const struct LatteContextRegister& lcr);
	static bool GetActiveDepthBufferMask(const struct LatteContextRegister& lcr);
	static Latte::E_GX2SURFFMT GetColorBufferFormat(const uint32 index, const LatteContextRegister& lcr);
	static Latte::E_GX2SURFFMT GetDepthBufferFormat(const struct LatteContextRegister& lcr);

	// FBO state management
	static void SetColorAttachment(uint32 index, LatteTextureView* view);
	static void SetDepthAndStencilAttachment(LatteTextureView* view, bool hasStencil);
	static LatteTextureView* GetColorAttachment(uint32 index);
	static LatteTextureView* GetDepthAttachment();

	static void ApplyCurrentState();

	static bool UpdateCurrentFBO(); // update FBO with info from current GPU state

	// helper functions
	static void BindColorBufferOnly(LatteTextureView* view);
	static void BindDepthBufferOnly(LatteTextureView* view);

	static void GetCurrentFragCoordScale(float* coordScale);
	static void GetVirtualViewportDimensions(sint32& width, sint32& height); // returns the width and height of the current GPU viewport (unaffected by graphic pack rules)

	// todo - move this into FBO destructor (?)
	static void DeleteCachedFBO(LatteCachedFBO* cfbo);

private:
	static LatteCachedFBO* CreateCachedFBO(uint64 key);
};
