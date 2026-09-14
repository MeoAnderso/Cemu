#pragma once

// Single source of truth for the layout of the special-uniform section shared by the decompiled
// shaders' uniform blocks:
// - Vulkan GLSL "ufBlock" (LatteDecompilerEmitGLSLHeader.hpp)
// - Metal "SupportBuffer" struct (LatteDecompilerEmitMSLHeader.hpp)
// The Metal SupportBuffer must be byte-identical to the Vulkan ufBlock: the Metal runtime fills it
// using uniformOffsetsVK, and translated graphic pack shaders (MetalShaderTranslator) are authored
// against the Vulkan ufBlock layout. Both emitters call ComputeSupportBufferLayout (which records
// every offset into the requested LatteDecompilerOutputUniformOffsets) and then only translate the
// returned field list into their own declaration syntax and naming. Do not add or reorder fields in
// one emitter without extending this table - a mismatch only corrupts uniforms silently in release
// builds (the debug-only layout parity oracle in LatteDecompilerEmitMSL.cpp verifies the emitted
// offsets of both emitters against each other).

#include <vector>

namespace LatteDecompiler
{
	struct SupportBufferFieldLayout
	{
		enum class Kind
		{
			RemappedUniforms,        // compacted remapped uniform entries (GLSL: uf_remappedVS/PS/GS, MSL: remapped)
			UniformRegisterFile,     // uniform register file (GLSL: uf_uniformRegisterVS/PS/GS, MSL: uniformRegister)
			WindowSpaceToClipSpaceTransform,
			AlphaTestRef,
			PointSize,
			FragCoordScale,
			FragCoordScaleCompatPadding, // invisible padding for pre-2026-06-15 shader replacements (no offset recorded)
			TexScale,                // one per texture unit (texUnit)
			VerticesPerInstance,
			StreamoutBufferBase,     // one per streamout buffer (streamoutIndex)
		};

		Kind kind;
		sint32 texUnit{ -1 };        // TexScale: texture unit
		sint32 streamoutIndex{ -1 }; // StreamoutBufferBase: streamout buffer index
		uint32 arrayCount{ 0 };      // RemappedUniforms / UniformRegisterFile: element count (16 bytes per element)
		sint32 offset{ -1 };         // byte offset within the block
		sint32 size{ 0 };            // byte size
	};

	// Computes the layout of the special-uniform section for the given shader context and fills
	// uniformOffsets with the recorded offsets. vulkanStyle selects the profile: the Vulkan ufBlock
	// and the Metal SupportBuffer use a vec4-sized fragCoordScale (origin stored in zw, 16-byte
	// compat padding); the OpenGL profile uses vec2 (8-byte compat padding).
	// inline, not static: a static function in a header gives every including TU its own copy, so
	// the two emitters could in theory be compiled from mixed revisions of this table
	inline std::vector<SupportBufferFieldLayout> ComputeSupportBufferLayout(LatteDecompilerShaderContext* decompilerContext, bool vulkanStyle, LatteDecompilerOutputUniformOffsets& uniformOffsets)
	{
		using Kind = SupportBufferFieldLayout::Kind;
		std::vector<SupportBufferFieldLayout> fields;

		uint32 uniformCurrentOffset = 0;
		auto shader = decompilerContext->shader;
		auto shaderType = decompilerContext->shader->shaderType;

		// appends a field at the current offset (optionally aligned) and advances the offset
		const auto append = [&](Kind kind, uint32 align, sint32 size) -> SupportBufferFieldLayout*
		{
			if (align > 0)
				uniformCurrentOffset = (uniformCurrentOffset + (align - 1)) & ~(align - 1);
			SupportBufferFieldLayout field{};
			field.kind = kind;
			field.offset = (sint32)uniformCurrentOffset;
			field.size = size;
			fields.push_back(field);
			uniformCurrentOffset += size;
			return &fields.back();
		};

		if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_REMAPPED)
		{
			// uniform registers or buffers are accessed statically with predictable offsets
			// this allows us to remap the used entries into a more compact array
			SupportBufferFieldLayout* field = append(Kind::RemappedUniforms, 0, 16 * (sint32)shader->list_remappedUniformEntries.size());
			field->arrayCount = (uint32)shader->list_remappedUniformEntries.size();
			uniformOffsets.offset_remapped = field->offset;
		}
		else if (shader->uniformMode == LATTE_DECOMPILER_UNIFORM_MODE_FULL_CFILE)
		{
			uint32 cfileSize = decompilerContext->analyzer.uniformRegisterAccessTracker.DetermineSize(decompilerContext->shaderBaseHash, 256);
			// full or partial uniform register file has to be present
			SupportBufferFieldLayout* field = append(Kind::UniformRegisterFile, 0, 16 * cfileSize);
			field->arrayCount = cfileSize;
			uniformOffsets.offset_uniformRegister = field->offset;
			uniformOffsets.count_uniformRegister = cfileSize;
		}

		// special uniforms
		bool hasAnyViewportScaleDisabled =
			!decompilerContext->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_X_SCALE_ENA() ||
			!decompilerContext->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_Y_SCALE_ENA() ||
			!decompilerContext->contextRegistersNew->PA_CL_VTE_CNTL.get_VPORT_Z_SCALE_ENA();
		if (shaderType == LatteConst::ShaderType::Vertex && hasAnyViewportScaleDisabled)
		{
			// aka GX2 special state 0
			uniformOffsets.offset_windowSpaceToClipSpaceTransform = append(Kind::WindowSpaceToClipSpaceTransform, 8, 8)->offset;
		}

		bool alphaTestEnable = decompilerContext->contextRegistersNew->SX_ALPHA_TEST_CONTROL.get_ALPHA_TEST_ENABLE();
		if (shaderType == LatteConst::ShaderType::Pixel && alphaTestEnable)
		{
			uniformOffsets.offset_alphaTestRef = append(Kind::AlphaTestRef, 4, 4)->offset;
		}

		if (decompilerContext->analyzer.outputPointSize && decompilerContext->analyzer.writesPointSize == false)
		{
			if ((shaderType == LatteConst::ShaderType::Vertex && decompilerContext->options->usesGeometryShader == false) ||
				shaderType == LatteConst::ShaderType::Geometry)
			{
				uniformOffsets.offset_pointSize = append(Kind::PointSize, 4, 4)->offset;
			}
		}

		// define fragCoordScale which holds the xy scale for render target resolution vs effective resolution
		bool compatNeedFragCoordScalePadding = false; // 2026-06-15 - fragCoordScale is only emitted when accessed now. To keep compatible with old shader replacements we insert padding if its not the last element
		if (shaderType == LatteConst::ShaderType::Pixel)
		{
			if (decompilerContext->analyzer.hasFragCoordAccess)
			{
				// on Vulkan uf_fragCoordScale additionally stores the render target origin in zw,
				// which is why the MSL profile has to use a float4 here
				const uint32 scaleSize = vulkanStyle ? 16 : 8;
				uniformOffsets.offset_fragCoordScale = append(Kind::FragCoordScale, scaleSize, scaleSize)->offset;
			}
			else
			{
				// omit fragCoordScale
				compatNeedFragCoordScalePadding = true;
			}
		}

		// provide scale factor for every texture that is accessed via texel coordinates (texelFetch)
		for (sint32 t = 0; t < LATTE_NUM_MAX_TEX_UNITS; t++)
		{
			if (decompilerContext->analyzer.texUnitUsesTexelCoordinates.test(t) == false)
				continue;
			if (compatNeedFragCoordScalePadding)
			{
				const uint32 paddingSize = vulkanStyle ? 16 : 8;
				append(Kind::FragCoordScaleCompatPadding, paddingSize, paddingSize);
				compatNeedFragCoordScalePadding = false;
			}
			uniformOffsets.offset_texScale[t] = append(Kind::TexScale, 8, 8)->offset;
			fields.back().texUnit = t;
		}

		// define verticesPerInstance + streamoutBufferBaseX
		// note - the missing parentheses around the && terms are load-bearing (&& binds tighter than
		// ||) and must stay identical between the GLSL and MSL emitters; the MSL mesh path receives
		// its vertex count via a dedicated buffer binding instead (see emitInputs)
		if (decompilerContext->analyzer.useSSBOForStreamout &&
			(shaderType == LatteConst::ShaderType::Vertex && decompilerContext->options->usesGeometryShader == false) ||
			(shaderType == LatteConst::ShaderType::Geometry))
		{
			// note - we dont need to handle compatNeedFragCoordScalePadding here because it's pixel shader only
			uniformOffsets.offset_verticesPerInstance = append(Kind::VerticesPerInstance, 0, 4)->offset;
			for (uint32 i = 0; i < LATTE_NUM_STREAMOUT_BUFFER; i++)
			{
				if (decompilerContext->output->streamoutBufferWriteMask[i])
				{
					SupportBufferFieldLayout* field = append(Kind::StreamoutBufferBase, 0, 4);
					field->streamoutIndex = (sint32)i;
					uniformOffsets.offset_streamoutBufferBase[i] = field->offset;
				}
			}
		}

		uniformOffsets.offset_endOfBlock = uniformCurrentOffset;
		return fields;
	}
}
