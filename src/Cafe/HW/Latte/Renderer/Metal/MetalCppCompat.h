#pragma once

// Shims for Metal APIs newer than the vendored metal-cpp snapshot (dependencies/metal-cpp is
// metal-cpp_macOS14.2_iOS17.2). Add shims here instead of editing the vendored headers.
// NS::Object::sendMessage is protected, so selectors go through objc_msgSend cast to the exact
// signature - what metal-cpp does internally. The availability attributes make the compiler
// require an `if (__builtin_available(...))` guard at every call site.

#include <Metal/Metal.hpp>
#include <objc/message.h>
#include <objc/runtime.h>

namespace MetalCppCompat
{
	// -[MTLBinaryArchive addMeshRenderPipelineFunctionsWithDescriptor:error:]
	__attribute__((availability(macos, introduced = 15.0)))
	inline bool BinaryArchiveAddMeshRenderPipelineFunctions(MTL::BinaryArchive* archive, const MTL::MeshRenderPipelineDescriptor* descriptor, NS::Error** error)
	{
		static const SEL selector = sel_registerName("addMeshRenderPipelineFunctionsWithDescriptor:error:");
		using Fn = bool (*)(const void*, SEL, const void*, NS::Error**);
		return reinterpret_cast<Fn>(&objc_msgSend)(archive, selector, descriptor, error);
	}

	// -[MTLMeshRenderPipelineDescriptor setBinaryArchives:]
	__attribute__((availability(macos, introduced = 15.0)))
	inline void MeshRenderPipelineDescriptorSetBinaryArchives(MTL::MeshRenderPipelineDescriptor* descriptor, const NS::Array* binaryArchives)
	{
		static const SEL selector = sel_registerName("setBinaryArchives:");
		using Fn = void (*)(const void*, SEL, const void*);
		reinterpret_cast<Fn>(&objc_msgSend)(descriptor, selector, binaryArchives);
	}
}
