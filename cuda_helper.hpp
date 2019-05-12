#pragma once

#include <cstdint>
#include "cuda_runtime.h"

template<typename T>
cudaError_t allocateBuffer(T* ptr, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	return cudaMalloc(static_cast<void**>(&ptr), size * sizeof(T));
}
