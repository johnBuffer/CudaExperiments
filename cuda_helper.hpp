#pragma once

#include <cstdint>
#include "cuda_runtime.h"
#include "cuda_except.hpp"


template<typename T>
void allocateCudaBuffer(T* ptr, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	cudaError_t cudaStatus(cudaMalloc(static_cast<void**>(&ptr), size * sizeof(T)));
	if (cudaStatus != cudaSuccess) {
		throw CudaException("Cannot allocate device memory", cudaStatus);
	}
}

template<typename T>
void hostToDeviceMemcopy(T* dest, const T* src, uint32_t size)
{
	// Allocate GPU buffers for three vectors (two input, one output).
	cudaError_t cudaStatus(cudaMemcpy(static_cast<void*>(dest), static_cast<void*>(src), size * sizeof(T), cudaMemcpyHostToDevice));
	if (cudaStatus != cudaSuccess) {
		throw CudaException("Cannot memcpy from host to device", cudaStatus);
	}
}


