#include "kernel.cuh"
#include "device_launch_parameters.h"
#include "cuda_helper.cuh"

#include <iostream>
#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int32_t* c, const int32_t* a, const int32_t* b, uint32_t size)
{
	int32_t* dev_a = 0;
	int32_t* dev_b = 0;
	int32_t* dev_c = 0;

	try {
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaSetDeviceExcept(0);
		std::cout << "setDevice" << std::endl;
		// Allocate buffers on GPU
		allocateCudaBuffer(dev_a, size);
		allocateCudaBuffer(dev_b, size);
		allocateCudaBuffer(dev_c, size);
		std::cout << "alloc" << std::endl;
		// Copy input vectors from host memory to GPU buffers.
		hostToDeviceMemcpy(dev_a, a, size);
		hostToDeviceMemcpy(dev_b, b, size);
		std::cout << "cpy" << std::endl;
		// Launch a kernel on the GPU with one thread for each element.
		addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
		std::cout << "addKernel" << std::endl;
		// Check for any errors launching the kernel
		cudaLastErrorToException();
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		cudaWaitForDevice();
		// Copy output vector from GPU buffer to host memory.
		deviceToHostMemcpy(c, dev_c, size);
		// Free buffers
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
	}
	catch (const CudaException& exception) {
		std::cout << exception.what() << std::endl;
		return exception.getErrorCode();
	}

	return cudaSuccess;
}
