#include "cuda_helper.cuh"

void checkCudaStatus(cudaError_t status)
{
	if (status != cudaSuccess) {
		throw CudaException(status);
	}
}

void cudaSetDeviceExcept(uint32_t deviceID)
{
	checkCudaStatus(cudaSetDevice(deviceID));
}

void cudaLastErrorToException()
{
	checkCudaStatus(cudaGetLastError());
}

void cudaWaitForDevice()
{
	checkCudaStatus(cudaDeviceSynchronize());
}
