#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "imgInversionCUDA.h"

__global__ void Inversion_CUDA(unsigned char* Image, int Channels);

void Image_Inversion_CUDA(unsigned char* Input_Image, int Height, int Width, int Channels) {
	unsigned char* Dev_Input_Image = NULL;

	cudaMalloc((void**)&Dev_Input_Image, Height * Width * Channels);

	// copy CPU data to GPU
	cudaMemcpy(Dev_Input_Image, Input_Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Inversion_CUDA <<< Grid_Image, 1 >>> (Dev_Input_Image, Channels);

	cudaMemcpy(Input_Image, Dev_Input_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	// free gpu mem
	cudaFree(Dev_Input_Image);
}

__global__ void Inversion_CUDA(unsigned char* Image, int Channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int idx = (x + y * gridDim.x) * Channels;

	for (int i = 0; i < Channels; i++) {
		Image[idx + i] = 255 - Image[idx + i];
	}
}