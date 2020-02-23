#include "device/image_gradient.h"

#include <cassert>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "device/cuda_debug.h"

__global__ void compute_horizontal_gradient_kernel(
		void* grayscale_img_ptr, uint32_t grayscale_image_pitch,
		void* gradient_img_ptr, uint32_t gradient_image_pitch,
		uint32_t image_width, uint32_t image_height) {
	
	const uint32_t image_index_u = blockIdx.x*blockDim.x + threadIdx.x;
	const uint32_t image_index_v = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (image_index_u >= image_width or image_index_v >= image_height) { return; }
	
	uint8_t* grayscale_byte_ptr = reinterpret_cast<uint8_t*>(grayscale_img_ptr);
	grayscale_byte_ptr += grayscale_image_pitch*image_index_v + sizeof(uchar1)*image_index_u;
	uint8_t* gradient_byte_ptr = reinterpret_cast<uint8_t*>(gradient_img_ptr);
	gradient_byte_ptr += gradient_image_pitch*image_index_v + sizeof(short1)*image_index_u;
}

void compute_gradient(const PitchedCUDABuffer& grayscale_image, PitchedCUDABuffer& horizontal_gradient, PitchedCUDABuffer& vertical_gradient) {

}
