#include "device/image_gradient.h"

#include <cassert>
#include <limits>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "device/cuda_debug.h"

__constant__ float scharr_hoizontal[3][3] = {
		{ -3.0f,  0.0f,  3.0f},
		{-10.0f,  0.0f, 10.0f},
		{ -3.0f,  0.0f,  3.0f}
};

__constant__ float scharr_vertical[3][3] = {
		{-3.0f, -10.0f, -3.0f},
		{ 0.0f,   0.0f,  0.0f},
		{ 3.0f,  10.0f,  3.0f}
};

template<bool horizontal>
__global__ void compute_gradient_kernel(
		void* grayscale_img_ptr, uint32_t grayscale_image_pitch,
		void* gradient_img_ptr, uint32_t gradient_image_pitch,
		uint32_t image_width, uint32_t image_height) {
	
	const uint32_t gradient_image_index_u = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t gradient_image_index_v = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (gradient_image_index_u >= image_width or gradient_image_index_v >= image_height) { return; }
	
	uint8_t* grayscale_byte_ptr = reinterpret_cast<uint8_t*>(grayscale_img_ptr);
	float filter_sum = 0;
	
	for (int offset_v = -1; offset_v <= 1; ++offset_v) {
		for (int offset_u = -1; offset_u <= 1; ++offset_u) {
			const int grayscale_image_index_u = min(max(0, gradient_image_index_u + offset_u), image_width - 1);
			const int grayscale_image_index_v = min(max(0, gradient_image_index_v + offset_v), image_height - 1);
			
			const uint8_t pixel_value = grayscale_byte_ptr[grayscale_image_index_v * grayscale_image_pitch + sizeof(uint8_t) * grayscale_image_index_u];
			
			if (horizontal) {
				const float filter_value = scharr_hoizontal[offset_v + 1][offset_u + 1];
				filter_sum += filter_value * static_cast<float>(pixel_value);
			} else {
				const float filter_value = scharr_vertical[offset_v + 1][offset_u + 1];
				filter_sum += filter_value * static_cast<float>(pixel_value);
			}
		}
	}
	
	filter_sum = fminf(fmaxf(filter_sum, static_cast<float>(std::numeric_limits<int16_t>::min())), static_cast<float>(std::numeric_limits<int16_t>::max()));
	
	uint8_t* gradient_byte_ptr = reinterpret_cast<uint8_t*>(gradient_img_ptr);
	gradient_byte_ptr += gradient_image_index_v * gradient_image_pitch + sizeof(int16_t) * gradient_image_index_u;
	
	int16_t* gradient_short_ptr = reinterpret_cast<int16_t*>(gradient_byte_ptr);
	
	*gradient_short_ptr = static_cast<int16_t>(__float2int_rn(filter_sum));
}

void compute_gradient(const PitchedCUDABuffer& grayscale_image, PitchedCUDABuffer& horizontal_gradient, PitchedCUDABuffer& vertical_gradient) {
	// sanity check the input
	assert(grayscale_image.get_element_size_in_bytes() == sizeof(uint8_t));
	assert(horizontal_gradient.get_element_size_in_bytes() == sizeof(int16_t));
	assert(vertical_gradient.get_element_size_in_bytes() == sizeof(int16_t));
	assert(grayscale_image.get_elements_per_row() == horizontal_gradient.get_elements_per_row());
	assert(grayscale_image.get_number_of_rows() == horizontal_gradient.get_number_of_rows());
	assert(grayscale_image.get_elements_per_row() == vertical_gradient.get_elements_per_row());
	assert(grayscale_image.get_number_of_rows() == vertical_gradient.get_number_of_rows());
	
	const uint32_t image_width = grayscale_image.get_elements_per_row();
	const uint32_t image_height = grayscale_image.get_number_of_rows();
	
	const dim3 grayscale_block_dim(32, 32, 1);
	const dim3 grayscale_grid_dim(image_width/grayscale_block_dim.x + (image_width % grayscale_block_dim.x == 0 ? 0 : 1),
	                              image_height/grayscale_block_dim.y + (image_height % grayscale_block_dim.y == 0 ? 0 : 1),
	                              1);
	
	//const dim3 grayscale_grid_dim(1, 1, 1);
	
	compute_gradient_kernel<true><<<grayscale_grid_dim, grayscale_block_dim>>>(
			grayscale_image.get_dev_ptr(), grayscale_image.get_pitch_in_bytes(),
			horizontal_gradient.get_dev_ptr(), horizontal_gradient.get_pitch_in_bytes(),
			image_width, image_height);
	
	compute_gradient_kernel<false><<<grayscale_grid_dim, grayscale_block_dim>>>(
			grayscale_image.get_dev_ptr(), grayscale_image.get_pitch_in_bytes(),
			vertical_gradient.get_dev_ptr(), vertical_gradient.get_pitch_in_bytes(),
			image_width, image_height);
	
	CUDA_SYNC_CHECK();
}
