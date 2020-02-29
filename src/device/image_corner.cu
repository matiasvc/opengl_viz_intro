#include "device/image_corner.h"

#include <cassert>

#include <Eigen/Core>

#include "device/cuda_debug.h"

__constant__ float gaussian_kernel[5][5] = {
		{1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f},
		{4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f},
		{7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f},
		{4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f},
		{1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f},
};

__global__ void compute_corner_response_kernel(
		void* horizontal_gradient_ptr, uint32_t horizontal_gradient_pitch,
		void* vertical_gradient_ptr, uint32_t vertical_gradient_pitch,
		void* corner_response_ptr, uint32_t corner_response_pitch,
		uint32_t image_width, uint32_t image_height) {
	
	const uint32_t image_index_u = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t image_index_v = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (image_index_u >= image_width or image_index_v >= image_height) { return; }
	
	float dxdx = 0.0f;
	float dydy = 0.0f;
	float dxdy = 0.0f;
	
	for (int offset_v = -2; offset_v <= 2; ++offset_v) {
		for (int offset_u = -2; offset_u <= 2; ++offset_u) {
			const int patch_image_index_u = min(max(0, image_index_u + offset_u), image_width - 1);
			const int patch_image_index_v = min(max(0, image_index_v + offset_v), image_height - 1);
			
			uint8_t* horizontal_gradient_byte_ptr = reinterpret_cast<uint8_t*>(horizontal_gradient_ptr);
			horizontal_gradient_byte_ptr += horizontal_gradient_pitch*patch_image_index_v + sizeof(int16_t)*patch_image_index_u;
			
			uint8_t* vertical_gradient_byte_ptr = reinterpret_cast<uint8_t*>(vertical_gradient_ptr);
			vertical_gradient_byte_ptr += vertical_gradient_pitch*patch_image_index_v + sizeof(int16_t)*patch_image_index_u;
			
			const float dx = static_cast<float>(*reinterpret_cast<int16_t*>(horizontal_gradient_byte_ptr))/500.0f;
			const float dy = static_cast<float>(*reinterpret_cast<int16_t*>(vertical_gradient_byte_ptr))/500.0f;
			
			const float gaussian_weight = gaussian_kernel[offset_v + 2][offset_u + 2];
			
			dxdx += gaussian_weight * dx*dx;
			dydy += gaussian_weight * dy*dy;
			dxdy += gaussian_weight * dx*dy;
		}
	}
	
	
	
	const float determinant = dxdx*dydy - 2*dxdy;
	const float trace = dxdx + dydy;
	
	constexpr float k = 0.05;
	
	float response = determinant - k * trace * trace;
	
	uint8_t* corner_response_byte_ptr = reinterpret_cast<uint8_t*>(corner_response_ptr);
	corner_response_byte_ptr += corner_response_pitch*image_index_v + sizeof(float)*image_index_u;
	
	*reinterpret_cast<float*>(corner_response_byte_ptr) = response;
}


__global__
void compute_corner_points_kernel(
		void* corner_response_ptr, uint32_t corner_response_pitch,
		uint32_t image_width, uint32_t image_height,
		CornerPoint* point_array_ptr, uint32_t point_array_capacity,
		uint32_t* numberof_points_ptr
		) {
	const uint32_t image_index_u = blockIdx.x * blockDim.x + 32*threadIdx.x;
	const uint32_t image_index_v = blockIdx.y * blockDim.y + 32*threadIdx.y;
	
	constexpr float threshold = 30.0f;
	
	float max_response = 0.0;
	uint32_t max_response_u = 0;
	uint32_t max_response_v = 0;
	
	for (int v_offset=0; v_offset < 32; v_offset++) {
		for (int u_offset=0; u_offset < 32; u_offset++) {
			uint8_t* corner_response_byte_ptr = reinterpret_cast<uint8_t*>(corner_response_ptr);
			corner_response_byte_ptr += corner_response_pitch*(image_index_v + v_offset) + sizeof(float)*(image_index_u + u_offset);
			
			float corner_response_value = *reinterpret_cast<float*>(corner_response_byte_ptr);
			
			if (corner_response_value > max_response and corner_response_value > threshold) {
				max_response = corner_response_value;
				max_response_u = image_index_u + u_offset;
				max_response_v = image_index_v + v_offset;
			}
		}
	}
	
	uint32_t point_array_index = atomicInc(numberof_points_ptr, 9999999999);
	
	if (point_array_index >= point_array_capacity) { return; }
	
	point_array_ptr[point_array_index].u = max_response_u;
	point_array_ptr[point_array_index].v = max_response_v;
}


void compute_corners(const PitchedCUDABuffer& horizontal_gradient, const PitchedCUDABuffer& vertical_gradient, PitchedCUDABuffer& corner_response,
                     CUDAArray<CornerPoint>& corner_points_array, CUDABuffer& number_of_points_buffer) {
	
	const uint32_t image_width = horizontal_gradient.get_elements_per_row();
	const uint32_t image_height = vertical_gradient.get_number_of_rows();
	
	const dim3 corner_block_dim(32, 32, 1);
	const dim3 corner_grid_dim(image_width / corner_block_dim.x + (image_width % corner_block_dim.x == 0 ? 0 : 1),
	                              image_height / corner_block_dim.y + (image_height % corner_block_dim.y == 0 ? 0 : 1),
	                           1);
	
	compute_corner_response_kernel<<<corner_grid_dim, corner_block_dim>>>(horizontal_gradient.get_dev_ptr(), horizontal_gradient.get_pitch_in_bytes(),
	                                                                      vertical_gradient.get_dev_ptr(), vertical_gradient.get_pitch_in_bytes(),
	                                                                      corner_response.get_dev_ptr(), corner_response.get_pitch_in_bytes(),
	                                                                      image_width, image_height);
	
	CUDA_SYNC_CHECK();
	
	cudaMemset(number_of_points_buffer.get_dev_ptr(), 0, sizeof(uint32_t));
	
	const dim3 corner_points_block_dim(image_width/32, image_height/32, 1);
	const dim3 corner_points_grid_dim(1, 1, 1);
	
	compute_corner_points_kernel<<<corner_points_grid_dim, corner_points_block_dim>>>(corner_response.get_dev_ptr(), corner_response.get_pitch_in_bytes(),
	                                                                                  image_width, image_height,
	                                                                                  corner_points_array.get_dev_ptr(), corner_points_array.get_number_of_elements(),
	                                                                                  reinterpret_cast<uint32_t*>(number_of_points_buffer.get_dev_ptr()));
	CUDA_SYNC_CHECK();
}
