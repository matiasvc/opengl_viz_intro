#include "device/image_grayscale.h"

#include <cassert>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "device/cuda_debug.h"

__global__ void compute_grayscale_kernel(
		void* color_img_ptr, uint32_t color_img_pitch,
		void* grayscale_img_ptr, uint32_t grayscale_image_pitch,
		uint32_t image_width, uint32_t image_height) {
	
	const uint32_t image_index_u = blockIdx.x*blockDim.x + threadIdx.x;
	const uint32_t image_index_v = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (image_index_u >= image_width or image_index_v >= image_height) { return; }
	
	uint8_t* color_byte_ptr = reinterpret_cast<uint8_t*>(color_img_ptr);
	color_byte_ptr += color_img_pitch*image_index_v + sizeof(uchar4)*image_index_u;
	uint8_t* grayscale_byte_ptr = reinterpret_cast<uint8_t*>(grayscale_img_ptr);
	grayscale_byte_ptr += grayscale_image_pitch*image_index_v + sizeof(uchar1)*image_index_u;
	
	uchar4* color_pixel_ptr = reinterpret_cast<uchar4*>(color_byte_ptr);
	uchar1* grayscale_pixel_ptr = reinterpret_cast<uchar1*>(grayscale_byte_ptr);
	
	uchar4 color_pixel = *color_pixel_ptr;
	float grayscale_value = 0.3f*static_cast<float>(color_pixel.x) + 0.59f*static_cast<float>(color_pixel.y) + 0.11f*static_cast<float>(color_pixel.z);
	uchar1 grayscale_pixel = make_uchar1(__float2int_rn(grayscale_value));
	*grayscale_pixel_ptr = grayscale_pixel;
}


void compute_grayscale(const PitchedCUDABuffer& color_image, PitchedCUDABuffer& grayscale_image) {
	// Sanity check the input
	assert(color_image.get_element_size_in_bytes() == 4 * sizeof(uint8_t));
	assert(grayscale_image.get_element_size_in_bytes() == sizeof(uint8_t));
	assert(color_image.get_elements_per_row() == grayscale_image.get_elements_per_row());
	assert(color_image.get_number_of_rows() == grayscale_image.get_number_of_rows());
	
	const uint32_t image_width = color_image.get_elements_per_row();
	const uint32_t image_height = color_image.get_number_of_rows();
	
	const dim3 grayscale_block_dim(32, 32, 1);
	const dim3 grayscale_grid_dim(image_width/grayscale_block_dim.x + (image_width % grayscale_block_dim.x == 0 ? 0 : 1),
	                              image_height/grayscale_block_dim.y + (image_height % grayscale_block_dim.y == 0 ? 0 : 1),
	                              1);
	
	compute_grayscale_kernel<<<grayscale_grid_dim, grayscale_block_dim>>>(color_image.get_dev_ptr(), color_image.get_pitch_in_bytes(),
	                                                                      grayscale_image.get_dev_ptr(), grayscale_image.get_pitch_in_bytes(),
	                                                                      image_width, image_height);
	
	CUDA_SYNC_CHECK();
}
