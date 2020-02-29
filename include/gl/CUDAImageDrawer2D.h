#pragma once

#include <cstdint>
#include <optional>
#include <Eigen/Core>

#include <driver_types.h>

#include "util/Resource.h"
#include "util/Rectangle.h"
#include "gl/Shader.h"

namespace Toucan {

class CUDAImageDrawer2D {
public:
	struct Settings {
		std::optional<bool> lock_image_aspect;
	};
	explicit CUDAImageDrawer2D(const Settings& settings = {});
	
	enum class ImageFormat {RGBX_U8, R_U8, R_S16, R_F32};
	struct CUDAImage {
		void* dev_buffer_ptr;
		ImageFormat format;
		uint32_t width_in_pixels;
		uint32_t height_in_pixels;
		uint32_t pixel_size_in_bytes;
		uint32_t pitch_in_bytes;
	};
	
	void set_image(const CUDAImage& cuda_image);
	void draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle);
	
private:
	
	void update_texture(const CUDAImage& cuda_image);
	
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Resource<uint32_t> m_ebo;
	Resource<uint32_t> m_texture;
	Shader m_shader;
	
	Resource<cudaStream_t> m_cuda_stream;
	cudaGraphicsResource_t m_cuda_graphics_resource = nullptr;
	
	uint32_t m_image_width = 1;
	uint32_t m_image_height = 1;
	ImageFormat m_image_format = ImageFormat::RGBX_U8;
	
	// Settings
	const bool m_lock_image_aspect;
};

} // namespace Toucan

