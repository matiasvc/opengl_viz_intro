#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include "util/Resource.h"
#include "util/Rectangle.h"
#include "gl/Shader.h"

namespace Toucan {

class ImageDrawer2D {
public:
	struct Settings {
		std::optional<bool> lock_image_aspect;
	};
	explicit ImageDrawer2D(const Settings& settings = {});
	
	enum class ImageFormat {RGB_U8};
	struct Image {
		void* buffer_ptr;
		ImageFormat format;
		uint32_t width_in_pixels;
		uint32_t height_in_pixels;
	};
	void set_image(const Image& image);
	void draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle);
	
private:
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Resource<uint32_t> m_ebo;
	Resource<uint32_t> m_texture;
	Shader m_shader;
	
	uint32_t m_image_width = 1;
	uint32_t m_image_height = 1;
	
	// Settings
	const bool m_lock_image_aspect;
};

} // namespace Toucan
