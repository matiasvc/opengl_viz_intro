#pragma once

#include <cstdint>

#include "util/Resource.h"
#include "util/Rectangle.h"
#include "gl/Shader.h"

namespace Toucan {

class ImageDrawer2D {
public:
	ImageDrawer2D();
	
	enum class ImageFormat {RGB_U8};
	struct Image {
		void* buffer_ptr;
		ImageFormat format;
		uint32_t width;
		uint32_t height;
	};
	void set_texture(const Image& image);
	void draw(const Rectangle& rectangle, int framebuffer_width, int framebuffer_height);
	
private:
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Resource<uint32_t> m_ebo;
	Resource<uint32_t> m_texture;
	Shader m_shader;
};

} // namespace Toucan
