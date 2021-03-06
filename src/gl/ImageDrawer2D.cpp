#include "gl/ImageDrawer2D.h"

#include <array>
#include <glad/glad.h>

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto image_2d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 uv_coordinate;

uniform mat4 view;
uniform mat4 projection;

void main() {
	gl_Position = projection * view * vec4(position, 1.0);
	uv_coordinate = uv;
}

)GLSL";

constexpr auto image_2d_fs = R"GLSL(
#version 450

in vec2 uv_coordinate;

uniform sampler2D image;

out vec4 fragment_color;

void main() {
	fragment_color = texture(image, uv_coordinate);
}

)GLSL";

Toucan::ImageDrawer2D::ImageDrawer2D(const Settings& settings)
: m_shader{image_2d_vs, image_2d_fs},
  m_lock_image_aspect{settings.lock_image_aspect.value_or(true)} {
	
	const std::array<float, 20> vertices = {
			1.0f, 1.0f, 0.0f,   1.0f, 1.0f, // top right
			1.0f, 0.0f, 0.0f,   1.0f, 0.0f, // bottom right
			0.0f, 0.0f, 0.0f,   0.0f, 0.0f, // bottom left
			0.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
	};
	const std::array<uint32_t, 6> indices = {
			0, 1, 3,
			1, 2, 3
	};
	
	m_vao = make_resource<uint32_t>(
			[](auto& r){ glGenVertexArrays(1, &r); glCheckError(); },
			[](auto r){ glDeleteVertexArrays(1, &r); glCheckError(); }
	);
	m_vbo = make_resource<uint32_t>(
			[](auto& r){ glGenBuffers(1, &r); glCheckError(); },
			[](auto r){ glDeleteBuffers(1, &r); glCheckError(); }
	);
	m_ebo = make_resource<uint32_t>(
			[](auto& r){ glGenBuffers(1, &r); glCheckError(); },
			[](auto r){ glDeleteBuffers(1, &r); glCheckError(); }
	);
	
	glBindVertexArray(m_vao); glCheckError();
	
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo); glCheckError();
	glBufferData(GL_ARRAY_BUFFER, 20*sizeof(float), vertices.data(), GL_STATIC_DRAW); glCheckError();
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo); glCheckError();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(uint32_t), indices.data(), GL_STATIC_DRAW); glCheckError();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), reinterpret_cast<void*>(0)); glCheckError();
	glEnableVertexAttribArray(position_location); glCheckError();
	
	// UV coordinate
	constexpr auto uv_coordinate_location = 1;
	glVertexAttribPointer(uv_coordinate_location, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), reinterpret_cast<void*>(3* sizeof(float))); glCheckError();
	glEnableVertexAttribArray(uv_coordinate_location); glCheckError();
	
	glBindVertexArray(0); glCheckError();
	
	m_texture = make_resource<uint32_t>(
			[](auto& r){ glGenTextures(1, &r); glCheckError(); },
			[](auto r){ glDeleteTextures(1, &r); glCheckError(); }
	);
	
	glBindTexture(GL_TEXTURE_2D, m_texture); glCheckError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glCheckError();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); glCheckError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glCheckError();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); glCheckError();
	
	
	glBindTexture(GL_TEXTURE_2D, 0); glCheckError();
}

void Toucan::ImageDrawer2D::set_image(const Image &image) {
	glBindTexture(GL_TEXTURE_2D, m_texture); glCheckError();
	glActiveTexture(GL_TEXTURE0); glCheckError();
	switch (image.format) {
		case ImageFormat::RGB_U8: {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image.width_in_pixels, image.height_in_pixels, 0, GL_RGB, GL_UNSIGNED_BYTE, image.buffer_ptr); glCheckError();
		} break;
	}
	glBindTexture(GL_TEXTURE_2D, 0); glCheckError();
	
	m_image_width = image.width_in_pixels;
	m_image_height = image.height_in_pixels;
}

void Toucan::ImageDrawer2D::draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle) {
	if (m_texture.is_empty()) { return; }
	
	Rectangle adjusted_draw_rectangle = draw_rectangle;
	
	if (m_lock_image_aspect) {
		// Adjust the smallest dimension of the image to preserve aspect ratio.
		const auto& source_width = static_cast<float>(m_image_width);
		const auto& source_height = static_cast<float>(m_image_height);
		
		const float& target_width = adjusted_draw_rectangle.get_size().x();
		const float& target_height = adjusted_draw_rectangle.get_size().y();
		
		const float width_scale = target_width/source_width;
		const float height_scale = target_height/source_height;
		
		const float scale = std::min(width_scale, height_scale);
		
		adjusted_draw_rectangle.set_size(scale * Eigen::Vector2f(source_width, source_height));
	}
	
	// Prepare shader
	m_shader.use();
	m_shader.set_uniform("image", 0);
	const Eigen::Matrix4f view_matrix = view_matrix_2d(adjusted_draw_rectangle);
	m_shader.set_uniform("view", view_matrix);
	const Eigen::Matrix4f projection_matrix = projection_matrix_2d(framebuffer_size.cast<float>());
	m_shader.set_uniform("projection", projection_matrix);
	
	// Draw
	glBindVertexArray(m_vao);
	glDisable(GL_DEPTH_TEST);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);
}
