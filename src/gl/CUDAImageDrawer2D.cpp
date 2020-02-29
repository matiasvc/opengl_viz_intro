#include "gl/CUDAImageDrawer2D.h"

#include <array>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"
#include "device/cuda_debug.h"

constexpr auto cuda_image_2d_vs = R"GLSL(
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

constexpr auto cuda_image_2d_rgb_fs = R"GLSL(
#version 450

in vec2 uv_coordinate;

uniform sampler2D image;

out vec4 fragment_color;

void main() {
	fragment_color = vec4(texture(image, uv_coordinate).rgb, 1.0);
}

)GLSL";

constexpr auto cuda_image_2d_grayscale_fs = R"GLSL(
#version 450

in vec2 uv_coordinate;

uniform sampler2D image;

out vec4 fragment_color;

void main() {
	fragment_color = vec4(texture(image, uv_coordinate).rrr, 1.0);
}

)GLSL";


constexpr auto cuda_image_2d_gradient_fs = R"GLSL(
#version 450

in vec2 uv_coordinate;

uniform isampler2D image;

out vec4 fragment_color;

void main() {
	vec3 tex_value = (texture(image, uv_coordinate)/3000.0).rrr;
	fragment_color = vec4(0.5 + tex_value, 1.0);
}

)GLSL";

Toucan::CUDAImageDrawer2D::CUDAImageDrawer2D(const Toucan::CUDAImageDrawer2D::Settings& settings)
: m_lock_image_aspect{settings.lock_image_aspect.value_or(true)} {
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
	
	m_cuda_stream = make_resource<cudaStream_t>(
			[](auto& r){ cudaStreamCreate(&r); },
			[](auto r){ cudaStreamSynchronize(r); cudaStreamDestroy(r); }
	);
}

void Toucan::CUDAImageDrawer2D::set_image(const CUDAImage &cuda_image) {
	
	update_texture(cuda_image);
	
	glBindTexture(GL_TEXTURE_2D, m_texture); glCheckError();
	glActiveTexture(GL_TEXTURE0); glCheckError();
	
	CUDA_CHECK( cudaGraphicsMapResources(1, &m_cuda_graphics_resource, m_cuda_stream) );
	cudaArray_t cuda_array = nullptr;
	CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray(&cuda_array, m_cuda_graphics_resource, 0, 0) );
	
	CUDA_CHECK( cudaMemcpy2DToArrayAsync(
			cuda_array, 0, 0,
			cuda_image.dev_buffer_ptr, cuda_image.pitch_in_bytes, cuda_image.pixel_size_in_bytes * cuda_image.width_in_pixels, cuda_image.height_in_pixels,
			cudaMemcpyDeviceToDevice, m_cuda_stream
	) );
	
	
	CUDA_CHECK( cudaGraphicsUnmapResources(1, &m_cuda_graphics_resource, m_cuda_stream) );
	CUDA_CHECK( cudaStreamSynchronize(m_cuda_stream) );
	
	glBindTexture(GL_TEXTURE_2D, 0); glCheckError();
	
	m_image_width = cuda_image.width_in_pixels;
	m_image_height = cuda_image.height_in_pixels;
}

void Toucan::CUDAImageDrawer2D::draw(const Eigen::Vector2i &framebuffer_size, const Rectangle &draw_rectangle) {
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

void Toucan::CUDAImageDrawer2D::update_texture(const Toucan::CUDAImageDrawer2D::CUDAImage& cuda_image) {
	if (m_image_width == cuda_image.width_in_pixels and
	    m_image_height == cuda_image.height_in_pixels and
	    m_image_format == cuda_image.format) { return; }
	
	m_texture = make_resource<uint32_t>(
			[](auto& r){ glGenTextures(1, &r); glCheckError(); },
			[](auto r){ glDeleteTextures(1, &r); glCheckError(); }
	);
	
	glBindTexture(GL_TEXTURE_2D, m_texture); glCheckError();
	glActiveTexture(GL_TEXTURE0); glCheckError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glCheckError();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); glCheckError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glCheckError();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); glCheckError();
	
	switch (cuda_image.format) {
		case ImageFormat::RGBX_U8: {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, cuda_image.width_in_pixels, cuda_image.height_in_pixels, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr); glCheckError();
			m_shader = Shader(cuda_image_2d_vs, cuda_image_2d_rgb_fs);
		} break;
		case ImageFormat::R_U8: {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, cuda_image.width_in_pixels, cuda_image.height_in_pixels, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr); glCheckError();
			m_shader = Shader(cuda_image_2d_vs, cuda_image_2d_grayscale_fs);
		} break;
		case ImageFormat::R_F32: {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, cuda_image.width_in_pixels, cuda_image.height_in_pixels, 0, GL_RED, GL_FLOAT, nullptr); glCheckError();
			m_shader = Shader(cuda_image_2d_vs, cuda_image_2d_grayscale_fs);
		} break;
		case ImageFormat::R_S16: {
			glTexImage2D(GL_TEXTURE_2D, 0, GL_R16I, cuda_image.width_in_pixels, cuda_image.height_in_pixels, 0, GL_RED_INTEGER, GL_SHORT, nullptr); glCheckError();
			m_shader = Shader(cuda_image_2d_vs, cuda_image_2d_gradient_fs);
		} break;
		default: {
			throw std::runtime_error("ERROR: Unknown image format.");
		}
	}
	
	glBindTexture(GL_TEXTURE_2D, 0); glCheckError();
	CUDA_CHECK( cudaGraphicsGLRegisterImage(&m_cuda_graphics_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) );
	
	m_image_width = cuda_image.width_in_pixels;
	m_image_height = cuda_image.height_in_pixels;
	m_image_format = cuda_image.format;
}
