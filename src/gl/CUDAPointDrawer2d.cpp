#include "gl/CUDAPointDrawer2d.h"

#include <cuda_gl_interop.h>

#include "device/cuda_debug.h"
#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto point_2d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec2 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
	gl_PointSize = 10.0;
	gl_Position = projection * view * model * vec4(position, 0.0, 1.0);
}
)GLSL";

constexpr auto point_2d_fs = R"GLSL(
#version 450 core

out vec4 frag_color;

void main(){
	vec2 center_coord = 2.0 * gl_PointCoord - 1.0;
	
	if(dot(center_coord, center_coord) > 1.0) {
		discard;
	}
	
	frag_color = vec4(1.0, 0.0, 0.0, 1.0);
}
)GLSL";

Toucan::CUDAPointDrawer2D::CUDAPointDrawer2D(const Settings& settings)
: m_shader(point_2d_vs, point_2d_fs),
  m_lock_image_aspect{settings.lock_image_aspect.value_or(true)}
{ }

void Toucan::CUDAPointDrawer2D::set_data(const CUDAPoints2D& points) {
	m_vao = make_resource<uint32_t>(
			[](auto& r){ glGenVertexArrays(1, &r); glCheckError(); },
			[](auto r){ glDeleteVertexArrays(1, &r); glCheckError(); }
	);
	m_vbo = make_resource<uint32_t>(
			[](auto& r){ glGenBuffers(1, &r); glCheckError(); },
			[](auto r){ glDeleteBuffers(1, &r); glCheckError(); }
	);
	
	glBindVertexArray(m_vao); glCheckError();
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo); glCheckError();
	glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*points.number_of_elements, nullptr, GL_DYNAMIC_DRAW); glCheckError();
	m_number_of_points = points.number_of_elements * 2;
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), nullptr); glCheckError();
	glEnableVertexAttribArray(position_location); glCheckError();
	
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_graphics_resource, m_vbo, cudaGraphicsRegisterFlagsWriteDiscard));
	
	CUDA_CHECK( cudaGraphicsMapResources(1, &m_cuda_graphics_resource, m_cuda_stream) );
	void* dev_ptr = nullptr;
	size_t size = 0;
	CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(&dev_ptr, &size, m_cuda_graphics_resource) );
	
	CUDA_CHECK(cudaMemcpyAsync(dev_ptr, points.array_dev_ptr, 2* sizeof(float)*points.number_of_elements, cudaMemcpyDeviceToDevice, m_cuda_stream));
	
	CUDA_CHECK( cudaGraphicsUnmapResources(1, &m_cuda_graphics_resource, m_cuda_stream) );
	CUDA_CHECK( cudaStreamSynchronize(m_cuda_stream) );
	
	glBindVertexArray(0); glCheckError();
}

void Toucan::CUDAPointDrawer2D::draw(const Eigen::Vector2i &framebuffer_size, const Rectangle &draw_rectangle, const Rectangle &data_rectangle) {
	if (m_number_of_points == 0){ return; }
	
	Rectangle adjusted_draw_rectangle = draw_rectangle;
	
	if (m_lock_image_aspect) {
		// Adjust the smallest dimension of the image to preserve aspect ratio.
		const auto& source_width = data_rectangle.get_size().x();
		const auto& source_height = data_rectangle.get_size().y();
		
		const float& target_width = adjusted_draw_rectangle.get_size().x();
		const float& target_height = adjusted_draw_rectangle.get_size().y();
		
		const float width_scale = target_width/source_width;
		const float height_scale = target_height/source_height;
		
		const float scale = std::min(width_scale, height_scale);
		
		adjusted_draw_rectangle.set_size(scale * Eigen::Vector2f(source_width, source_height));
	}
	
	// Prepare shader
	m_shader.use();
	const Eigen::Matrix4f model = model_matrix_2d(data_rectangle);
	m_shader.set_uniform("model", model);
	const Eigen::Matrix4f view = view_matrix_2d(adjusted_draw_rectangle);
	m_shader.set_uniform("view", view);
	const Eigen::Matrix4f projection = projection_matrix_2d(framebuffer_size.cast<float>());
	m_shader.set_uniform("projection", projection);
	
	// Draw
	glBindVertexArray(m_vao); glCheckError();
	glEnable(GL_PROGRAM_POINT_SIZE); glCheckError();
	glDisable(GL_DEPTH_TEST); glCheckError();
	glDrawArrays(GL_POINTS, 0, m_number_of_points); glCheckError();
	glBindVertexArray(0); glCheckError();
}
