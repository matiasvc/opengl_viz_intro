#include "gl/PointDrawer2D.h"

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto point_2d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;
layout (location = 2) in float size;
layout (location = 3) in int shape;

out vec3 point_color;
out int point_shape;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
	point_color = color;
	point_shape = shape;

	gl_PointSize = size;
	gl_Position = projection * view * model * vec4(position, 0.0, 1.0);
}
)GLSL";

constexpr auto point_2d_fs = R"GLSL(
#version 450 core

in vec3 point_color;
flat in int point_shape;

out vec4 frag_color;

void main(){
	vec2 center_coord = 2.0 * gl_PointCoord - 1.0;
	
	switch (point_shape) {
		case 0: { // Square
			// Square means we never discard
		} break;
		case 1: { // Cirle
			if(dot(center_coord, center_coord) > 1.0) {
				discard;
			}
		} break;
		case 2: { // Diamond
			if ((abs(center_coord.x) + abs(center_coord.y)) > 1.0) {
				discard;
			}
		} break;
		case 3: { // Cross
			if (abs(abs(center_coord.x) - abs(center_coord.y)) > 0.25) {
				discard;
			}
		} break;
		default: { // Draw pink color as debug
			frag_color = vec4(1.0, 0.0, 1.0, 1.0);
			return;
		}
	}
	
	frag_color = vec4(point_color, 1.0);
}
)GLSL";


Toucan::PointDrawer2D::PointDrawer2D(const Settings& settings)
: m_shader{point_2d_vs, point_2d_fs},
  m_lock_image_aspect{settings.lock_image_aspect.value_or(true)}
{ }

void Toucan::PointDrawer2D::set_data(const std::vector<Point2D> &points) {
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(Point2D) * points.size(), points.data(), GL_DYNAMIC_DRAW); glCheckError();
	m_number_of_points = points.size();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, sizeof(Point2D), reinterpret_cast<void*>(offset_of(&Point2D::position))); glCheckError();
	glEnableVertexAttribArray(position_location); glCheckError();
	
	// Color
	constexpr auto color_location = 1;
	glVertexAttribPointer(color_location, 3, GL_FLOAT, GL_FALSE, sizeof(Point2D), reinterpret_cast<void*>(offset_of(&Point2D::color))); glCheckError();
	glEnableVertexAttribArray(color_location); glCheckError();
	
	// Size
	constexpr auto size_location = 2;
	glVertexAttribPointer(size_location, 1, GL_FLOAT, GL_FALSE, sizeof(Point2D), reinterpret_cast<void*>(offset_of(&Point2D::size))); glCheckError();
	glEnableVertexAttribArray(size_location); glCheckError();
	
	// Shape
	constexpr auto shape_location = 3;
	glVertexAttribIPointer(shape_location, 1, GL_UNSIGNED_BYTE, sizeof(Point2D), reinterpret_cast<void*>(offset_of(&Point2D::shape))); glCheckError();
	glEnableVertexAttribArray(shape_location); glCheckError();
	
	glBindVertexArray(0); glCheckError();
}

void Toucan::PointDrawer2D::draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle, const Rectangle& data_rectangle) {
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
