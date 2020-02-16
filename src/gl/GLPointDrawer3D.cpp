#include "gl/GLPointDrawer3D.h"

#include <glad/glad.h>

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto point_3d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;
layout (location = 2) in float size;
layout (location = 3) in int shape;

out vec4 point_color;
out int point_shape;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
	point_color = color;
	point_shape = shape;
	gl_PointSize = size;
	gl_Position = projection * view * model * vec4(position, 1.0);
}
)GLSL";

constexpr auto point_3d_fs = R"GLSL(
#version 450 core

in vec4 point_color;
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
	
	frag_color = point_color;
}
)GLSL";


GLPointDrawer3D::GLPointDrawer3D()
: m_shader{point_3d_vs, point_3d_fs}
{ }

void GLPointDrawer3D::set_data(const std::vector<Point3D>& points)
{
	m_vao = make_resource<uint32_t>(
		[](auto& r){ glGenVertexArrays(1, &r); glCheckError(); },
		[](auto r){ glDeleteVertexArrays(1, &r); glCheckError(); }
	);
	m_vbo = make_resource<uint32_t>(
		[](auto& r){ glGenBuffers(1, &r); glCheckError(); },
		[](auto r){ glDeleteBuffers(1, &r); glCheckError(); }
	);
	
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Point3D) * points.size(), points.data(), GL_STATIC_DRAW);
	m_number_of_points = points.size();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::position)));
	glEnableVertexAttribArray(position_location);
	
	// Color
	constexpr auto color_location = 1;
	glVertexAttribPointer(color_location, 4, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::color)));
	glEnableVertexAttribArray(color_location);
	
	// Size
	constexpr auto size_location = 2;
	glVertexAttribPointer(size_location, 1, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::size)));
	glEnableVertexAttribArray(size_location);
	
	// Shape
	constexpr auto shape_location = 3;
	glVertexAttribIPointer(shape_location, 1, GL_INT, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::shape)));
	glEnableVertexAttribArray(shape_location);
	
	glBindVertexArray(0);
}

void GLPointDrawer3D::draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera, const Transform& world_to_object) const {
	if(m_number_of_points == 0) { return; }
	
	// Prepare shader
	m_shader.use();
	const Eigen::Matrix4f model = world_to_object.get_inverse().get_matrix();
	m_shader.set_uniform("model", model);
	const Eigen::Matrix4f view = world_to_camera.get_matrix();
	m_shader.set_uniform("view", view);
	m_shader.set_uniform("projection", camera_projection);
	
	// Draw
	glBindVertexArray(m_vao);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDrawArrays(GL_POINTS, 0, m_number_of_points); glCheckError();
	glBindVertexArray(0);
}
