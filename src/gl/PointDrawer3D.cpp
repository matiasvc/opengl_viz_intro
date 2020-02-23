#include "gl/PointDrawer3D.h"

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
	
	vec4 point_position_cam_space = view * model * vec4(position, 1.0);

	gl_PointSize = 10.0 * size / distance(vec3(0.0, 0.0, 0.0), point_position_cam_space.xyz);
	gl_Position = projection * point_position_cam_space;
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


Toucan::PointDrawer3D::PointDrawer3D()
: m_shader{point_3d_vs, point_3d_fs}
{ }

void Toucan::PointDrawer3D::set_data(const std::vector<Point3D>& points) {
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(Point3D) * points.size(), points.data(), GL_DYNAMIC_DRAW); glCheckError();
	m_number_of_points = points.size();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::position))); glCheckError();
	glEnableVertexAttribArray(position_location); glCheckError();
	
	// Color
	constexpr auto color_location = 1;
	glVertexAttribPointer(color_location, 4, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::color))); glCheckError();
	glEnableVertexAttribArray(color_location); glCheckError();
	
	// Size
	constexpr auto size_location = 2;
	glVertexAttribPointer(size_location, 1, GL_FLOAT, GL_FALSE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::size))); glCheckError();
	glEnableVertexAttribArray(size_location); glCheckError();
	
	// Shape
	constexpr auto shape_location = 3;
	glVertexAttribIPointer(shape_location, 1, GL_UNSIGNED_BYTE, sizeof(Point3D), reinterpret_cast<void*>(offset_of(&Point3D::shape))); glCheckError();
	glEnableVertexAttribArray(shape_location); glCheckError();
	
	glBindVertexArray(0); glCheckError();
}

void Toucan::PointDrawer3D::draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera, const Transform& world_to_object) const {
	if (m_number_of_points == 0) { return; }
	
	// Prepare shader
	m_shader.use();
	const Eigen::Matrix4f model = world_to_object.get_inverse().get_matrix();
	m_shader.set_uniform("model", model);
	const Eigen::Matrix4f view = world_to_camera.get_matrix();
	m_shader.set_uniform("view", view);
	m_shader.set_uniform("projection", camera_projection);
	
	// Draw
	glBindVertexArray(m_vao); glCheckError();
	glEnable(GL_PROGRAM_POINT_SIZE); glCheckError();
	glEnable(GL_DEPTH_TEST); glCheckError();
	glEnable(GL_BLEND); glCheckError();
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glCheckError();
	glDrawArrays(GL_POINTS, 0, m_number_of_points); glCheckError();
	glBindVertexArray(0); glCheckError();
}
