#include "gl/LineDrawer3D.h"

#include <glad/glad.h>

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto line_3d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;

out vec4 line_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	line_color = color;
	gl_Position = projection * view * model * vec4(position, 1.0);
}
)GLSL";

constexpr auto line_3d_fs = R"GLSL(
#version 450 core

in vec4 line_color;

out vec4 frag_color;

void main() {
	frag_color = line_color;
}
)GLSL";

Toucan::LineDrawer3D::LineDrawer3D()
: m_shader{line_3d_vs, line_3d_fs}
{ }

void Toucan::LineDrawer3D::set_data(const std::vector<LineVertex>& lines, DrawMode draw_mode, float line_width)
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(LineVertex) * lines.size(), lines.data(), GL_STATIC_DRAW);
	m_number_of_line_vertices = lines.size();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offset_of(&LineVertex::position)));
	glEnableVertexAttribArray(position_location);
	
	// Color
	constexpr auto color_location = 1;
	glVertexAttribPointer(color_location, 4, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offset_of(&LineVertex::color)));
	glEnableVertexAttribArray(color_location);
	
	glBindVertexArray(0);
	
	m_draw_mode = draw_mode;
	m_line_width = line_width;
}

void Toucan::LineDrawer3D::draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera, const Transform& world_to_object) const {
	if (m_number_of_line_vertices == 0) { return; }
	
	// Prepare shader
	m_shader.use();
	const Eigen::Matrix4f model = world_to_object.get_inverse().get_matrix();
	m_shader.set_uniform("model", model);
	const Eigen::Matrix4f view = world_to_camera.get_matrix();
	m_shader.set_uniform("view", view);
	m_shader.set_uniform("projection", camera_projection);
	
	// Draw
	glBindVertexArray(m_vao);
	glLineWidth(m_line_width);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	switch (m_draw_mode) {
		case DrawMode::Lines: {
			glDrawArrays(GL_LINES, 0, m_number_of_line_vertices);
		} break;
		case DrawMode::LineStrip: {
			glDrawArrays(GL_LINE_STRIP, 0, m_number_of_line_vertices);
		} break;
		case DrawMode::LineLoop: {
			glDrawArrays(GL_LINE_LOOP, 0, m_number_of_line_vertices);
		} break;
	}
	glLineWidth(1.0f);
	glBindVertexArray(0);
}
