#include "gl/GLPrimitiveDrawer3D.h"

#include <glad/glad.h>

#include "gl/GLDebug.h"
#include "gl/GLUtils.h"

constexpr auto primitive_3d_vs = R"GLSL(
#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;

out vec3 vertex_normal;
out vec4 vertex_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	vertex_normal = mat3(view)*mat3(model)*normal;
	vertex_color = color;
	gl_Position = projection * view * model * vec4(position, 1.0);
}
)GLSL";

constexpr auto primitive_3d_fs = R"GLSL(
#version 450 core

in vec3 vertex_normal;
in vec4 vertex_color;

out vec4 frag_color;

void main() {
	float angle = dot(vertex_normal, vec3(0, 0, -1));
	frag_color = vec4((0.5 + 0.5*angle)*vertex_color.xyz, vertex_color.w);
}

)GLSL";

GLPrimitiveDrawer3D::GLPrimitiveDrawer3D()
: m_shader{primitive_3d_vs, primitive_3d_fs}
{ }


void GLPrimitiveDrawer3D::set_data(const Primitive3D& primitive) {
	std::vector<PrimitiveVertex> primitive_vertices;
	
	create_primitive_data(primitive, primitive_vertices);
	
	set_primitive_data(primitive_vertices);
}

void GLPrimitiveDrawer3D::set_data(const std::vector<Primitive3D>& primitives) {
	std::vector<PrimitiveVertex> primitive_vertices;
	
	for (const auto& primitive : primitives) {
		create_primitive_data(primitive, primitive_vertices);
	}
	
	set_primitive_data(primitive_vertices);
}

void GLPrimitiveDrawer3D::draw(const Eigen::Matrix4f &camera_projection, const Transform &world_to_camera, const Transform &world_to_object) const {
	if(m_number_of_vertices == 0) { return; }
	
	// Prepare shader
	m_shader.use();
	const Eigen::Matrix4f model = world_to_object.get_inverse().get_matrix();
	m_shader.set_uniform("model", model);
	const Eigen::Matrix4f view = world_to_camera.get_matrix();
	m_shader.set_uniform("view", view);
	m_shader.set_uniform("projection", camera_projection);
	
	// Draw
	glBindVertexArray(m_vao);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_CULL_FACE);
	glDrawArrays(GL_TRIANGLES, 0, m_number_of_vertices);
}

void GLPrimitiveDrawer3D::set_primitive_data(const std::vector<PrimitiveVertex>& primitive_vertices)
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
	glBufferData(GL_ARRAY_BUFFER, sizeof(PrimitiveVertex)*primitive_vertices.size(), primitive_vertices.data(), GL_STATIC_DRAW);
	
	m_number_of_vertices = primitive_vertices.size();
	
	// Position
	constexpr auto position_location = 0;
	glVertexAttribPointer(position_location, 3, GL_FLOAT, GL_FALSE, sizeof(PrimitiveVertex), reinterpret_cast<void*>(offset_of(&PrimitiveVertex::position)));
	glEnableVertexAttribArray(position_location);
	
	// Normal
	constexpr auto normal_location = 1;
	glVertexAttribPointer(normal_location, 3, GL_FLOAT, GL_FALSE, sizeof(PrimitiveVertex), reinterpret_cast<void*>(offset_of(&PrimitiveVertex::normal)));
	glEnableVertexAttribArray(normal_location);
	
	// Color
	constexpr auto color_location = 2;
	glVertexAttribPointer(color_location, 4, GL_FLOAT, GL_FALSE, sizeof(PrimitiveVertex), reinterpret_cast<void*>(offset_of(&PrimitiveVertex::color)));
	glEnableVertexAttribArray(color_location);
	
	glBindVertexArray(0);
}

void GLPrimitiveDrawer3D::create_primitive_data(const Primitive3D& primitive, std::vector<PrimitiveVertex>& primitive_vertices)
{
	const Eigen::Vector3f& position = primitive.position;
	const Eigen::Matrix3f orientation = primitive.orientation.toRotationMatrix();
	const Eigen::Matrix3f scaled_orientation = orientation.array().colwise() * primitive.scale.array();
	const Eigen::Matrix3f normal_orientation = scaled_orientation.inverse().transpose();
	
	switch (primitive.shape) {
		case PrimitiveShape::Cube: {
			
			const std::array<PrimitiveVertex, 36> current_primitive_vertices = {
				// Top
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 3
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 1
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 2
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 3
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 0
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitY())),primitive.color}, // 1
				
				// Left
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 3
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 4
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 0
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 3
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, 0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 7
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitX())),primitive.color}, // 4
				
				// Back
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 0
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 5
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 1
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 0
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 4
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, -0.5f) + position, (normal_orientation * (-Eigen::Vector3f::UnitZ())),primitive.color}, // 5
				
				// Right
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 1
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 6
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 2
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 1
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 5
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitX())),primitive.color}, // 6
				
				// Bottom
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 4
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 6
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 5
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, -0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 4
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 7
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitY())),primitive.color}, // 6
				
				// Front
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 7
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 2
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 6
				
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, 0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 7
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(-0.5f, -0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 3
				PrimitiveVertex{scaled_orientation * Eigen::Vector3f(0.5f, -0.5f, 0.5f) + position, (normal_orientation * (Eigen::Vector3f::UnitZ())),primitive.color}, // 2
			};
			primitive_vertices.reserve(primitive_vertices.size() + 8);
			primitive_vertices.insert(std::end(primitive_vertices), std::begin(current_primitive_vertices), std::end(current_primitive_vertices));
			
			
		} break;
		case PrimitiveShape::Sphere: {
		
		} break;
		case PrimitiveShape::Cylinder: {
		
		} break;
	}
}
