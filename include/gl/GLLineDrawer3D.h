#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Core>

#include "util/Transform.h"
#include "util/Resource.h"
#include "gl/GLShader.h"


class GLLineDrawer3D
{
public:
	GLLineDrawer3D();
	
	struct LineVertex {
		Eigen::Vector3f position;
		Eigen::Vector4f color;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	enum class DrawMode {Lines, LineStrip, LineLoop};
	void set_data(const std::vector<LineVertex>& lines, DrawMode draw_mode = DrawMode::LineStrip, float line_width = 1.0f);
	
	void draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera = Transform::Identity(), const Transform& world_to_object = Transform::Identity()) const;
	
private:
	size_t m_number_of_line_vertices = 0;
	DrawMode m_draw_mode = DrawMode::LineStrip;
	float m_line_width = 1.0f;
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	GLShader m_shader;
};

