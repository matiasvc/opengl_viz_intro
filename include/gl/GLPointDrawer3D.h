#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Core>

#include "util/Transform.h"
#include "util/Resource.h"
#include "gl/GLShader.h"


class GLPointDrawer3D
{
public:
	GLPointDrawer3D();
	
	enum class PointShape : uint8_t {Square = 0, Circle = 1, Diamond = 2, Cross = 3};
	struct GLPoint3D {
		Eigen::Vector3f position;
		Eigen::Vector4f color;
		float size;
		PointShape shape;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	void set_data(const std::vector<GLPoint3D>& points);
	
	void draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera = Transform::Identity(), const Transform& world_to_object = Transform::Identity()) const;
	
private:
	size_t m_number_of_points = 0;
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	GLShader m_shader;
};

