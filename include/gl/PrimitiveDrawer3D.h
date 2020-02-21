#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Core>

#include "util/Transform.h"
#include "util/Resource.h"
#include "gl/Shader.h"
#include "Shader.h"

namespace Toucan {

class PrimitiveDrawer3D {
public:
	PrimitiveDrawer3D();
	
	enum class PrimitiveShape { Cube, Sphere, Cylinder };
	
	struct Primitive3D {
		Eigen::Vector3f position;
		Eigen::Quaternionf orientation;
		Eigen::Vector3f scale;
		Eigen::Vector4f color;
		PrimitiveShape shape;
		
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	
	void set_data(const Primitive3D& primitive);
	void set_data(const std::vector<Primitive3D>& primitives);
	
	void draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera = Transform::Identity(), const Transform& world_to_object = Transform::Identity()) const;
private:
	struct PrimitiveVertex {
		Eigen::Vector3f position;
		Eigen::Vector3f normal;
		Eigen::Vector4f color;
		
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	
	void set_primitive_data(const std::vector<PrimitiveVertex>& primitive_vertices);
	static void create_primitive_data(const Primitive3D& primitive, std::vector<PrimitiveVertex>& primitive_vertices);
	
	size_t m_number_of_vertices = 0;
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Shader m_shader;
};

} // namespace Toucan
