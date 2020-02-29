#pragma once


#pragma once

#include <vector>
#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include <util/Rectangle.h>
#include <util/Resource.h>
#include <gl/Shader.h>


namespace Toucan {

class CUDAPointDrawer2D {
public:
	struct Settings {
		std::optional<bool> lock_image_aspect;
	};
	explicit PointDrawer2D(const Settings& = {});
	
	enum class PointShape : uint8_t { Square = 0, Circle = 1, Diamond = 2, Cross = 3 };
	struct Point2D {
		Eigen::Vector2f position;
		Eigen::Vector3f color;
		float size;
		PointShape shape;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	};
	
	void set_data(const std::vector<Point2D>& points);
	void draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle, const Rectangle& data_rectangle);

private:
	size_t m_number_of_points = 0;
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Shader m_shader;
	
	// Settings
	const bool m_lock_image_aspect;
};

} // namespace Toucan


