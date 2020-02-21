#pragma once

#include <Eigen/Core>

namespace Toucan {

class Rectangle {
public:
	Rectangle(const Eigen::Vector2f& top_left, const Eigen::Vector2f& size);
	
	[[nodiscard]] const Eigen::Vector2f& get_top_left() const;
	[[nodiscard]] const Eigen::Vector2f& get_size() const;
	
private:
	const Eigen::Vector2f m_top_left;
	const Eigen::Vector2f m_size;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

}
