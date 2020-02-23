#pragma once

#include <Eigen/Core>

namespace Toucan {

class Rectangle {
public:
	Rectangle(const Eigen::Vector2f& top_left, const Eigen::Vector2f& size);
	
	[[nodiscard]] const Eigen::Vector2f& get_top_left() const;
	void set_top_left(const Eigen::Vector2f& top_left);
	[[nodiscard]] const Eigen::Vector2f& get_size() const;
	void set_size(const Eigen::Vector2f& size);
	
private:
	Eigen::Vector2f m_top_left;
	Eigen::Vector2f m_size;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

}
