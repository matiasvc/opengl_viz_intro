#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>


class Transform {
public:
	Transform();
	explicit Transform(const Eigen::Vector3f& translation);
	explicit Transform(Eigen::Vector3f&& translation);
	Transform(const Eigen::Vector3f& translation, const Eigen::Quaternionf& orientation);
	Transform(Eigen::Vector3f&& translation, Eigen::Quaternionf&& orientation);
	
	static Transform Identity();
	
	[[nodiscard]] const Eigen::Vector3f& get_translation() const;
	[[nodiscard]] const Eigen::Quaternionf& get_orientation() const;
	
	[[nodiscard]] Transform get_inverse() const;
	[[nodiscard]] Eigen::Matrix4f get_matrix() const;
	
private:
	const Eigen::Vector3f m_translation;
	const Eigen::Quaternionf m_orientation;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

Transform operator*(const Transform& t1, const Transform& t2);
