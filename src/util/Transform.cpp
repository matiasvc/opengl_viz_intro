#include "util/Transform.h"

Transform::Transform()
: m_translation{Eigen::Vector3f::Zero()}, m_orientation{Eigen::Quaternionf::Identity()}
{ }

Transform::Transform(const Eigen::Vector3f& translation) // NOLINT
: m_translation{translation}, m_orientation{Eigen::Quaternionf::Identity()}
{ }

Transform::Transform(Eigen::Vector3f&& translation) // NOLINT
: m_translation{std::move(translation)}, m_orientation{Eigen::Quaternionf::Identity()}
{ }

Transform::Transform(const Eigen::Vector3f& translation, const Eigen::Quaternionf& orientation) // NOLINT
: m_translation{translation}, m_orientation{orientation}
{ }

Transform::Transform(Eigen::Vector3f&& translation, Eigen::Quaternionf&& orientation) // NOLINT
: m_translation{std::move(translation)}, m_orientation{orientation}
{ }

Transform Transform::Identity() {
	return Transform(Eigen::Vector3f::Zero(), Eigen::Quaternionf::Identity());
}

const Eigen::Vector3f& Transform::get_translation() const { return m_translation; }
const Eigen::Quaternionf& Transform::get_orientation() const { return m_orientation; }

Transform Transform::get_inverse() const {
	const Eigen::Quaternionf orientation_inverse = m_orientation.inverse();
	return Transform(-orientation_inverse.toRotationMatrix()*m_translation, orientation_inverse);
}

Eigen::Matrix4f Transform::get_matrix() const {
	Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Zero();
	transform_matrix.block<3,3>(0,0) = m_orientation.toRotationMatrix();
	transform_matrix.block<3,1>(0, 3) = m_translation;
	transform_matrix(3, 3) = 1.0f;
	return transform_matrix;
}

Transform operator*(const Transform& t1, const Transform& t2) {
	return Transform(t1.get_orientation().toRotationMatrix()*t2.get_translation() + t1.get_translation(), t1.get_orientation() * t2.get_orientation());
}
