#include "gl/OrbitCamera.h"

#include "gl/GLUtils.h"

#include <cmath>

void Toucan::OrbitCamera::rotate(float pitch_delta, float yaw_delta) {
	m_pitch += pitch_delta;
	m_yaw += yaw_delta;
}

void Toucan::OrbitCamera::move(const Eigen::Vector3f &position_delta) {
	m_orbit_center_position += Eigen::Vector3f(-cosf(m_yaw)*position_delta.x() + sinf(m_yaw)*position_delta.z(),
	                                           position_delta.y(),
	                                           sinf(m_yaw)*position_delta.x() + cosf(m_yaw)*position_delta.z()) * m_distance;
}

void Toucan::OrbitCamera::change_distance(float distance_delta) {
	m_distance += distance_delta;
}

Eigen::Matrix4f Toucan::OrbitCamera::get_camera_projection(int screen_width, int screen_height, float near_clip, float far_clip, float focal_length) const {
	return projection_matrix_symmetric(near_clip, far_clip, screen_width, screen_height, focal_length);
}

Transform Toucan::OrbitCamera::get_camera_transform() const {
	
	const Eigen::Quaternionf camera_orientation = Eigen::AngleAxisf(m_yaw, Eigen::Vector3f::UnitY()) *
	                                              Eigen::AngleAxisf(m_pitch, Eigen::Vector3f::UnitX());
	
	const Eigen::Vector3f camera_position = camera_orientation.toRotationMatrix() * Eigen::Vector3f(0.0f, 0.0f, -m_distance) + m_orbit_center_position;
	
	return Transform(camera_position, camera_orientation).get_inverse();
}
