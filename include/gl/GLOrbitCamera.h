#pragma once

#include <cmath>
#include <Eigen/Core>

#include "util/Transform.h"

class GLOrbitCamera {
public:

	void rotate(float pitch_delta, float yaw_delta);
	void move(const Eigen::Vector3f& position_delta);
	void change_distance(float distance_delta);
	
	[[nodiscard]] Eigen::Matrix4f get_camera_projection(int screen_width, int screen_height, float near_clip, float far_clip, float focal_length) const;
	[[nodiscard]] Transform get_camera_transform() const;

private:
	float m_pitch = -M_PI/4.0f;
	float m_yaw = M_PI/3.0f;
	float m_distance = 3.5f;
	
	Eigen::Vector3f m_orbit_center_position = Eigen::Vector3f::Zero();
	
};

