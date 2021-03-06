#pragma once

#include "util/Rectangle.h"

template<typename T, typename U> constexpr size_t offset_of(U T::*member){
	return (char*)&((T*)nullptr->*member) - (char*)nullptr;
}

inline Eigen::Matrix4f projection_matrix(float z_near, float z_far, float x_left, float x_right, float y_top, float y_bottom){
	Eigen::Matrix4f A;
	A << 2.0f * z_near / (x_right - x_left),  0.0f,                               -(x_right + x_left)/(x_right - x_left),  0.0f,
	     0.0f,                               -2.0f * z_near / (y_bottom - y_top),  (y_top + y_bottom)/(y_bottom - y_top),  0.0f,
	     0.0f,                                0.0f,                                (z_far + z_near) / (z_far - z_near),   -2.0f*z_far*z_near / (z_far - z_near),
	     0.0f,                                0.0f,                                1.0f,                                   0.0f;
	
	return A;
}

inline Eigen::Matrix4f projection_matrix(float z_near, float z_far, float fx, float fy, float cx, float cy, const Eigen::Vector2f& image_size){
	float x_left = -cx/fx * z_near;
	float x_right = (image_size.x() - cx)/fx * z_near;
	float y_top = -cy/fy * z_near;
	float y_bottom = (image_size.y() - cy)/fy * z_near;
	return projection_matrix(z_near, z_far, x_left, x_right, y_top, y_bottom);
}

inline Eigen::Matrix4f projection_matrix_symmetric(float z_near, float z_far, const Eigen::Vector2i& screen_size, float f){
	return projection_matrix(z_near, z_far, f, f, 0.5f* static_cast<float>(screen_size.x()), 0.5f* static_cast<float>(screen_size.y()), screen_size.cast<float>());
}

inline Eigen::Matrix4f projection_matrix_2d(const Eigen::Vector2f& framebuffer_size) {
	const Eigen::Matrix4f P =
	[&]{
		Eigen::Matrix4f matrix;
		matrix << 2.0f/framebuffer_size.x(),  0.0f,                      0.0f, -1.0f,
		          0.0f,                      -2.0f/framebuffer_size.y(), 0.0f,  1.0f,
		          0.0f,                       0.0f,                      1.0f,  0.0f,
		          0.0f,                       0.0f,                      0.0f,  1.0f;
		return matrix;
	}();
	return P;
}

inline Eigen::Matrix4f view_matrix_2d(const Toucan::Rectangle& rectangle) {
	const float& x = rectangle.get_top_left().x();
	const float& y = rectangle.get_top_left().y();
	const float& width = rectangle.get_size().x();
	const float& height = rectangle.get_size().y();
	
	const Eigen::Matrix4f P =
	[&]{
		Eigen::Matrix4f matrix;
		matrix << width, 0.0f,   0.0f, x,
		          0.0f,  height, 0.0f, y,
		          0.0f,  0.0f,   1.0f, 0.0f,
		          0.0f,  0.0f,   0.0f, 1.0f;
		return matrix;
	}();
	return P;
}

inline Eigen::Matrix4f model_matrix_2d(const Toucan::Rectangle& rectangle) {
	const float& x = rectangle.get_top_left().x();
	const float& y = rectangle.get_top_left().y();
	const float& width = rectangle.get_size().x();
	const float& height = rectangle.get_size().y();
	
	const Eigen::Matrix4f P =
	[&]{
		Eigen::Matrix4f matrix;
		matrix << 1.0f/width, 0.0f,        0.0f, -x/width,
		          0.0f,       1.0f/height, 0.0f, -y/height,
		          0.0f,       0.0f,        1.0f,  0.0f,
		          0.0f,       0.0f,        0.0f,  1.0f;
		return matrix;
	}();
	return P;
}
