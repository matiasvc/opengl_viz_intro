#pragma once

#include <vector>
#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include <driver_types.h>

#include "util/Rectangle.h"
#include "util/Resource.h"
#include "gl/Shader.h"


namespace Toucan {

class CUDAPointDrawer2D {
public:
	struct Settings {
		std::optional<bool> lock_image_aspect;
	};
	explicit CUDAPointDrawer2D(const Settings& settings = {});
	
	struct CUDAPoints2D {
		float* array_dev_ptr;
		uint32_t number_of_elements;
	};
	
	void set_data(const CUDAPoints2D& points);
	void draw(const Eigen::Vector2i& framebuffer_size, const Rectangle& draw_rectangle, const Rectangle& data_rectangle);

private:
	size_t m_number_of_points = 0;
	Resource<uint32_t> m_vbo;
	Resource<uint32_t> m_vao;
	Shader m_shader;
	
	Resource<cudaStream_t> m_cuda_stream;
	cudaGraphicsResource_t m_cuda_graphics_resource = nullptr;
	
	// Settings
	const bool m_lock_image_aspect;
};

} // namespace Toucan


