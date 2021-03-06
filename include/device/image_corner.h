#pragma once

#include "device/PitchedCUDABuffer.h"
#include "device/CUDAArray.h"
#include "device/CUDABuffer.h"

struct CornerPoint {
	float u;
	float v;
};

void compute_corners(const PitchedCUDABuffer& horizontal_gradient, const PitchedCUDABuffer& vertical_gradient, PitchedCUDABuffer& corner_response,
                     CUDAArray<CornerPoint>& corner_points_array, CUDABuffer& number_of_points_buffer);
