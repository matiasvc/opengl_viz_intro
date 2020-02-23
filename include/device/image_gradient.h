#pragma once

#include "device/PitchedCUDABuffer.h"

void compute_gradient(const PitchedCUDABuffer& grayscale_image, PitchedCUDABuffer& horizontal_gradient, PitchedCUDABuffer& vertical_gradient);
