#pragma once

#include "device/PitchedCUDABuffer.h"

void compute_grayscale(const PitchedCUDABuffer& color_image, PitchedCUDABuffer& grayscale_image);
