#pragma once

#include <vector>
#include <utility>
#include <experimental/filesystem>

#include "device/PitchedCUDABuffer.h"

class ImageLoader {
public:
	explicit ImageLoader(std::experimental::filesystem::path dataset_path);
	
	[[nodiscard]] size_t size() const;
	[[nodiscard]] bool has_next() const;
	void next();
	
	void get_image(PitchedCUDABuffer& image_buffer) const;
	
private:
	size_t m_image_index = 0;
	
	const std::experimental::filesystem::path m_dataset_path;
	
	std::vector<std::pair<double, const std::string>> m_rgb_files;
};


