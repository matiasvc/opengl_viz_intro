#pragma once

#include <vector>
#include <utility>
#include <experimental/filesystem>

class ImageLoader {
public:
	explicit ImageLoader(std::experimental::filesystem::path dataset_path);
	
	[[nodiscard]] size_t size() const;
	[[nodiscard]] bool has_next() const;
	void next();
	
	struct Image {
		void* buffer_ptr;
		int width, height;
	};
	[[nodiscard]] Image get_image() const;
	
private:
	size_t m_image_index = 0;
	
	const std::experimental::filesystem::path m_dataset_path;
	
	std::vector<std::pair<double, const std::string>> m_rgb_files;
};


