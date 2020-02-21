#include "ImageLoader.h"

#include <ostream>
#include <fstream>

#include <stb/stb_image.h>

ImageLoader::ImageLoader(std::experimental::filesystem::path dataset_path)
: m_dataset_path{std::move(dataset_path)} {
	const std::experimental::filesystem::path rgb_file_path = m_dataset_path / "rgb.txt";
	
	std::ifstream rgb_file(rgb_file_path.string());
	if (!rgb_file.is_open()) {
		std::ostringstream ss;
		ss << "ERROR! Unable to open: " << rgb_file_path.string() << '\n';
		throw std::invalid_argument(ss.str());
	}
	
	std::string line;
	const std::string delimiter = " ";
	
	while (std::getline(rgb_file, line)) {
		if (line[0] == '#') { continue; }
		const size_t delimiter_pos = line.find(delimiter);
		
		const double timestamp = std::stod(line.substr(0, delimiter_pos));
		const std::string file_name = line.substr(delimiter_pos + 1, line.length());
		
		m_rgb_files.emplace_back(timestamp, file_name);
	}
}

size_t ImageLoader::size() const { return m_rgb_files.size(); }

bool ImageLoader::has_next() const { return m_image_index < m_rgb_files.size() - 1; }

void ImageLoader::next() {
	if (has_next()) {
		m_image_index++;
	}
}

ImageLoader::Image ImageLoader::get_image() const {
	const auto rgb_image_path = m_dataset_path / m_rgb_files.at(m_image_index).second;
	
	const int requested_number_of_channels = 3;
	int received_number_of_channels = 0;
	
	Image image{};
	image.buffer_ptr = stbi_load(rgb_image_path.c_str(), &image.width, &image.height, &received_number_of_channels, requested_number_of_channels);
	
	return image;
}
