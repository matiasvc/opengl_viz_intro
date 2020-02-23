#pragma once

#include <sstream>
#include <array>
#include <vector>
#include <cassert>


#include <cuda.h>
#include <cuda_runtime.h>

class PitchedCUDABuffer
{
public:
	PitchedCUDABuffer() : m_dev_ptr{nullptr}, m_element_size_in_bytes{0}, m_elements_per_row{0}, m_number_of_rows{0}, m_pitch_in_bytes{0} {}
	
	PitchedCUDABuffer(size_t element_size_in_bytes, size_t elements_per_row, size_t number_of_rows)
	: m_dev_ptr{nullptr}, m_element_size_in_bytes{0}, m_elements_per_row{0}, m_number_of_rows{0}, m_pitch_in_bytes{0} {
		allocate(element_size_in_bytes, elements_per_row, number_of_rows);
	}
	
	~PitchedCUDABuffer() {
		deallocate();
	}
	
	PitchedCUDABuffer(PitchedCUDABuffer&& rhs) noexcept
	: m_dev_ptr{rhs.m_dev_ptr}, m_element_size_in_bytes{0}, m_elements_per_row{0}, m_number_of_rows{0}, m_pitch_in_bytes{0} {
		// Make sure the destructor on rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_element_size_in_bytes = 0;
		rhs.m_elements_per_row = 0;
		rhs.m_number_of_rows = 0;
		rhs.m_pitch_in_bytes = 0;
	}
	
	PitchedCUDABuffer(const PitchedCUDABuffer&) = delete; // Delete the copy constructor
	
	PitchedCUDABuffer& operator=(PitchedCUDABuffer&& rhs) noexcept {
		// Release any memory we currently own
		deallocate();
		
		// Take ownership of the memory of rhs object
		m_dev_ptr = rhs.m_dev_ptr;
		m_element_size_in_bytes = rhs.m_element_size_in_bytes;
		m_elements_per_row = rhs.m_elements_per_row;
		m_number_of_rows = rhs.m_number_of_rows;
		m_pitch_in_bytes = rhs.m_pitch_in_bytes;
		
		// Make sure the destructor on the rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_element_size_in_bytes = 0;
		rhs.m_elements_per_row = 0;
		rhs.m_number_of_rows = 0;
		rhs.m_pitch_in_bytes = 0;
		
		return *this;
	}
	
	PitchedCUDABuffer& operator=(const PitchedCUDABuffer&) = delete;
	
	[[nodiscard]] void* get_dev_ptr() const { return m_dev_ptr; }
	[[nodiscard]] CUdeviceptr get_cudev_ptr() const { return reinterpret_cast<CUdeviceptr>(m_dev_ptr); }
	[[nodiscard]] size_t get_element_size_in_bytes() const { return m_element_size_in_bytes; }
	[[nodiscard]] size_t get_pitch_in_bytes() const { return m_pitch_in_bytes; }
	[[nodiscard]] size_t get_elements_per_row() const { return m_elements_per_row; }
	[[nodiscard]] size_t get_number_of_rows() const { return m_number_of_rows; }
	
	void resize(size_t element_size_in_bytes, size_t elements_per_row, size_t number_of_rows) {
		if (element_size_in_bytes*elements_per_row == m_element_size_in_bytes*m_elements_per_row and number_of_rows == m_number_of_rows) {
			return;
		}
		
		deallocate();
		allocate(element_size_in_bytes, elements_per_row, number_of_rows);
	}
	
	void clear() {
		deallocate();
		m_element_size_in_bytes = 0;
		m_elements_per_row = 0;
		m_number_of_rows = 0;
		m_pitch_in_bytes = 0;
	}
	
	void upload(const void* source_buffer, size_t element_size_in_bytes, size_t elements_per_row, size_t number_of_rows) {
		if (element_size_in_bytes*elements_per_row != m_element_size_in_bytes*m_elements_per_row or number_of_rows != m_number_of_rows) {
			resize(element_size_in_bytes, elements_per_row, number_of_rows);
		}
		
		const auto source_width_in_bytes = element_size_in_bytes * m_elements_per_row;
		auto result = cudaMemcpy2D(m_dev_ptr, m_pitch_in_bytes, source_buffer, source_width_in_bytes, source_width_in_bytes, number_of_rows, cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload to a pitched CUDA buffer, with a width of " << elements_per_row << " elements, a height of " << number_of_rows << "elements, and a element size of" << element_size_in_bytes << "bytes\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void upload_async(const void* source_buffer, size_t element_size_in_bytes, size_t elements_per_row, size_t number_of_rows, cudaStream_t cuda_stream) {
		if (element_size_in_bytes*elements_per_row != m_element_size_in_bytes*m_elements_per_row or number_of_rows != m_number_of_rows) {
			resize(element_size_in_bytes, elements_per_row, number_of_rows);
		}
		
		const auto source_width_in_bytes = element_size_in_bytes * m_elements_per_row;
		auto result = cudaMemcpy2DAsync(m_dev_ptr, m_pitch_in_bytes, source_buffer, source_width_in_bytes, source_width_in_bytes, number_of_rows, cudaMemcpyHostToDevice, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload to a pitched CUDA buffer, with a width of " << elements_per_row << " elements, a height of " << number_of_rows << "elements, and a element size of" << element_size_in_bytes << "bytes\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download(void* destination_buffer) {
		const auto width_in_bytes = m_element_size_in_bytes * m_elements_per_row;
		auto result = cudaMemcpy2D(destination_buffer, width_in_bytes, m_dev_ptr, m_pitch_in_bytes, width_in_bytes, m_number_of_rows, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download a pitched CUDA buffer, with a width of " << m_elements_per_row << " elements, a height of " << m_number_of_rows << "elements, and a element size of" << m_element_size_in_bytes << "bytes\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download_async(void* destination_buffer, cudaStream_t cuda_stream) {
		const auto width_in_bytes = m_element_size_in_bytes * m_elements_per_row;
		auto result = cudaMemcpy2DAsync(destination_buffer, width_in_bytes, m_dev_ptr, m_pitch_in_bytes, width_in_bytes, m_number_of_rows, cudaMemcpyDeviceToHost, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download a pitched CUDA buffer, with a width of " << m_elements_per_row << " elements, a height of " << m_number_of_rows << "elements, and a element size of" << m_element_size_in_bytes << "bytes\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	// Disallow uploading or downloading of std::vector, as it is almost certainly a mistake, and it will not upload the contained data. You should use CUDAArray instead.
	template <typename T>
	void upload(const std::vector<T>&) = delete;
	
	template <typename T>
	void download(const std::vector<T>&) = delete;
	
	// Uploading std::array would work, but CUDAArray is better suited, so we disallow them to avoid mistakes.
	template <typename T, size_t N>
	void upload(const std::array<T, N>&) = delete;
	
	template <typename T, size_t N>
	void download(const std::array<T, N>&) = delete;
	
	
private:
	void* m_dev_ptr;
	size_t m_element_size_in_bytes, m_elements_per_row, m_number_of_rows;
	size_t m_pitch_in_bytes;
	
	void allocate(size_t element_size_in_bytes, size_t elements_per_row, size_t number_of_rows) {
		
		if (element_size_in_bytes <= 0 or elements_per_row <= 0 or number_of_rows <= 0) {
			return;
		}
		
		auto result = cudaMallocPitch(&m_dev_ptr, &m_pitch_in_bytes, element_size_in_bytes*elements_per_row, number_of_rows);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to allocate pitched CUDA buffer, with a width of " << elements_per_row << " elements, a height of " << number_of_rows << "elements, and a element size of" << element_size_in_bytes << "bytes\n";
			throw std::runtime_error(ss.str());
		}
		m_element_size_in_bytes = element_size_in_bytes;
		m_elements_per_row = elements_per_row;
		m_number_of_rows = number_of_rows;
	}
	
	void deallocate() {
		if (m_dev_ptr != nullptr) {
			cudaFree(m_dev_ptr);
		}
	}
};

