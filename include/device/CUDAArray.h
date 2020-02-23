#pragma once

#include <sstream>
#include <vector>
#include <array>

#include <cuda.h>

template <typename T>
class CUDAArray
{
public:
	CUDAArray()
	: m_dev_ptr{nullptr}, m_number_of_elements{0} {}
	
	explicit CUDAArray(size_t number_of_elements)
	: m_dev_ptr{nullptr}, m_number_of_elements{0} {
		allocate(number_of_elements);
	}
	
	~CUDAArray() {
		deallocate();
	}
	
	CUDAArray(CUDAArray&& rhs) noexcept
	: m_dev_ptr{rhs.m_dev_ptr}, m_number_of_elements{rhs.m_number_of_elements} {
		// Make sure the destructor on the rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_number_of_elements = 0;
	}
	
	CUDAArray(const CUDAArray&) = delete; // Delete the copy constructor
	
	CUDAArray& operator=(CUDAArray&& rhs) noexcept {
		// Release any memory we currently own
		deallocate();
		
		// Take ownership of the memory of the rhs object
		m_dev_ptr = rhs.m_dev_ptr;
		m_number_of_elements = rhs.m_number_of_elements;
		
		// Make sure the destructor on the rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_number_of_elements = 0;
		
		return *this;
	}
	
	CUDAArray& operator=(const CUDAArray&) = delete;
	
	[[nodiscard]] __host__ __device__ T* get_dev_ptr() const { return reinterpret_cast<T*>(m_dev_ptr); }
	[[nodiscard]] __host__ __device__ CUdeviceptr get_cudev_ptr() const { return reinterpret_cast<CUdeviceptr>(m_dev_ptr); }
	[[nodiscard]] __host__ __device__ size_t get_size_in_bytes() const { return sizeof(T)*m_number_of_elements; }
	[[nodiscard]] __host__ __device__ size_t get_number_of_elements() const { return m_number_of_elements; }
	
	void resize(size_t number_of_elements) {
		if (number_of_elements == m_number_of_elements) {
			return;
		}
		
		deallocate();
		allocate(number_of_elements);
	}

	template <typename Allocator>
	void upload(const std::vector<T, Allocator>& vector) {
		if (m_number_of_elements != vector.size()) { // Resize the CUDA buffer if it does not match the size of the vector
			resize(vector.size());
		}
		auto result = cudaMemcpy(m_dev_ptr, reinterpret_cast<const void*>(vector.data()), sizeof(T)*vector.size(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << vector.size() << " elements to CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <size_t N>
	void upload(const std::array<T, N>& array) {
		if (m_number_of_elements != array.size()) { // Resize the CUDA buffer if it does not match the size of the vector
			resize(array.size());
		}
		auto result = cudaMemcpy(m_dev_ptr, reinterpret_cast<const void*>(array.data()), sizeof(T)*array.size(), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << array.size() << " elements to CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <typename Allocator>
	void upload_async(const std::vector<T, Allocator>& vector, cudaStream_t cuda_stream) {
		if (m_number_of_elements != vector.size()) { // Resize the CUDA buffer if it does not match the size of the vector
			resize(vector.size());
		}
		auto result = cudaMemcpyAsync(m_dev_ptr, reinterpret_cast<const void*>(vector.data()), sizeof(T)*vector.size(), cudaMemcpyHostToDevice, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << vector.size() << " elements to CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <size_t N>
	void upload_async(const std::array<T, N>& array, cudaStream_t cuda_stream) {
		if (m_number_of_elements != array.size()) { // Resize the CUDA buffer if it does not match the size of the vector
			resize(array.size());
		}
		auto result = cudaMemcpy(m_dev_ptr, reinterpret_cast<const void*>(array.data()), sizeof(T)*array.size(), cudaMemcpyHostToDevice, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << array.size() << " elements to CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download(std::vector<T>& vector) const {
		if (vector.size() != m_number_of_elements) { // Resize the vector if it does not match the CUDA buffer
			vector.resize(m_number_of_elements);
		}
		
		auto result = cudaMemcpy(vector.data(), m_dev_ptr, sizeof(T)*m_number_of_elements, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_number_of_elements << " elements from CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <size_t N>
	void download(std::array<T, N>& array) const {
		if (m_number_of_elements != N) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download a CUDA array with " << m_number_of_elements << " to a std::array with " << N << " elements.\n";
			throw std::runtime_error(ss.str());
		}
		
		auto result = cudaMemcpy(array.data(), m_dev_ptr, sizeof(T)*m_number_of_elements, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_number_of_elements << " elements from CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download_async(std::vector<T>& vector, cudaStream_t cuda_stream) const {
		if (vector.size() != m_number_of_elements) { // Resize the vector if it does not match the CUDA buffer
			vector.resize(m_number_of_elements);
		}
		
		auto result = cudaMemcpyAsync(vector.data(), m_dev_ptr, sizeof(T)*m_number_of_elements, cudaMemcpyDeviceToHost, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_number_of_elements << " elements from CUDA array.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <size_t N>
	void download_async(std::array<T, N>& array, cudaStream_t cuda_stream) const {
		if (m_number_of_elements != N) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download a CUDA array with " << m_number_of_elements << " to a std::array with " << N << " elements.\n";
			throw std::runtime_error(ss.str());
		}
		
		auto result = cudaMemcpyAsync(array.data(), m_dev_ptr, sizeof(T)*m_number_of_elements, cudaMemcpyDeviceToHost, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_number_of_elements << " elements from CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
private:
	void* m_dev_ptr;
	size_t m_number_of_elements;
	
	void allocate(size_t number_of_elements) {
		if (number_of_elements == 0) {
			return;
		}
		
		const size_t size_in_bytes = sizeof(T) * number_of_elements;
		auto result = cudaMalloc(&m_dev_ptr, size_in_bytes);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to allocate CUDA array of " << number_of_elements << " elements\n";
			throw std::runtime_error(ss.str());
		}
		
		m_number_of_elements = number_of_elements;
	}
	
	void deallocate() {
		if (m_dev_ptr != nullptr) {
			cudaFree(m_dev_ptr);
		}
	}
};