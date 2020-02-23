#pragma once

#include <sstream>
#include <array>
#include <vector>
#include <cassert>


#include <cuda.h>
#include <cuda_runtime.h>

class CUDABuffer
{
public:
	CUDABuffer() : m_dev_ptr{nullptr}, m_buffer_size{0} {}
	
	explicit CUDABuffer(size_t size) : m_dev_ptr{nullptr}, m_buffer_size{0} {
		allocate(size);
	}
	
	~CUDABuffer() {
		deallocate();
	}
	
	CUDABuffer(CUDABuffer&& rhs) noexcept
	: m_dev_ptr{rhs.m_dev_ptr}, m_buffer_size{rhs.m_buffer_size} {
		// Make sure the destructor on rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_buffer_size = 0;
	}
	
	CUDABuffer(const CUDABuffer&) = delete; // Delete the copy constructor
	
	CUDABuffer& operator=(CUDABuffer&& rhs) noexcept {
		// Release any memory we currently own
		deallocate();
		
		// Take ownership of the memory of rhs object
		m_dev_ptr = rhs.m_dev_ptr;
		m_buffer_size = rhs.m_buffer_size;
		
		// Make sure the destructor on the rhs object does not deallocate our buffer
		rhs.m_dev_ptr = nullptr;
		rhs.m_buffer_size = 0;
		
		return *this;
	}
	
	CUDABuffer& operator=(const CUDABuffer&) = delete;
	
	[[nodiscard]] __host__ __device__ void* get_dev_ptr() const { return m_dev_ptr; }
	[[nodiscard]] __host__ __device__ CUdeviceptr get_cudev_ptr() const { return reinterpret_cast<CUdeviceptr>(m_dev_ptr); }
	[[nodiscard]] __host__ __device__ size_t get_size_in_bytes() const { return m_buffer_size; }
	
	void resize(size_t size) {
		if (size == m_buffer_size) {
			return;
		}
		
		deallocate();
		allocate(size);
	}
	
	void clear() {
		deallocate();
		m_buffer_size = 0;
	}
	
	template <typename T>
	void upload(const T& object) {
		if (m_buffer_size != sizeof(T)) { // Resize the CUDA buffer if it does not match the size of the object
			resize(sizeof(T));
		}
		
		auto result = cudaMemcpy(m_dev_ptr, reinterpret_cast<const void*>(&object), sizeof(T), cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << sizeof(T) << " bytes to CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void upload(const void* source_buffer, size_t size) {
		if (m_buffer_size != size) { // Resize the CUDA buffer if it does not match the size of the buffer
			resize(size);
		}
		
		auto result = cudaMemcpy(m_dev_ptr, source_buffer, size, cudaMemcpyHostToDevice);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << size << " bytes to CUDA buffer. CUDA Error: " << cudaGetErrorString(result);
			throw std::runtime_error(ss.str());
		}
	}
	
	template <typename T>
	void upload_async(const T& object, cudaStream_t cuda_stream) {
		if (m_buffer_size != sizeof(T)) { // Resize the CUDA buffer if it does not match the size of the object
			resize(sizeof(T));
		}
		
		auto result = cudaMemcpyAsync(m_dev_ptr, reinterpret_cast<const void*>(&object), sizeof(T), cudaMemcpyHostToDevice, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << sizeof(T) << " bytes to CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void upload_async(const void* source_buffer, size_t size, cudaStream_t cuda_stream) {
		if (m_buffer_size != size) { // Resize the CUDA buffer if it does not match the size of the buffer
			resize(size);
		}
		
		auto result = cudaMemcpyAsync(m_dev_ptr, source_buffer, size, cudaMemcpyHostToDevice, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to upload " << size << " bytes to CUDA buffer. CUDA Error: " << cudaGetErrorString(result);
			throw std::runtime_error(ss.str());
		}
	}
	
	template <typename T>
	void download(T& object) const {
		if (sizeof(T) != m_buffer_size) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download CUDA buffer of  " << m_buffer_size << " to a object of " << sizeof(T) << " bytes.\n";
			throw std::runtime_error(ss.str());
		}
		
		auto result = cudaMemcpy(reinterpret_cast<void*>(&object), m_dev_ptr, m_buffer_size, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_buffer_size << " bytes from CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download(void* destination_buffer) {
		auto result = cudaMemcpy(destination_buffer, m_dev_ptr, m_buffer_size, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_buffer_size << " bytes from CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	template <typename T>
	void download_async(T& object, cudaStream_t cuda_stream) const {
		if (sizeof(T) != m_buffer_size) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download CUDA buffer of  " << m_buffer_size << " to a object of " << sizeof(T) << " bytes.\n";
			throw std::runtime_error(ss.str());
		}
		
		auto result = cudaMemcpyAsync(reinterpret_cast<void*>(&object), m_dev_ptr, m_buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_buffer_size << " bytes from CUDA buffer.\n";
			throw std::runtime_error(ss.str());
		}
	}
	
	void download_async(void* destination_buffer, cudaStream_t cuda_stream) {
		auto result = cudaMemcpyAsync(destination_buffer, m_dev_ptr, m_buffer_size, cudaMemcpyDeviceToHost, cuda_stream);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to download " << m_buffer_size << " bytes from CUDA buffer.\n";
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
	size_t m_buffer_size;
	
	void allocate(size_t size) {
		
		if (size <= 0) {
			return;
		}
		
		auto result = cudaMalloc(&m_dev_ptr, size);
		if (result != cudaSuccess) {
			std::ostringstream ss;
			ss << "ERROR! Unable to allocate CUDA buffer of " << size << " bytes\n";
			throw std::runtime_error(ss.str());
		}
		
		m_buffer_size = size;
	}
	
	void deallocate() {
		if (m_dev_ptr != nullptr) {
			cudaFree(m_dev_ptr);
		}
	}
};

