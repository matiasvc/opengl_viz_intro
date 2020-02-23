#pragma once

#include <cuda_runtime.h>

#if NDEBUG // Release

#define CUDA_CHECK(cudaOp) (cudaOp)

#else // Debug

#include <sstream>

#define CUDA_CHECK(cudaOp) \
do { \
	cudaError_t err = (cudaOp); \
	if (err != cudaSuccess) { \
		std::ostringstream ostr; \
		ostr << "CUDA Error in " << #cudaOp << __FILE__ << " file " << __LINE__ << " line : " << "Code: " << err << " = " << cudaGetErrorString(err); \
		throw std::runtime_error(ostr.str()); \
	} \
} while (0)


#endif // NDEBUG

#define CUDA_SYNC_CHECK() \
do { \
	cudaDeviceSynchronize(); \
	cudaError_t error = cudaGetLastError(); \
	if( error != cudaSuccess ) { \
		fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
		exit( 2 ); \
	} \
} while (0)
