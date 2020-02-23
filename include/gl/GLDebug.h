#pragma once

#ifdef NDEBUG // Release

#define glCheckError() ((void)0)

#else // Debug

#include <string>
#include <iostream>
#include <glad/glad.h>

inline void glCheckError_(const char *file, int line) {
	std::ostringstream ss;
	ss << "OpenGL Error: \n";
	GLenum errorCode;
	
	bool error = false;
	
	while ((errorCode = glGetError()) != GL_NO_ERROR) {
		error = true;
		std::string error;
		switch (errorCode)
		{
			case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
			case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
			case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
			case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
			case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
			case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
			case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
			default:                               error = "UNKNOWN_ERROR"; break;
		}
		ss << "\t- " << error << " | " << file << " (line: " << line << ")\n";
	}
	if (error) {
		throw std::runtime_error(ss.str());
	}
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

#endif

