
#include "gl/GLShader.h"
#include <sstream>

GLShader::GLShader(const std::string& vertex_source, const std::string& fragment_source){
	
	auto vertex_shader = compile(vertex_source, GL_VERTEX_SHADER);
	auto fragment_shader = compile(fragment_source, GL_FRAGMENT_SHADER);
	
	program = make_resource(glCreateProgram(), [](auto v){ glDeleteProgram(v); });
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);
	
	int success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if(!success){
		char info_log[512];
		glGetProgramInfoLog(program, 512, nullptr, info_log);
		std::stringstream ss;
		ss << "Shader program linking failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
	
}

GLShader::GLShader(const std::string& vertex_source, const std::string& geometry_source, const std::string& fragment_source){
	auto vertex_shader = compile(vertex_source, GL_VERTEX_SHADER);
	auto geometry_shader = compile(geometry_source, GL_GEOMETRY_SHADER);
	auto fragment_shader = compile(fragment_source, GL_FRAGMENT_SHADER);
	
	program = make_resource(glCreateProgram(), [](auto v){ glDeleteProgram(v); });
	glAttachShader(program, vertex_shader);
	glAttachShader(program, geometry_shader);
	glAttachShader(program, fragment_shader);
	glLinkProgram(program);
	
	int success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if(!success){
		char info_log[512];
		glGetProgramInfoLog(program, 512, nullptr, info_log);
		std::stringstream ss;
		ss << "Shader program linking failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
}

Resource<unsigned int> GLShader::compile(const std::string& source_code, const GLenum type){
	
	auto shader = make_resource(glCreateShader(type), [](auto v){ glDeleteShader(v); });
	
	const char* source_str = source_code.c_str();
	glShaderSource(shader, 1, &source_str, nullptr);
	glCompileShader(shader);
	
	int success = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if(!success){
		char info_log[512];
		glGetShaderInfoLog(shader, 512, nullptr, info_log);
		std::stringstream ss;
		ss << "Shader compilation failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
	
	return shader;
}

void GLShader::set_uniform(const std::string& name, int value) const {
	auto location = glGetUniformLocation(program, name.c_str());
	if (location == -1){
		return;
	}
	glUniform1i(location, value);
}

void GLShader::set_uniform(const std::string& name, float value) const {
	auto location = glGetUniformLocation(program, name.c_str());
	if (location == -1){
		return;
	}
	glUniform1f(location, value);
}

void GLShader::set_uniform(const std::string& name, bool value) const {
	set_uniform(name, static_cast<int>(value));
}

void GLShader::set_uniform(const std::string& name, const Eigen::Vector2f& value) const {
	auto location = glGetUniformLocation(program, name.c_str());
	if (location == -1){
		return;
	}
	glUniform2f(location, value(0), value(1));
}

void GLShader::set_uniform(const std::string& name, const Eigen::Vector3f& value) const {
	auto location = glGetUniformLocation(program, name.c_str());
	if (location == -1){
		return;
	}
	glUniform3f(location, value(0), value(1), value(2));
}

void GLShader::set_uniform(const std::string& name, const Eigen::Matrix4f& value) const {
	auto location = glGetUniformLocation(program, name.c_str());
	if (location == -1){
		return;
	}
	glUniformMatrix4fv(location, 1, GL_FALSE, value.data());
}

void GLShader::use() const {
	glUseProgram(program);
}
