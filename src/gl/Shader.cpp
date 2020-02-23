#include "gl/Shader.h"

#include <sstream>

#include "gl/GLDebug.h"
#include "util/Resource.h"


Toucan::Shader::Shader(const std::string& vertex_source, const std::string& fragment_source){
	
	auto vertex_shader = compile(vertex_source, GL_VERTEX_SHADER);
	auto fragment_shader = compile(fragment_source, GL_FRAGMENT_SHADER);
	
	m_program = make_resource<uint32_t>(
			[](auto& r) { r = glCreateProgram(); glCheckError(); },
			[](auto v) { glDeleteProgram(v); glCheckError(); }
	);
	glAttachShader(m_program, vertex_shader); glCheckError();
	glAttachShader(m_program, fragment_shader); glCheckError();
	glLinkProgram(m_program); glCheckError();
	
	int success = 0;
	glGetProgramiv(m_program, GL_LINK_STATUS, &success); glCheckError();
	if(!success){
		char info_log[512];
		glGetProgramInfoLog(m_program, 512, nullptr, info_log); glCheckError();
		std::stringstream ss;
		ss << "Shader program linking failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
	
}

Toucan::Shader::Shader(const std::string& vertex_source, const std::string& geometry_source, const std::string& fragment_source){
	auto vertex_shader = compile(vertex_source, GL_VERTEX_SHADER);
	auto geometry_shader = compile(geometry_source, GL_GEOMETRY_SHADER);
	auto fragment_shader = compile(fragment_source, GL_FRAGMENT_SHADER);
	
	m_program = make_resource<uint32_t>(
			[](auto& r) { r = glCreateProgram(); glCheckError(); },
			[](auto v){ glDeleteProgram(v); glCheckError(); }
	);
	glAttachShader(m_program, vertex_shader); glCheckError();
	glAttachShader(m_program, geometry_shader); glCheckError();
	glAttachShader(m_program, fragment_shader); glCheckError();
	glLinkProgram(m_program); glCheckError();
	
	int success = 0;
	glGetProgramiv(m_program, GL_LINK_STATUS, &success); glCheckError();
	if(!success){
		char info_log[512];
		glGetProgramInfoLog(m_program, 512, nullptr, info_log); glCheckError();
		std::stringstream ss;
		ss << "Shader program linking failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
}

Toucan::Resource<unsigned int> Toucan::Shader::compile(const std::string& source_code, const GLenum type){
	
	auto shader = make_resource<uint32_t>(
			[=](auto& r) { r = glCreateShader(type); glCheckError();},
			[](auto r){ glDeleteShader(r); glCheckError(); }
	);
	
	
	const char* source_str = source_code.c_str();
	glShaderSource(shader, 1, &source_str, nullptr); glCheckError();
	glCompileShader(shader); glCheckError();
	
	int success = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success); glCheckError();
	if(!success){
		char info_log[512];
		glGetShaderInfoLog(shader, 512, nullptr, info_log); glCheckError();
		std::stringstream ss;
		ss << "Shader compilation failed: " << info_log;
		throw std::runtime_error(ss.str());
	}
	
	return shader;
}

void Toucan::Shader::set_uniform(const std::string& name, int value) const {
	auto location = glGetUniformLocation(m_program, name.c_str()); glCheckError();
	if (location == -1){
		return;
	}
	glUniform1i(location, value); glCheckError();
}

void Toucan::Shader::set_uniform(const std::string& name, float value) const {
	auto location = glGetUniformLocation(m_program, name.c_str()); glCheckError();
	if (location == -1){
		return;
	}
	glUniform1f(location, value); glCheckError();
}

void Toucan::Shader::set_uniform(const std::string& name, bool value) const {
	set_uniform(name, static_cast<int>(value));
}

void Toucan::Shader::set_uniform(const std::string& name, const Eigen::Vector2f& value) const {
	auto location = glGetUniformLocation(m_program, name.c_str()); glCheckError();
	if (location == -1){
		return;
	}
	glUniform2f(location, value(0), value(1)); glCheckError();
}

void Toucan::Shader::set_uniform(const std::string& name, const Eigen::Vector3f& value) const {
	auto location = glGetUniformLocation(m_program, name.c_str()); glCheckError();
	if (location == -1){
		return;
	}
	glUniform3f(location, value(0), value(1), value(2)); glCheckError();
}

void Toucan::Shader::set_uniform(const std::string& name, const Eigen::Matrix4f& value) const {
	auto location = glGetUniformLocation(m_program, name.c_str()); glCheckError();
	if (location == -1){
		return;
	}
	glUniformMatrix4fv(location, 1, GL_FALSE, value.data()); glCheckError();
}

void Toucan::Shader::use() const {
	if (!m_program.is_empty()) {
		glUseProgram(m_program); glCheckError();
	}
}
