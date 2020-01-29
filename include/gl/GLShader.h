#pragma once

#include <glad/glad.h>
#include <eigen3/Eigen/Core>
#include <string>

#include "util/Resource.h"

class GLShader
{
public:
	GLShader(const std::string& vertex_source, const std::string& fragment_source);
	GLShader(const std::string& vertex_source, const std::string& geometry_source, const std::string& fragment_source);
	
	void set_uniform(const std::string& name, int value) const;
	void set_uniform(const std::string& name, float value) const;
	void set_uniform(const std::string& name, bool value) const;
	void set_uniform(const std::string& name, const Eigen::Vector2f& value) const;
	void set_uniform(const std::string& name, const Eigen::Vector3f& value) const;
	void set_uniform(const std::string& name, const Eigen::Matrix4f& value) const;
	
	void use() const;

private:
	Resource<unsigned int> compile(const std::string& source_code, GLenum type);
	Resource<unsigned int> program;
};

