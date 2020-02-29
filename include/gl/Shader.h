#pragma once

#include <string>
#include <cstdint>

#include <glad/glad.h>
#include <Eigen/Core>

#include "util/Resource.h"

namespace Toucan {

class Shader {
public:
	Shader() = default;
	Shader(const std::string& vertex_source, const std::string& fragment_source);
	Shader(const std::string& vertex_source, const std::string& geometry_source, const std::string& fragment_source);
	
	void set_uniform(const std::string& name, int value) const;
	void set_uniform(const std::string& name, float value) const;
	void set_uniform(const std::string& name, bool value) const;
	void set_uniform(const std::string& name, const Eigen::Vector2f& value) const;
	void set_uniform(const std::string& name, const Eigen::Vector3f& value) const;
	void set_uniform(const std::string& name, const Eigen::Matrix4f& value) const;
	
	void use() const;

private:
	Resource<uint32_t> compile(const std::string& source_code, GLenum type);
	
	Resource<uint32_t> m_program;
};

} // namespace Toucan
