#pragma once

#include <utility>
#include <functional>
#include <type_traits>


namespace Toucan {

template<typename T>
class Resource
{
public:
	
	Resource() = default;
	
	Resource(T handle, std::function<void(T)> releaser)
			: m_enabled(true), m_handle(std::move(handle)), m_releaser(std::move(releaser))
	{
	}
	
	~Resource()
	{
		if (m_enabled) {
			m_releaser(std::move(m_handle));
		}
	}
	
	Resource(const Resource&) = delete;
	Resource& operator=(const Resource&) = delete;
	
	Resource(Resource&& rhs) noexcept
			:
			m_enabled(rhs.m_enabled),
			m_handle(std::move(rhs.m_handle)),
			m_releaser(std::move(rhs.m_releaser))
	{
		rhs.m_enabled = false;
	}
	
	Resource& operator=(Resource&& rhs) noexcept
	{
		if (m_enabled) {
			m_releaser(m_handle);
		}
		m_enabled = rhs.m_enabled;
		m_handle = std::move(rhs.m_handle);
		m_releaser = std::move(rhs.m_releaser);
		rhs.m_enabled = false;
		return *this;
	}
	
	operator T() const { return m_handle; } // NOLINT
	
	[[nodiscard]]  bool is_empty() const { return !m_enabled; }

private:
	bool m_enabled{false};
	T m_handle;
	std::function<void(T)> m_releaser;
};

template<typename T, typename L>
auto make_resource(T handle, L&& releaser)
{
	return Resource<T>(std::move(handle), std::forward<L>(releaser));
}

template<typename T, typename L>
auto make_resource(std::function<void(T&)> creator, L&& releaser)
{
	T handle;
	creator(handle);
	return Resource<T>(std::move(handle), std::forward<L>(releaser));
}

} // namespace Toucan


