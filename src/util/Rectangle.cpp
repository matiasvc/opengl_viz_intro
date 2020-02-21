#include "util/Rectangle.h"

Toucan::Rectangle::Rectangle(const Eigen::Vector2f& top_left, const Eigen::Vector2f& size)
: m_top_left{top_left}, m_size{size}
{ }

const Eigen::Vector2f& Toucan::Rectangle::get_top_left() const { return m_top_left; }

const Eigen::Vector2f& Toucan::Rectangle::get_size() const { return m_size; }
