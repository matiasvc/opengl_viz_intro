#pragma once

#include "gl/LineDrawer3D.h"
#include "LineDrawer3D.h"

namespace Toucan {

class WorldGridDrawer {
public:
	WorldGridDrawer();
	
	void draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera);

private:
	LineDrawer3D m_axis_lines_drawer;
	LineDrawer3D m_major_grid_lines_drawer;
	LineDrawer3D m_minor_grid_lines_drawer;
};

} // namespace Toucan
