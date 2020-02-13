#pragma once

#include "gl/GLLineDrawer3D.h"

class GLWorldGridDrawer
{
public:
	GLWorldGridDrawer();
	
	void draw(const Eigen::Matrix4f& camera_projection, const Transform& world_to_camera);
	
private:
	GLLineDrawer3D m_axis_lines_drawer;
	GLLineDrawer3D m_major_grid_lines_drawer;
	GLLineDrawer3D m_minor_grid_lines_drawer;
};


