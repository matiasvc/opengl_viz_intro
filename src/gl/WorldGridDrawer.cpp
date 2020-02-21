#include "gl/WorldGridDrawer.h"
#include "gl/LineDrawer3D.h"

#include <vector>

Toucan::WorldGridDrawer::WorldGridDrawer() {
	const float grid_extent = 50.0f;
	
	std::vector<LineDrawer3D::LineVertex> axis_lines;
	
	const Eigen::Vector4f x_axis_color = Eigen::Vector4f(1.0, 0.0f, 0.0f, 0.25f);
	axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(0.0f, 0.0f, 0.0f), x_axis_color});
	axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(grid_extent, 0.0f, 0.0f), x_axis_color});
	
	const Eigen::Vector4f y_axis_color = Eigen::Vector4f(0.0, 0.0f, 1.0f, 0.25f);
	axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(0.0f, 0.0f, 0.0f), y_axis_color});
	axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(0.0f, 0.0f, grid_extent), y_axis_color});
	
	m_axis_lines_drawer.set_data(axis_lines, LineDrawer3D::DrawMode::Lines, 2.0f);
	
	std::vector<LineDrawer3D::LineVertex> major_axis_lines;
	const Eigen::Vector4f major_axis_color = Eigen::Vector4f(0.75f, 0.75f, 0.75f, 0.25f);
	
	const float major_axis_step = 5.0f;
	for (int major_axis_index = -10; major_axis_index <= 10; ++major_axis_index) {
		// X-axis
		if (major_axis_index != 0) {
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(-grid_extent, 0.0f, major_axis_step * static_cast<float>(major_axis_index)), major_axis_color});
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(grid_extent, 0.0f, major_axis_step * static_cast<float>(major_axis_index)), major_axis_color});
		}
		else
		{
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(-grid_extent, 0.0f, major_axis_step * static_cast<float>(major_axis_index)), major_axis_color});
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(0, 0.0f, major_axis_step * static_cast<float>(major_axis_index)), major_axis_color});
			
		}
		
		// Z-axis
		if (major_axis_index != 0) {
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(major_axis_step * static_cast<float>(major_axis_index), 0.0f, -grid_extent), major_axis_color});
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(major_axis_step * static_cast<float>(major_axis_index), 0.0f, grid_extent), major_axis_color});
		}
		else
		{
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(major_axis_step * static_cast<float>(major_axis_index), 0.0f, -grid_extent), major_axis_color});
			major_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(major_axis_step * static_cast<float>(major_axis_index), 0.0f, 0.0f), major_axis_color});
			
		}
	}
	
	m_major_grid_lines_drawer.set_data(major_axis_lines, LineDrawer3D::DrawMode::Lines, 2.0f);
	
	std::vector<LineDrawer3D::LineVertex> minor_axis_lines;
	const Eigen::Vector4f minor_axis_color = Eigen::Vector4f(0.5f, 0.5f, 0.5f, 0.25f);
	
	const float minor_axis_step = 1.0f;
	for (int minor_axis_index = -50; minor_axis_index <= 50; ++minor_axis_index) {
		if (minor_axis_index % 5 == 0) { continue; }
		
		// X-axis
		if (minor_axis_index != 0) {
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(-grid_extent, 0.0f, minor_axis_step * static_cast<float>(minor_axis_index)), minor_axis_color});
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(grid_extent, 0.0f, minor_axis_step * static_cast<float>(minor_axis_index)), minor_axis_color});
		}
		else
		{
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(-grid_extent, 0.0f, minor_axis_step * static_cast<float>(minor_axis_index)), minor_axis_color});
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(0, 0.0f, minor_axis_step * static_cast<float>(minor_axis_index)), minor_axis_color});
			
		}
		
		// Z-axis
		if (minor_axis_index != 0) {
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(minor_axis_step * static_cast<float>(minor_axis_index), 0.0f, -grid_extent), minor_axis_color});
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(minor_axis_step * static_cast<float>(minor_axis_index), 0.0f, grid_extent), minor_axis_color});
		}
		else
		{
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(minor_axis_step * static_cast<float>(minor_axis_index), 0.0f, -grid_extent), minor_axis_color});
			minor_axis_lines.emplace_back(LineDrawer3D::LineVertex{Eigen::Vector3f(minor_axis_step * static_cast<float>(minor_axis_index), 0.0f, 0.0f), minor_axis_color});
			
		}
	}
	
	m_minor_grid_lines_drawer.set_data(minor_axis_lines, LineDrawer3D::DrawMode::Lines, 1.0f);
}


void Toucan::WorldGridDrawer::draw(const Eigen::Matrix4f &camera_projection, const Transform &world_to_camera) {
	m_axis_lines_drawer.draw(camera_projection, world_to_camera);
	m_major_grid_lines_drawer.draw(camera_projection, world_to_camera);
	m_minor_grid_lines_drawer.draw(camera_projection, world_to_camera);
}
