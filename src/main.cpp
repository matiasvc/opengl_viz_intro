#include <iostream>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb/stb_image.h>

#include "util/Resource.h"
#include "gl/Shader.h"

#include "util/Transform.h"
#include "gl/PointDrawer3D.h"
#include "gl/LineDrawer3D.h"
#include "gl/WorldGridDrawer.h"
#include "gl/PrimitiveDrawer3D.h"
#include "gl/OrbitCamera.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "gl/WorldGridDrawer.h"
#include "gl/LineDrawer3D.h"
#include "gl/OrbitCamera.h"
#include "gl/PointDrawer3D.h"
#include "gl/PrimitiveDrawer3D.h"
#include "gl/ImageDrawer2D.h"

#include "ImageLoader.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void process_input(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

void init_imgui(GLFWwindow* window) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	
	const char* glsl_version = "#version 450";
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
}

int main() {
	if (glfwInit() != GLFW_TRUE) {
		std::cerr << "ERROR! Unable to initialize GLFW.\n";
		glfwTerminate();
		return -1;
	}
	
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	
	// Create GLFW Window
	Toucan::Resource<GLFWwindow*> window = Toucan::make_resource(
		glfwCreateWindow(800, 600, "OpenGL Visualizer", nullptr, nullptr),
		[](auto r){ glfwDestroyWindow(r); }
	);
	
	if (window == nullptr) {
		std::cerr << "ERROR! Unable to create GLFW window.\n";
		glfwTerminate();
		return -1;
	}
	
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	
	// Load OpenGL with glad
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
		std::cerr << "ERROR! Unable to load OpenGL.\n";
		glfwTerminate();
		return -1;
	}
	
	init_imgui(window);
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(5.0f,20.0f);
	std::vector<Toucan::PointDrawer3D::Point3D> points;
	for (int i = 0; i < 200; ++i) {
		points.emplace_back(Toucan::PointDrawer3D::Point3D{Eigen::Vector3f::Random() * 3.5f, Eigen::Vector4f(0.8f, 0.7f, 0.2f, 1.0f), distribution(generator), Toucan::PointDrawer3D::PointShape::Circle});
	}
	Toucan::PointDrawer3D point_drawer;
	point_drawer.set_data(points);
	
	std::vector<Toucan::LineDrawer3D::LineVertex> line_vertices;
	const Eigen::Vector4f point_color(1.0f, 0.0f, 0.0f, 1.0f);
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f::Zero(), point_color});
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f(1.0f, 0.0f, 0.0f), point_color});
	
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f::Zero(), point_color});
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f(0.0f, 1.0f, 0.0f), point_color});
	
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f::Zero(), point_color});
	line_vertices.emplace_back(Toucan::LineDrawer3D::LineVertex{Eigen::Vector3f(0.0f, 0.0f, 1.0f), point_color});
	
	Toucan::LineDrawer3D line_drawer;
	line_drawer.set_data(line_vertices, Toucan::LineDrawer3D::DrawMode::Lines, 3.0f);
	
	Toucan::OrbitCamera camera;
	
	Eigen::Vector3f pos = Eigen::Vector3f::Zero();
	
	Toucan::WorldGridDrawer world_grid_drawer;
	
	Toucan::PrimitiveDrawer3D primitive_drawer;
	
	primitive_drawer.set_data(
			Toucan::PrimitiveDrawer3D::Primitive3D{
					Eigen::Vector3f::Zero(),
					Eigen::Quaternionf::Identity(),
					Eigen::Vector3f::Ones(),
					Eigen::Vector4f(0.25f, 0.95f, 0.8f, 1.0f),
					Toucan::PrimitiveDrawer3D::PrimitiveShape::Sphere
	}
	);
	
	Eigen::Vector3f cube_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f cube_euler = Eigen::Vector3f::Zero();
	Eigen::Vector2f top_left = Eigen::Vector2f::Zero();
	Eigen::Vector2f size = Eigen::Vector2f(100.0f, 100.0f);
	
	Toucan::ImageDrawer2D image_drawer;
	ImageLoader image_loader("/home/matiasvc/datasets/rgbd_dataset_freiburg1_desk2/");
	
	while (!glfwWindowShouldClose(window)) {
		process_input(window);
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		glfwMakeContextCurrent(window);
		
		
		
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		int fb_width, fb_height;
		glfwGetFramebufferSize(window, &fb_width, &fb_height);
		
		Eigen::Matrix4f camera_projection = camera.get_camera_projection(fb_width, fb_height, 0.01f, 100.0f, 1500.0f);
		Transform camera_transform = camera.get_camera_transform();
		
		point_drawer.draw(camera_projection, camera_transform);
		line_drawer.draw(camera_projection, camera_transform, Transform(pos));
		world_grid_drawer.draw(camera_projection, camera_transform);
		
		auto image = image_loader.get_image();
		image_loader.next();
		Toucan::ImageDrawer2D::Image toucan_image{};
		toucan_image.buffer_ptr = image.buffer_ptr;
		toucan_image.format = Toucan::ImageDrawer2D::ImageFormat::RGB_U8;
		toucan_image.width = image.width;
		toucan_image.height = image.height;
		image_drawer.set_texture(toucan_image);
		image_drawer.draw(Toucan::Rectangle(top_left, size), fb_width, fb_height);
		
		const Eigen::Quaternionf cube_orient = Eigen::AngleAxisf(cube_euler.x(), Eigen::Vector3f::UnitX()) *
				Eigen::AngleAxisf(cube_euler.y(), Eigen::Vector3f::UnitY()) *
				Eigen::AngleAxisf(cube_euler.z(), Eigen::Vector3f::UnitZ());
		primitive_drawer.draw(camera_projection, camera_transform, Transform(cube_pos, cube_orient));
		
		ImGuiIO& io = ImGui::GetIO();
		// Right mouse-button drag
		if (ImGui::IsMouseDragging(0) && !ImGui::IsAnyWindowFocused())
		{
			ImVec2 vec = io.MouseDelta;
			const float rotateSpeed = 0.003f;
			camera.rotate(-vec.y*rotateSpeed, vec.x*rotateSpeed);
		}
		
		// Left mouse-button drag
		if (ImGui::IsMouseDragging(1) && !ImGui::IsAnyWindowFocused())
		{
			ImVec2 vec = io.MouseDelta;
			const float moveSpeed = 0.002f;
			camera.move(Eigen::Vector3f(vec.x*moveSpeed, 0.0f, vec.y*moveSpeed));
		}
		
		// Mouse wheel
		if (!ImGui::IsAnyWindowFocused())
		{
			const float scroll = io.MouseWheel;
			const float zoomSpeed = 0.5f;
			
			if (scroll != 0.0f)
			{
				camera.change_distance(-scroll*zoomSpeed);
			}
		}
		
		if (ImGui::Begin("Camera")) {
			ImGui::DragFloat3("Pos", cube_pos.data(), 0.01f, -1.0f, 1.0f);
			ImGui::DragFloat3("rot", cube_euler.data(), 0.01f, -1.0f, 1.0f);
			ImGui::DragFloat2("Top lef", top_left.data(), 1.0f, 0.0f, 100.0f);
		} ImGui::End();
		
		
		
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	return 0;
}
