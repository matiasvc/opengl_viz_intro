#include <iostream>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "util/Resource.h"
#include "gl/Shader.h"

#include "gl/ImageDrawer2D.h"
#include <gl/CUDAImageDrawer2D.h>

#include "device/image_gradient.h"
#include "device/image_grayscale.h"
#include "device/image_corner.h"
#include "device/PitchedCUDABuffer.h"
#include "device/CUDABuffer.h"
#include "device/CUDAArray.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

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
		glfwCreateWindow(1400, 1600, "OpenGL Visualizer", nullptr, nullptr),
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
	
	Toucan::CUDAImageDrawer2D cuda_color_image_drawer;
	Toucan::CUDAImageDrawer2D cuda_grayscale_image_drawer;
	Toucan::CUDAImageDrawer2D cuda_horizontal_gradient_image_drawer;
	Toucan::CUDAImageDrawer2D cuda_vertical_gradient_image_drawer;
	Toucan::CUDAImageDrawer2D cuda_corner_image_drawer;
	
	
	ImageLoader image_loader("/home/matiasvc/datasets/rgbd_dataset_freiburg3_long_office_household/");
	
	PitchedCUDABuffer color_image;
	color_image.resize(4*sizeof(uint8_t), 640, 480);
	
	PitchedCUDABuffer grayscale_image;
	grayscale_image.resize(sizeof(uint8_t), 640, 480);
	
	PitchedCUDABuffer horizontal_gradient_image;
	horizontal_gradient_image.resize(sizeof(int16_t), 640, 480);
	
	PitchedCUDABuffer vertical_gradient_image;
	vertical_gradient_image.resize(sizeof(int16_t), 640, 480);
	
	PitchedCUDABuffer corner_response_image;
	corner_response_image.resize(sizeof(float), 640, 480);
	
	CUDABuffer number_of_points_buffer;
	number_of_points_buffer.resize(sizeof(uint32_t));
	CUDAArray<CornerPoint> corner_points_buffer;
	corner_points_buffer.resize(500);
	
	while (!glfwWindowShouldClose(window)) {
		process_input(window);
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		glfwMakeContextCurrent(window);
		
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		Eigen::Vector2i framebuffer_size;
		glfwGetFramebufferSize(window, &framebuffer_size.x(), &framebuffer_size.y());
		
		
		image_loader.get_image(color_image);
		image_loader.next();
		
		compute_grayscale(color_image, grayscale_image);
		compute_gradient(grayscale_image, horizontal_gradient_image, vertical_gradient_image);
		compute_corners(horizontal_gradient_image, vertical_gradient_image, corner_response_image, corner_points_buffer, number_of_points_buffer);
		
		uint32_t number_of_points = 0;
		number_of_points_buffer.download(number_of_points);
		
		std::cout << number_of_points << '\n';
		
		cuda_color_image_drawer.set_image(Toucan::CUDAImageDrawer2D::CUDAImage{
			.dev_buffer_ptr = color_image.get_dev_ptr(),
			.format = Toucan::CUDAImageDrawer2D::ImageFormat::RGBX_U8,
			.width_in_pixels = static_cast<uint32_t>(color_image.get_elements_per_row()),
			.height_in_pixels = static_cast<uint32_t>(color_image.get_number_of_rows()),
			.pixel_size_in_bytes = static_cast<uint32_t>(color_image.get_element_size_in_bytes()),
			.pitch_in_bytes = static_cast<uint32_t>(color_image.get_pitch_in_bytes())
		});
		cuda_color_image_drawer.draw(framebuffer_size, Toucan::Rectangle(Eigen::Vector2f(0.0f, 0.0f), Eigen::Vector2f(512.0f, 350.0f)));
		
		cuda_grayscale_image_drawer.set_image(Toucan::CUDAImageDrawer2D::CUDAImage{
				.dev_buffer_ptr = grayscale_image.get_dev_ptr(),
				.format = Toucan::CUDAImageDrawer2D::ImageFormat::R_U8,
				.width_in_pixels = static_cast<uint32_t>(grayscale_image.get_elements_per_row()),
				.height_in_pixels = static_cast<uint32_t>(grayscale_image.get_number_of_rows()),
				.pixel_size_in_bytes = static_cast<uint32_t>(grayscale_image.get_element_size_in_bytes()),
				.pitch_in_bytes = static_cast<uint32_t>(grayscale_image.get_pitch_in_bytes())
		});
		cuda_grayscale_image_drawer.draw(framebuffer_size, Toucan::Rectangle(Eigen::Vector2f(0.0f, 350.0f), Eigen::Vector2f(512.0f, 350.0f)));
		
		cuda_horizontal_gradient_image_drawer.set_image(Toucan::CUDAImageDrawer2D::CUDAImage{
				.dev_buffer_ptr = horizontal_gradient_image.get_dev_ptr(),
				.format = Toucan::CUDAImageDrawer2D::ImageFormat::R_S16,
				.width_in_pixels = static_cast<uint32_t>(horizontal_gradient_image.get_elements_per_row()),
				.height_in_pixels = static_cast<uint32_t>(horizontal_gradient_image.get_number_of_rows()),
				.pixel_size_in_bytes = static_cast<uint32_t>(horizontal_gradient_image.get_element_size_in_bytes()),
				.pitch_in_bytes = static_cast<uint32_t>(horizontal_gradient_image.get_pitch_in_bytes())
		});
		cuda_horizontal_gradient_image_drawer.draw(framebuffer_size, Toucan::Rectangle(Eigen::Vector2f(0.0f, 2*350.0f), Eigen::Vector2f(512.0f, 350.0f)));
		
		cuda_vertical_gradient_image_drawer.set_image(Toucan::CUDAImageDrawer2D::CUDAImage{
				.dev_buffer_ptr = vertical_gradient_image.get_dev_ptr(),
				.format = Toucan::CUDAImageDrawer2D::ImageFormat::R_S16,
				.width_in_pixels = static_cast<uint32_t>(vertical_gradient_image.get_elements_per_row()),
				.height_in_pixels = static_cast<uint32_t>(vertical_gradient_image.get_number_of_rows()),
				.pixel_size_in_bytes = static_cast<uint32_t>(vertical_gradient_image.get_element_size_in_bytes()),
				.pitch_in_bytes = static_cast<uint32_t>(vertical_gradient_image.get_pitch_in_bytes())
		});
		cuda_vertical_gradient_image_drawer.draw(framebuffer_size, Toucan::Rectangle(Eigen::Vector2f(0.0f, 3*350.0f), Eigen::Vector2f(512.0f, 350.0f)));
		
		cuda_corner_image_drawer.set_image(Toucan::CUDAImageDrawer2D::CUDAImage{
				.dev_buffer_ptr = corner_response_image.get_dev_ptr(),
				.format = Toucan::CUDAImageDrawer2D::ImageFormat::R_F32,
				.width_in_pixels = static_cast<uint32_t>(corner_response_image.get_elements_per_row()),
				.height_in_pixels = static_cast<uint32_t>(corner_response_image.get_number_of_rows()),
				.pixel_size_in_bytes = static_cast<uint32_t>(corner_response_image.get_element_size_in_bytes()),
				.pitch_in_bytes = static_cast<uint32_t>(corner_response_image.get_pitch_in_bytes())
		});
		cuda_corner_image_drawer.draw(framebuffer_size, Toucan::Rectangle(Eigen::Vector2f(512.0f, 0.0f), Eigen::Vector2f(512.0f, 350.0f)));
		
		ImGuiIO& io = ImGui::GetIO();
		
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	return 0;
}
