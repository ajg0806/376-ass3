#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "bloom.cl"
#define KERNEL_1 "reduction_vector"
#define KERNEL_2 "reduction_complete"
#define KERNEL_3 "output_pass_threshold"
#define KERNEL_4a "smart_blur_verticle"
#define KERNEL_4b "smart_blur_horizontal"
#define KERNEL_5 "final_bloom_step"
#define INPUT_FILE "lena.bmp"
#define OUTPUT_FILE "output.bmp"
#define OUTPUT_FILE2 "output2.bmp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bmpfuncs.h"
#include <iostream>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
	cl_platform_id platform;
	cl_device_id dev;
	int err;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("Couldn't identify a platform");
		exit(1);
	}

	/* Access a device */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if (err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if (err < 0) {
		perror("Couldn't access any devices");
		exit(1);
	}

	return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	int err;

	/* Read program file and place content into buffer */
	program_handle = fopen(filename, "rb");
	if (program_handle == NULL) {
		perror("Couldn't find the program file");
		exit(1);
	}
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	/* Create program from file */
	program = clCreateProgramWithSource(ctx, 1,
		(const char**)&program_buffer, &program_size, &err);
	if (err < 0) {
		perror("Couldn't create the program");
		exit(1);
	}
	free(program_buffer);

	/* Build program */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		getchar();
		exit(1);
	}

	return program;
}

int main(int argc, char **argv) {

	/* Host/device data structures */
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel, vector_kernel, complete_kernel, kernel4a, kernel4b, kernel5;
	cl_int err;
	size_t global_size[2], loc_size, glob_size;

	/* Image data */
	unsigned char* inputImage;
	unsigned char* outputImage;

	cl_image_format img_format;
	cl_mem input_image, input_image2, output_image;
	size_t origin[3], region[3];
	size_t width, height;
	int w, h;
	int dimension;
	float thres;



	std::cout << "Please enter 3, 5 or 7: ";
	std::cin >> dimension;
	std::cin.ignore(100, '\n');
	//If you won't play by the rules, you can't play
	if (dimension != 3 && dimension != 5 && dimension != 7) {
		dimension = 3;
	}

	/* Open input file and read image data */
	inputImage = readRGBImage(INPUT_FILE, &w, &h);
	width = w;
	height = h;
	//outputinput = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);
	outputImage = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);

	/* Data and buffers */
	float *data = new float[w*h];
	float sum;
	cl_mem data_buffer, sum_buffer;

	/* Initialize data */
	for (int i = 0; i < (w*h * 4); i += 4) {
		data[i / 4] = 1.0f*((inputImage[i + 0] * 0.299) + (inputImage[i + 1] * 0.587) + (inputImage[i + 2] * 0.114));
	}

	/* Create a device and context */
	device = create_device();

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	/* Build the program and create a kernel */
	program = build_program(context, device, PROGRAM_FILE);
	vector_kernel = clCreateKernel(program, KERNEL_1, &err);
	complete_kernel = clCreateKernel(program, KERNEL_2, &err);
	kernel = clCreateKernel(program, KERNEL_3, &err);
	kernel4a = clCreateKernel(program, KERNEL_4a, &err);
	kernel4b = clCreateKernel(program, KERNEL_4b, &err);
	kernel5 = clCreateKernel(program, KERNEL_5, &err);
	if (err < 0) {
		printf("Couldn't create a kernel: %d", err);
		exit(1);
	};

	/* Create image object */
	img_format.image_channel_order = CL_RGBA;
	img_format.image_channel_data_type = CL_UNORM_INT8;

	data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
		CL_MEM_USE_HOST_PTR, w * h * sizeof(float), data, &err);
	sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		sizeof(float), NULL, &err);
	input_image = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&img_format, width, height, 0, (void*)inputImage, &err);
	output_image = clCreateImage2D(context,
		CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
	if (err < 0) {
		perror("Couldn't create the image object");
		exit(1);
	};

	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(loc_size), &loc_size, NULL);

	/* Set arguments for vector kernel */
	err = clSetKernelArg(vector_kernel, 0, sizeof(cl_mem), &data_buffer);
	err |= clSetKernelArg(vector_kernel, 1, loc_size * 4 * sizeof(float), NULL);
	/* Set arguments for complete kernel */
	err = clSetKernelArg(complete_kernel, 0, sizeof(cl_mem), &data_buffer);
	err |= clSetKernelArg(complete_kernel, 1, loc_size * 4 * sizeof(float), NULL);
	err |= clSetKernelArg(complete_kernel, 2, sizeof(cl_mem), &sum_buffer);

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);







	/* Create kernel arguments */



	/* Create a command queue */
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err < 0) {
		perror("Couldn't create a command queue");
		exit(1);
	};






	/* Enqueue kernel */

	glob_size = (w*h) / 4;
	err = clEnqueueNDRangeKernel(queue, vector_kernel, 1, NULL, &glob_size,
		&loc_size, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}

	/* Perform successive stages of the reduction */
	while (glob_size / loc_size > loc_size) {
		glob_size = glob_size / loc_size;
		err = clEnqueueNDRangeKernel(queue, vector_kernel, 1, NULL, &glob_size,
			&loc_size, 0, NULL, NULL);
		if (err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}
	}
	glob_size = glob_size / loc_size;
	err = clEnqueueNDRangeKernel(queue, complete_kernel, 1, NULL, &glob_size,
		NULL, 0, NULL, NULL);

	/* Read the result */
	err = clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0,
		sizeof(float), &sum, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read the buffer");
		exit(1);
	}


	std::cout << "Threshold: ";
	std::cin >> thres;
	std::cin.ignore(100, '\n');
	if (thres < 0)
		thres = sum / (w*h);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &thres);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}



	//Enque pass kernel, read from pass kernel
	global_size[0] = width; global_size[1] = height;
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size,
		NULL, 0, NULL, NULL);
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin,
		region, 0, 0, (void*)outputImage, 0, NULL, NULL);


	//Reset input_image buffer to have the output from the pass kernel
	input_image = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&img_format, width, height, 0, (void*)outputImage, &err);
	output_image = clCreateImage2D(context,
		CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
	//Pass it to the first pass blur kernel

	err = clSetKernelArg(kernel4a, 0, sizeof(cl_mem), &input_image);
	err |= clSetKernelArg(kernel4a, 1, sizeof(cl_mem), &output_image);
	err |= clSetKernelArg(kernel4a, 2, sizeof(cl_int), &dimension);
	if (err < 0) {
		printf("Couldn't set a kernel argument");
		exit(1);
	};
	//Enque first pass blur kernel 
	global_size[0] = width; global_size[1] = height;
	err = clEnqueueNDRangeKernel(queue, kernel4a, 2, NULL, global_size,
		NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}
	//Read from first pass blur kernel
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin,
		region, 0, 0, (void*)outputImage, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read from the image object");
		exit(1);
	}


	input_image = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&img_format, width, height, 0, (void*)outputImage, &err);
	output_image = clCreateImage2D(context,
		CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
	err = clSetKernelArg(kernel4b, 0, sizeof(cl_mem), &input_image);
	err |= clSetKernelArg(kernel4b, 1, sizeof(cl_mem), &output_image);
	err |= clSetKernelArg(kernel4b, 2, sizeof(cl_int), &dimension);
	if (err < 0) {
		printf("Couldn't set a kernel argument");
		exit(1);
	};
	
	err = clEnqueueNDRangeKernel(queue, kernel4b, 2, NULL, global_size,
		NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}
	//Read in from the second pass blur kernel
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin,
		region, 0, 0, (void*)outputImage, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read from the image object");
		exit(1);
	}





	/****************************************/
	input_image = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&img_format, width, height, 0, (void*)inputImage, &err);
	input_image2 = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&img_format, width, height, 0, (void*)outputImage, &err);
	output_image = clCreateImage2D(context,
		CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
	err = clSetKernelArg(kernel5, 0, sizeof(cl_mem), &input_image);
	err |= clSetKernelArg(kernel5, 1, sizeof(cl_mem), &input_image2);
	err |= clSetKernelArg(kernel5, 2, sizeof(cl_int), &output_image);
	if (err < 0) {
		printf("Couldn't set a kernel argument");
		exit(1);
	};

	err = clEnqueueNDRangeKernel(queue, kernel5, 2, NULL, global_size,
		NULL, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}
	//Read in from the second pass blur kernel
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin,
		region, 0, 0, (void*)outputImage, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read from the image object");
		exit(1);
	}
	/************************************************/



	/* Create output BMP file and write data */
	storeRGBImage(outputImage, OUTPUT_FILE, h, w, INPUT_FILE);
	/* Create output BMP file and write data */

	getchar();

	/* Deallocate resources */
	free(inputImage);
	free(outputImage);
	clReleaseMemObject(sum_buffer);
	clReleaseMemObject(data_buffer);

	clReleaseMemObject(input_image);
	clReleaseMemObject(output_image);
	clReleaseKernel(vector_kernel);
	clReleaseKernel(complete_kernel);
	clReleaseKernel(kernel);
	clReleaseKernel(kernel4a);
	clReleaseKernel(kernel4b);
	clReleaseKernel(kernel5);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return 0;
}
