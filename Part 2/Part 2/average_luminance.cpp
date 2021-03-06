#define _CRT_SECURE_NO_WARNINGS
#define INPUT_FILE "lena.bmp"
#define OUTPUT_FILE "output.bmp"
#define PROGRAM_FILE "average_luminance.cl"

#define KERNEL_1 "image_to_data"
#define KERNEL_2a "reduction_vector"
#define KERNEL_2b "reduction_complete"

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
		exit(1);
	}

	return program;
}

double avg_lum(unsigned char* image, int size) {
	double avg = 0;
	for (int i = 0; i < size * 4; i += 4) {
		avg += ((image[i + 0] * 0.299) + (image[i + 1] * 0.587) + (image[i + 2] * 0.114));
	}
	avg /= size;
	return avg;
}

int main(int argc, char **argv) {

   /* Image data */
   unsigned char* inputImage;
   unsigned char* outputImage;

   cl_image_format img_format;
   cl_mem input_image, output_image;
   size_t origin[3], region[3];
   size_t width, height;
   int w, h;

   /* Open input file and read image data */
   inputImage = readRGBImage(INPUT_FILE, &w, &h);
   width = w;
   height = h;
   outputImage = (unsigned char*)malloc(sizeof(unsigned char)*w*h*4);
  
   std::cout << "Average luminance: " << avg_lum(inputImage, w*h) << std::endl;

   std::cout << "Please press ENTER enter to see parallel reduction results." << std::endl;
   getchar();
  

   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel vector_kernel, complete_kernel, transform_kernel;
   cl_command_queue queue;
   cl_int err;
   size_t loc_size, glob_size, global_size[2];

   img_format.image_channel_order = CL_RGBA;
   img_format.image_channel_data_type = CL_UNORM_INT8;

   /* Data and buffers */
   float *data = new float[w*h];
   float sum;
   cl_mem data_buffer, sum_buffer, image_data;


   /* Create device and determine local size */
   device = create_device();
   err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
	   sizeof(loc_size), &loc_size, NULL);
   if (err < 0) {
	   perror("Couldn't obtain device information");
	   exit(1);
   }

   /* Create a context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if (err < 0) {
	   perror("Couldn't create a context");
	   exit(1);
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);


   /* Create a command queue */
   queue = clCreateCommandQueue(context, device,
	   CL_QUEUE_PROFILING_ENABLE, &err);
   if (err < 0) {
	   perror("Couldn't create a command queue");
	   exit(1);
   };

   /* Create kernels */
   transform_kernel = clCreateKernel(program, KERNEL_1, &err);
   vector_kernel = clCreateKernel(program, KERNEL_2a, &err);
   complete_kernel = clCreateKernel(program, KERNEL_2b, &err);
   if (err < 0) {
	   perror("Couldn't create a kernel");
	   exit(1);
   };

   input_image = clCreateImage2D(context,
	   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	   &img_format, width, height, 0, (void*)inputImage, &err);
   image_data = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	   sizeof(float)*w*h, NULL, &err);
   if (err < 0) {
	   perror("Couldn't create a buffer");
	   getchar();
	   exit(1);
   };

   err = clSetKernelArg(transform_kernel, 0, sizeof(cl_mem), &input_image);
   err |= clSetKernelArg(transform_kernel, 1, sizeof(cl_mem), &image_data);
   err |= clSetKernelArg(transform_kernel, 2, sizeof(cl_int), &h);
   if (err < 0) {
	   perror("Couldn't create a kernel argument");
	   getchar();
	   exit(1);
   }

   global_size[0] = width; global_size[1] = height;
   err = clEnqueueNDRangeKernel(queue, transform_kernel, 2, NULL, global_size,
	   NULL, 0, NULL, NULL);

   /* Read the result */
   err = clEnqueueReadBuffer(queue, image_data, CL_TRUE, 0,
	   w * h * sizeof(float), data, 0, NULL, NULL);
   if (err < 0) {
	   perror("Couldn't read the buffer");
	   getchar();
	   exit(1);
   }

   data_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE |
	   CL_MEM_USE_HOST_PTR, w * h * sizeof(float), data, &err);
   sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	   sizeof(float), NULL, &err);
   if (err < 0) {
	   perror("Couldn't create a buffer");
	   exit(1);
   };

   /* Set arguments for vector kernel */
   err = clSetKernelArg(vector_kernel, 0, sizeof(cl_mem), &data_buffer);
   err |= clSetKernelArg(vector_kernel, 1, loc_size * 4 * sizeof(float), NULL);

   /* Set arguments for complete kernel */
   err = clSetKernelArg(complete_kernel, 0, sizeof(cl_mem), &data_buffer);
   err |= clSetKernelArg(complete_kernel, 1, loc_size * 4 * sizeof(float), NULL);
   err |= clSetKernelArg(complete_kernel, 2, sizeof(cl_mem), &sum_buffer);
   if (err < 0) {
	   perror("Couldn't create a kernel argument");
	   exit(1);
   }

   /* Enqueue kernels */
   glob_size = (w*h)/ 4;
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

   /* Wait for key press before exiting */
   std::cout << "Average luminance (found using parrellel reduction): " << sum/(h*w) << std::endl;
   getchar();

   /* Deallocate resources */
   clReleaseMemObject(sum_buffer);
   clReleaseMemObject(data_buffer);
   clReleaseKernel(vector_kernel);
   clReleaseKernel(complete_kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   
   return 0;
}
