#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "gaussian_blur.cl"
#define KERNEL_FUNC_1 "naive_blur"
#define KERNEL_FUNC_2a "smart_blur_verticle"
#define KERNEL_FUNC_2b "smart_blur_horizontal"

#define INPUT_FILE "bunnycity2.bmp"
#define OUTPUT_FILE_1 "output_naive.bmp"
#define OUTPUT_FILE_2 "output_smart.bmp"
#define NUM_ROUNDS 1000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmpfuncs.h"
#include <iostream>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_device_id default_device() {
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

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {

   cl_platform_id *platforms;
   cl_device_id dev;
   cl_device_id *devices;
   int err;
   cl_uint platformCount, deviceCount;
   char* value;
   size_t valueSize;
   cl_uint maxComputeUnits;
	
   clGetPlatformIDs(0, NULL, &platformCount);
   platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
   err = clGetPlatformIDs(platformCount, platforms, NULL);
   if (err < 0) {
	   perror("Couldn't identify a platform");
	   exit(1);
   }

   for (int i = 0; i < platformCount; i++) {

	   // get all devices
	   clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	   devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	   clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

	   // for each device print critical attributes
	   for (int j = 0; j < deviceCount; j++) {

		   // print device name
		   clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
		   value = (char*)malloc(valueSize);
		   clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
		   printf("%d.%d Device: %s\n", i +1, j + 1, value);
		   free(value);

		   // print parallel compute units
		   clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
			   sizeof(maxComputeUnits), &maxComputeUnits, NULL);
		   printf(" Parallel compute units: %d\n", maxComputeUnits);

		   std::cout << std::endl;
	   }

	   free(devices);

   }
   int platnum, devnum;
   std::cout << "Please enter device number (e.g. 1.1): ";
   std::cin >> platnum;
   getchar();
   std::cin >> devnum;
   std::cin.ignore(100, '\n');

   platnum--;
   devnum--;

   if (platnum+1 > platformCount) {
	   free(platforms);
	   return default_device();
   }

   clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
   devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
   clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
   
   if (devnum+1 > deviceCount) {
	   free(platforms);
	   free(devices);
	   return default_device();
   }
   
   dev = devices[devnum];

   free(devices);
   free(platforms);

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
   if(program_handle == NULL) {
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
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
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

void print_device(cl_device_id device) {
	char name_data[48];
	int err;
	/* Access device name */
	err = clGetDeviceInfo(device, CL_DEVICE_NAME,
		48 * sizeof(char), name_data, NULL);
	if (err < 0) {
		perror("Couldn't read extension data");
		exit(1);
	}

	printf("\nNAME: %s\n\n",
		name_data);
}

int main(int argc, char **argv) {

	/* Host/device data structures */
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel1, kernel2a, kernel2b;
	cl_int err;
	size_t global_size[2];
	cl_event evnt1, evnt2a, evnt2b;

	/* Image data */
	unsigned char* inputImage;
	unsigned char* outputinput;
	unsigned char* outputImage1;
	unsigned char* outputImage2;

	cl_image_format img_format;
	cl_mem input_image, output_image1, output_input, output_image2, buffer_dim;
	size_t origin[3], region[3];
	size_t width, height;
	int w, h;
	int dimension;

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
	outputImage1 = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);
	outputinput = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);
	outputImage2 = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);

	/* Create a device and context */
	device = create_device();

	print_device(device);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if (err < 0) {
		perror("Couldn't create a context");
		exit(1);
	}

	/* Build the program and create a kernel */
	program = build_program(context, device, PROGRAM_FILE);
	kernel1 = clCreateKernel(program, KERNEL_FUNC_1, &err);
	kernel2a = clCreateKernel(program, KERNEL_FUNC_2a, &err);
	kernel2b = clCreateKernel(program, KERNEL_FUNC_2b, &err);
	if (err < 0) {
		printf("Couldn't create a kernel: %d", err);
		exit(1);
	};

	/* Create image object */
	img_format.image_channel_order = CL_RGBA;
	img_format.image_channel_data_type = CL_UNORM_INT8;
	double sum1 = 0, sum2 = 0;
	
	for(int i = 0; i < NUM_ROUNDS; i++)
	{
		input_image = clCreateImage2D(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			&img_format, width, height, 0, (void*)inputImage, &err);
		output_image1 = clCreateImage2D(context,
			CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
		output_input = clCreateImage2D(context,
			CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
		output_image2 = clCreateImage2D(context,
			CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
		if (err < 0) {
			perror("Couldn't create the image object");
			exit(1);
		};

		/* Create kernel arguments */
		err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &input_image);
		err |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &output_image1);
		err |= clSetKernelArg(kernel1, 2, sizeof(cl_int), &dimension);
		err |= clSetKernelArg(kernel2a, 0, sizeof(cl_mem), &input_image);
		err |= clSetKernelArg(kernel2a, 1, sizeof(cl_mem), &output_input);
		err |= clSetKernelArg(kernel2a, 2, sizeof(cl_int), &dimension);
		err |= clSetKernelArg(kernel2b, 1, sizeof(cl_mem), &output_image2);
		err |= clSetKernelArg(kernel2b, 2, sizeof(cl_int), &dimension);
		if (err < 0) {
			printf("Couldn't set a kernel argument");
			exit(1);
		};

		/* Create a command queue */
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
		if (err < 0) {
			perror("Couldn't create a command queue");
			exit(1);
		};

	/* Enqueue kernel */
	global_size[0] = width; global_size[1] = height;
	err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global_size,
		NULL, 0, NULL, &evnt1);
	err = clEnqueueNDRangeKernel(queue, kernel2a, 2, NULL, global_size,
		NULL, 0, NULL, &evnt2a);
	if (err < 0) {
		perror("Couldn't enqueue the kernel");
		exit(1);
	}

	/* Read the image object */
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_image1, CL_TRUE, origin,
		region, 0, 0, (void*)outputImage1, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read from the image object");
		exit(1);
	}

	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	err = clEnqueueReadImage(queue, output_input, CL_TRUE, origin,
		region, 0, 0, (void*)outputinput, 0, NULL, NULL);
	if (err < 0) {
		perror("Couldn't read from the image object");
			exit(1);
		}

		output_input = clCreateImage2D(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			&img_format, width, height, 0, (void*)outputinput, &err);

		err = clSetKernelArg(kernel2b, 0, sizeof(cl_mem), &output_input);
		if (err < 0) {
			printf("Couldn't set a kernel argument");
			exit(1);
		};

		err = clEnqueueNDRangeKernel(queue, kernel2b, 2, NULL, global_size,
			NULL, 0, NULL, &evnt2b);
		if (err < 0) {
			perror("Couldn't enqueue the kernel");
			exit(1);
		}

		origin[0] = 0; origin[1] = 0; origin[2] = 0;
		region[0] = width; region[1] = height; region[2] = 1;
		err = clEnqueueReadImage(queue, output_image2, CL_TRUE, origin,
			region, 0, 0, (void*)outputImage2, 0, NULL, NULL);
		if (err < 0) {
			perror("Couldn't read from the image object");
			exit(1);
		}

		clWaitForEvents(1, &evnt1);
		clWaitForEvents(1, &evnt2a);
		clWaitForEvents(1, &evnt2b);

		clFinish(queue);

		cl_ulong time_start_1, time_start_2a, time_start_2b;
		cl_ulong time_end_1, time_end_2a, time_end_2b;

		clGetEventProfilingInfo(evnt1, CL_PROFILING_COMMAND_START, sizeof(time_start_1), &time_start_1, NULL);
		clGetEventProfilingInfo(evnt2a, CL_PROFILING_COMMAND_START, sizeof(time_start_2a), &time_start_2a, NULL);
		clGetEventProfilingInfo(evnt2b, CL_PROFILING_COMMAND_START, sizeof(time_start_2b), &time_start_2b, NULL);
		
		clGetEventProfilingInfo(evnt1, CL_PROFILING_COMMAND_END, sizeof(time_end_1), &time_end_1, NULL);
		clGetEventProfilingInfo(evnt2a, CL_PROFILING_COMMAND_END, sizeof(time_end_2a), &time_end_2a, NULL);
		clGetEventProfilingInfo(evnt2b, CL_PROFILING_COMMAND_END, sizeof(time_end_2b), &time_end_2b, NULL);


		double nanoSeconds_1 = time_end_1 - time_start_1;
		double nanoSeconds_2a = time_end_2a - time_start_2a;
		double nanoSeconds_2b = time_end_2b - time_start_2b;


		sum1 += nanoSeconds_1;
		sum2 += nanoSeconds_2a;
		sum2 += nanoSeconds_2b;

		clReleaseMemObject(input_image);
		clReleaseMemObject(output_image1);
		clReleaseMemObject(output_image2);
		clReleaseMemObject(output_input);
}

   /* Create output BMP file and write data */
   storeRGBImage(outputImage1, OUTPUT_FILE_1, h, w, INPUT_FILE);
   storeRGBImage(outputImage2, OUTPUT_FILE_2, h, w, INPUT_FILE);

   std::cout << "\nTested " << NUM_ROUNDS << " times:" << std::endl;
   printf("\tAverage naive Execution time is: %0.3f milliseconds \n", (sum1/NUM_ROUNDS) / 1000000.0);
   printf("\tAverage two pass Execution time is: %0.3f milliseconds \n", (sum2/NUM_ROUNDS) / 1000000.0);
   getchar();

   /* Deallocate resources */
   free(inputImage);
   free(outputImage1);
   free(outputinput);
   free(outputImage2);
   clReleaseKernel(kernel1);
   clReleaseKernel(kernel2a);
   clReleaseKernel(kernel2b);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
