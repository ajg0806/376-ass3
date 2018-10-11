#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "simple_conv.cl"
#define KERNEL_FUNC "simple_conv"

#define INPUT_FILE "lena.bmp"
#define OUTPUT_FILE_1 "output_naive.bmp"
#define OUTPUT_FILE_2 "output_smart.bmp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bmpfuncs.h"
#include <iostream>
using namespace std;

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
   cl_device_id dev, dev2;
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

		   cout << endl;
	   }

	   free(devices);

   }
   int platnum, devnum;
   cout << "Please enter device number (e.g. 1.1): ";
   cin >> platnum;
   getchar();
   cin >> devnum;
   cin.ignore(100, '\n');

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
   cl_kernel kernel;
   cl_int err;
   size_t global_size[2];

   /* Image data */
   unsigned char* inputImage;
   unsigned char* outputImage1;
   unsigned char* outputImage2;

   cl_image_format img_format;
   cl_mem input_image, output_image1, output_image2, buffer_dim;
   size_t origin[3], region[3];
   size_t width, height;
   int w, h;
   int dimension;

   cout << "Please enter 3, 5 or 7: ";
   cin >> dimension;
   cin.ignore(100, '\n');

   /* Open input file and read image data */
   inputImage = readRGBImage(INPUT_FILE, &w, &h);
   width = w;
   height = h;
   outputImage1 = (unsigned char*)malloc(sizeof(unsigned char)*w*h*4);
   outputImage2 = (unsigned char*)malloc(sizeof(unsigned char)*w*h * 4);

   /* Create a device and context */
   device = create_device();

   print_device(device);

   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }

   /* Build the program and create a kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %d", err);
      exit(1);
   };

   /* Create image object */
   img_format.image_channel_order = CL_RGBA;
   img_format.image_channel_data_type = CL_UNORM_INT8;

   input_image = clCreateImage2D(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         &img_format, width, height, 0, (void*)inputImage, &err);
   output_image1 = clCreateImage2D(context, 
         CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
   output_image2 = clCreateImage2D(context,
	   CL_MEM_WRITE_ONLY, &img_format, width, height, 0, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the image object");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image1);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &dimension);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &output_image2);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   }; 

   /* Create a command queue */
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Enqueue kernel */
   global_size[0] = width; global_size[1] = height;
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, 
         NULL, 0, NULL, NULL);  
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the image object */
   origin[0] = 0; origin[1] = 0; origin[2] = 0;
   region[0] = width; region[1] = height; region[2] = 1;
   err = clEnqueueReadImage(queue, output_image1, CL_TRUE, origin, 
         region, 0, 0, (void*)outputImage1, 0, NULL, NULL);
   err = clEnqueueReadImage(queue, output_image2, CL_TRUE, origin,
	   region, 0, 0, (void*)outputImage2, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read from the image object");
      exit(1);   
   }

   /* Create output BMP file and write data */
   storeRGBImage(outputImage1, OUTPUT_FILE_1, h, w, INPUT_FILE);
   storeRGBImage(outputImage2, OUTPUT_FILE_2, h, w, INPUT_FILE);

   printf("Done.");
   getchar();

   /* Deallocate resources */
   free(inputImage);
   free(outputImage1);
   free(outputImage2);
   clReleaseMemObject(input_image);
   clReleaseMemObject(output_image1);
   clReleaseMemObject(output_image2);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
