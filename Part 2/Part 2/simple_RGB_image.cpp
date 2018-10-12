#define _CRT_SECURE_NO_WARNINGS
#define INPUT_FILE "lena.bmp"
#define OUTPUT_FILE "output.bmp"

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
  
   std::cout << avg_lum(inputImage, w*h) << std::endl;


   printf("Done.");
   getchar();

   /* Deallocate resources */
   free(inputImage);
   free(outputImage);
  
   return 0;
}
