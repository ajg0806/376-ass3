__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__constant float SmartFilter1[3] = {0.27901, 0.44198, 0.27901};
__constant float SmartFilter2[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};
__constant float SmartFilter3[7] = {0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598};


__kernel void smart_blur_verticle(read_only image2d_t src_image,
					write_only image2d_t dst_image, int dim) {


   /* Get work-item’s row and column position */
   int column = get_global_id(0); 
   int row = get_global_id(1);

   /* Accumulated pixel value */
   float4 sum = (float4)(0.0);

   /* Filter's current index */
   int filter_index =  0;

   int2 coord;
   float4 pixel;


   int start = 0 - (int)floor(dim/2.0f);
   int end = 0 + (int)floor(dim/2.0f);

      /* Iterate over the rows */
   for(int i = start; i <= end; i++) {
	  coord.y =  row + i;
	  coord.x = column;

	  	/* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * SmartFilter1[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * SmartFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * SmartFilter3[filter_index++];

   }

	  coord = (int2)(column, row); 
	  write_imagef(dst_image, coord, sum);
}

__kernel void smart_blur_horizontal(read_only image2d_t src_image,
					write_only image2d_t dst_image, int dim) {


   /* Get work-item’s row and column position */
   int column = get_global_id(0); 
   int row = get_global_id(1);

   /* Accumulated pixel value */
   float4 sum = (float4)(0.0);

   /* Filter's current index */
   int filter_index =  0;

   int2 coord;
   float4 pixel;


   int start = 0 - (int)floor(dim/2.0f);
   int end = 0 + (int)floor(dim/2.0f);

      /* Iterate over the rows */
   for(int i = start; i <= end; i++) {
	  coord.y =  row;
	  coord.x = column + i;

	  	/* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * SmartFilter1[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * SmartFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * SmartFilter3[filter_index++];

   }

	  coord = (int2)(column, row); 
	  write_imagef(dst_image, coord, sum);
}

__kernel void reduction_vector(__global float4* data, 
      __local float4* partial_sums) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      data[get_group_id(0)] = partial_sums[0];
   }
}

__kernel void reduction_complete(__global float4* data, 
      __local float4* partial_sums, __global float* sum) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_local_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      *sum = partial_sums[0].s0 + partial_sums[0].s1 +
             partial_sums[0].s2 + partial_sums[0].s3;
   }
}

bool test_lum(float4 pixel, float thres){
	float lum = 1.0f*((pixel.s0 * 0.299)+(pixel.s1 * 0.587)+(pixel.s2 * 0.114));

	if(thres > lum)
		return false;
	
	return true;
}

__kernel void output_pass_threshold(	read_only image2d_t src_image,
							write_only image2d_t dst_image, float thres) {

   /* Get pixel coordinate */
   int2 coord = (int2)(get_global_id(0), get_global_id(1));

   /* Read pixel value */
   float4 pixel = read_imagef(src_image, sampler, coord);

   /* Write new pixel value to output */
   thres = thres*0.005f;
  if(!test_lum(pixel, thres))
	pixel = pixel * 0;
 
   write_imagef(dst_image, coord, pixel);
}

__kernel void final_bloom_step(	read_only image2d_t src_image1, read_only image2d_t src_image2,
							write_only image2d_t dst_image) {
   /* Get pixel coordinate */
   int2 coord = (int2)(get_global_id(0), get_global_id(1));




   /* Read pixel value */
   float4 pixel1 = read_imagef(src_image1, sampler, coord);
   float4 pixel2 = read_imagef(src_image2, sampler, coord);

   float4 pixel = pixel1 + pixel2;

   /* Write new pixel value to output */
   write_imagef(dst_image, coord, pixel);
}