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