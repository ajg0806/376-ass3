__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 


// 3x3 Blurring filter
__constant float BlurringFilter[9] = {	0.077847, 0.123317, 0.077847,
0.123317, 0.195346, 0.123317,
0.077847, 0.123317, 0.077847};


// 4x4 Blurring filter
__constant float BlurringFilter2[25] = {	0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
0.003765, 0.015019, 0.023792, 0.015019, 0.003765};


__constant float BlurringFilter3[49] = { 0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036 };


__kernel void simple_conv(read_only image2d_t src_image,
					write_only image2d_t dst_image, int dim, write_only image2d_t dst_image2) {


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

      /* Iterate over the columns */
	  for(int j = start; j <= end; j++) {
         coord.x = column + j;

		 /* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];
	  }
   }

   /* Write new pixel value to output */
   coord = (int2)(column, row); 
   write_imagef(dst_image, coord, sum);







      /* Iterate over the rows */
   for(int i = start; i <= end; i++) {
	  coord.y =  row + i;
	  coord.x = column;

	  	/* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];

   }

   
      /* Iterate over the columns */
	  for(int j = start; j <= end; j++) {
         coord.x = column + j;
		 coord.y = row;
		 /* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];
	  }




	  coord = (int2)(column, row); 
	  write_imagef(dst_image2, coord, sum);
}


__kernel void smart_blur(read_only image2d_t src_image,
					write_only image2d_t dst_image, int dim, write_only image2d_t dst_image2) {


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

      /* Iterate over the columns */
	  for(int j = start; j <= end; j++) {
         coord.x = column + j;

		 /* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];
	  }
   }

   /* Write new pixel value to output */
   coord = (int2)(column, row); 
   write_imagef(dst_image, coord, sum);







      /* Iterate over the rows */
   for(int i = start; i <= end; i++) {
	  coord.y =  row + i;
	  coord.x = column;

	  	/* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];

   }

   
      /* Iterate over the columns */
	  for(int j = start; j <= end; j++) {
         coord.x = column + j;
		 coord.y = row;
		 /* Read value pixel from the image */ 		
		 pixel = read_imagef(src_image, sampler, coord);
		 /* Acculumate weighted sum */
		 if(dim == 3)
			sum.xyz += pixel.xyz * BlurringFilter[filter_index++];
		if(dim == 5)
			sum.xyz += pixel.xyz * BlurringFilter2[filter_index++];
		if(dim == 7)
			sum.xyz += pixel.xyz * BlurringFilter3[filter_index++];
	  }




	  coord = (int2)(column, row); 
	  write_imagef(dst_image2, coord, sum);
}