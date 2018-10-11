__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

__kernel void gradient(	read_only image2d_t src_image,
						write_only image2d_t dst_image) {

   /* Get pixel coordinate */
   int2 coord = (int2)(get_global_id(0), get_global_id(1));

   /* Read pixel value */
   float4 pixel = read_imagef(src_image, sampler, coord);

   /* Darken pixel based on coordinate */
   pixel.x -= (float)coord.x/get_image_width(src_image);

   /* Write new pixel value to output */
   write_imagef(dst_image, coord, pixel);
}
