/*
 * 
 * NICK Local Image thresholding Algorithm
 * M. Hassan Najafi
 *
 */

#include <stdio.h>
#include "Nick.h"

//Declare Constant Memory Variable
__constant__ float Nc[IMAGE_WIDTH*IMAGE_HEIGTH];


__global__ void NickKernel( Matrix N, Matrix P, int Image_width, int Image_height )
{

	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	//Determine row and col 
	unsigned int row = (blockIdx.y * BLOCK_SIZE ) + ty;
	unsigned int col = (blockIdx.x * BLOCK_SIZE ) + tx;
	
	//Determining the borders of local window
	unsigned int  w_start_x = ( row < window_half_size) ? 0 : row - window_half_size;
	unsigned int  w_start_y = ( col < window_half_size) ? 0 : col - window_half_size;
	unsigned int  w_end_x = ( Image_height-1 < row + window_half_size) ? Image_height-1 : row + window_half_size;
	unsigned int  w_end_y = ( Image_width-1 < col + window_half_size) ? Image_width-1 : col + window_half_size;
	
	//Compute total number of elements in the local window
	//Because of margin elements the window size for marginal elements is smaller than center elements
	unsigned int Current_window_size = (w_end_x-w_start_x+1) * (w_end_y-w_start_y+1);
	
	if (row < Image_height && col < Image_width )
	{
		float temp;
		float Total_sum = 0;
		float Total_sum_pow2 = 0;
		for (unsigned int i=w_start_x ; i <= w_end_x ; i = i + 1 )
			for (unsigned int j=w_start_y ; j <= w_end_y ; j = j + 1 )			
			   {temp = N.elements[i * Image_width + j];
				Total_sum = Total_sum + temp ;
				Total_sum_pow2 = Total_sum_pow2 + ( temp * temp );}			
				
		float mean = 	Total_sum / Current_window_size;		
		float Threshold = mean + K_parameter * sqrtf( (Total_sum_pow2 - mean*mean) / Current_window_size ) ;
		
//		P.elements[row * Image_width + col] = Threshold;
		
 		if ( Threshold < N.elements[row * Image_width + col])
 			P.elements[row * Image_width + col] = 1;
 		else
 			P.elements[row * Image_width + col] = 0;
	}	
}




__global__ void NickKernel_shared1( Matrix N, Matrix P, int Image_width, int Image_height )
{
	//Declare Shared Memory variables
	__shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//row and col for output
	int row = (blockIdx.y * BLOCK_SIZE ) + ty;
	int col = (blockIdx.x * BLOCK_SIZE ) + tx;
	
	//Determining start and end borders of each tile
	int Tile_start_row = blockIdx.y * BLOCK_SIZE;
	int Tile_end_row   = (blockIdx.y+1) * BLOCK_SIZE-1;
	int Tile_start_col = blockIdx.x * BLOCK_SIZE;
	int Tile_end_col   = (blockIdx.x+1) * BLOCK_SIZE-1;
	
	//each thread Load one element into shared memory
	if ( (row < N.height) && (col < N.width))
		Nds[ty][tx] = N.elements[row * Image_width + col]; //Works correct
	else
		Nds[ty][tx] = 0.0f;
	
	//Ensures that all threads have finished loading the tiles
	__syncthreads();	
	
	//Determining the borders of local window - y:col:tx   x:row:ty
	 int  w_start_x = ( row < window_half_size) ? 0 : row - window_half_size;
	 int  w_start_y = ( col < window_half_size) ? 0 : col - window_half_size;
	 int  w_end_x = ( Image_height-1 < row + window_half_size) ? Image_height-1 : row + window_half_size;
	 int  w_end_y = ( Image_width-1 < col + window_half_size) ? Image_width-1 : col + window_half_size;
	
	//Compute total number of elements in the local window
	int Current_window_size = (w_end_x-w_start_x+1) * (w_end_y-w_start_y+1);
		
	if (row < N.height && col < N.width )
	{
		float temp;
		float Total_sum = 0;
		float Total_sum_pow2 = 0;
		for (int i=w_start_x ; i <= w_end_x ; i = i + 1 )
			for ( int j=w_start_y ; j <= w_end_y ; j = j + 1 )			
			   {
			   
			    if ( (Tile_start_row<i)&& (Tile_start_col<j)&&(i<Tile_end_row)&&(j<Tile_end_col)   )				
					temp = Nds[i%BLOCK_SIZE][j%BLOCK_SIZE];
				else
					temp = N.elements[i * Image_width + j];
			   
				Total_sum = Total_sum + temp ;
				Total_sum_pow2 = Total_sum_pow2 + ( temp * temp );}			
				
		float mean = 	Total_sum / Current_window_size;		
		float Threshold = mean + K_parameter * sqrtf( (Total_sum_pow2 - mean*mean) / Current_window_size ) ;
	
		if (ty < BLOCK_SIZE && tx < BLOCK_SIZE)
		{
	//		P.elements[row * Image_width + col] = Threshold;
			
	 		if ( Threshold < N.elements[row * Image_width + col])
	 			P.elements[row * Image_width + col] = 1;
	 		else
	 			P.elements[row * Image_width + col] = 0;
		}
	
	}
			
}



__global__ void NickKernel_shared2( Matrix N, Matrix P, int Image_width, int Image_height )
{
	//Declare Shared Memory variables
	__shared__ float Nds[BLOCK_SIZE+WINDOW_SIZE-1][BLOCK_SIZE+WINDOW_SIZE-1];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//row and col for output
	int row_o = (blockIdx.y * BLOCK_SIZE ) + ty;
	int col_o = (blockIdx.x * BLOCK_SIZE ) + tx;
	int row_i = row_o - window_half_size;
	int col_i = col_o - window_half_size;
	
	
	if ((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width))
		Nds[ty][tx] = N.elements[row_i * N.width + col_i];
	else 
		Nds[ty][tx] = 0.0f;
	
	//Ensures that all threads have finished loading the tiles
	__syncthreads();	
	
	//Determining the borders of local window - y:col:tx   x:row:ty
	 int  w_start_x = ( row_o < window_half_size) ? 0 : row_o - window_half_size;
	 int  w_start_y = ( col_o < window_half_size) ? 0 : col_o - window_half_size;
	 int  w_end_x = ( Image_height-1 < row_o + window_half_size) ? Image_height-1 : row_o + window_half_size;
	 int  w_end_y = ( Image_width-1 < col_o + window_half_size) ? Image_width-1 : col_o + window_half_size;
	
	//Compute total number of elements in the local window
	int Current_window_size = (w_end_x-w_start_x+1) * (w_end_y-w_start_y+1);
	
	float Threshold;
	//Only the threads that fall within the output tile dimensions need to perform a computation
	if (ty < BLOCK_SIZE && tx < BLOCK_SIZE)
	{
		float temp;
		float Total_sum = 0;
		float Total_sum_pow2 = 0;
	
		for (int i=0 ; i <= WINDOW_SIZE-1 ; i = i + 1 )
			for ( int j=0 ; j <= WINDOW_SIZE-1 ; j = j + 1 )							
			{		
				temp = Nds[i+ty][j+tx];
								   
				Total_sum = Total_sum + temp ;
				Total_sum_pow2 = Total_sum_pow2 + ( temp * temp );
			}		
				
		float mean = 	Total_sum / Current_window_size;		
		Threshold = mean + K_parameter * sqrtf( (Total_sum_pow2 - mean*mean) / Current_window_size ) ;
	}
	
	
	if (row_o < N.height && col_o < N.width )
	  if (ty < BLOCK_SIZE && tx < BLOCK_SIZE)
	  {
	//	P.elements[row_o * N.width + col_o] = Threshold;
		if ( Threshold < N.elements[row_o * Image_width + col_o])
	 		P.elements[row_o * Image_width + col_o] = 1;
	 	else
	 		P.elements[row_o * Image_width + col_o] = 0;
	   }
		
			
}



__global__ void NickKernel_shared_OneBlock( Matrix N, Matrix P, int Image_width, int Image_height )
{
	//Declare Shared Memory variables
	__shared__ float Nds[IMAGE_WIDTH*IMAGE_HEIGTH];


	//Determine row and col 
	unsigned int index_row = threadIdx.y;
	unsigned int index_col = threadIdx.x;
	
	unsigned int index = index_row * BLOCK_SIZE + index_col;
	
	//Load image pixels into the shared memory
	//It seems that our accesses are coalesced.
	while ( index < IMAGE_WIDTH*IMAGE_HEIGTH )
	{
		Nds[index] = N.elements[index];
		index = index + (BLOCK_SIZE*BLOCK_SIZE);
	}	
	__syncthreads();
	
	
	index = index_row * BLOCK_SIZE + index_col;

	while( index < IMAGE_WIDTH*IMAGE_HEIGTH )
	{
		index_row = index/Image_width;
		index_col = index%Image_width;
		//Determining the borders of local window
		unsigned int  w_start_x = ( index_row < window_half_size) ? 0 : index_row - window_half_size;
		unsigned int  w_start_y = ( index_col < window_half_size) ? 0 : index_col - window_half_size;
		unsigned int  w_end_x = ( Image_height-1 < index_row + window_half_size) ? Image_height-1 : index_row + window_half_size;
		unsigned int  w_end_y = ( Image_width-1 < index_col + window_half_size) ? Image_width-1 : index_col + window_half_size;

		unsigned int Current_window_size = (w_end_x-w_start_x+1) * (w_end_y-w_start_y+1);
	
		float temp;
		float Total_sum = 0;
		float Total_sum_pow2 = 0;
		for (unsigned int i=w_start_x ; i <= w_end_x ; i = i + 1 )
			for (unsigned int j=w_start_y ; j <= w_end_y ; j = j + 1 )
			{
				temp = Nds[i * Image_width + j];
				Total_sum = Total_sum + temp ;
				Total_sum_pow2 = Total_sum_pow2 + ( temp * temp );
			}
				
		float mean = 	Total_sum / Current_window_size;		
		float Threshold = mean + K_parameter * sqrt( (Total_sum_pow2 - mean*mean) / Current_window_size ) ;
		P.elements[index_row * Image_width + index_col] = Threshold;
		
		index = index + (BLOCK_SIZE*BLOCK_SIZE);

	}	
	
}