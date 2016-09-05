/*
 * 
 * NICK Local Image thresholding Algorithm
 * M. Hassan Najafi
 * Najaf011@umn.edu
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Nick.h"
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold( float*, const float*, unsigned int, unsigned int);


void computeGold(float* C, const float* B, unsigned int hB, unsigned int wB)
{
    for (unsigned int i = 0; i < hB; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            float threshold, mean;
            float sum= 0;
            float square_sum = 0;
            int curr_index = i*wB + j;
            unsigned int mbegin = (i < window_half_size)?0: i-window_half_size ;
            unsigned int nbegin = (j < window_half_size)?0: j-window_half_size ;
            unsigned int mend = (hB-1 <i + window_half_size)?hB-1: i+window_half_size;
            unsigned int nend = (wB-1 <j + window_half_size)?wB-1: j+window_half_size;
            unsigned int curr_window_size = 0;

            for(unsigned int m = mbegin; m <= mend; m++)
                    for(unsigned int n = nbegin; n <= nend; n++) {
                       float curr_element =  B[m*wB + n];
                       sum+= curr_element;
                       square_sum+= curr_element * curr_element;}

            curr_window_size = (mend-mbegin+1) * (nend - nbegin +1);
            mean = sum/(curr_window_size);
            threshold = mean + K_parameter* sqrt((square_sum - mean*mean)/(curr_window_size));
			
//			C[curr_index] = threshold;
			if (threshold < B[curr_index] ){
	              C[curr_index] = 1;
	          } else {
	              C[curr_index] = 0;}
         }
}

