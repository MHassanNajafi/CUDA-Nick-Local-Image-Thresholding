/*
 * 
 * NICK Local Image thresholding Algorithm
 * M. Hassan Najafi
 * Najaf011@umn.edu
 */
#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define K_parameter -0.2

#define WINDOW_SIZE 9 //9 15 33
#define window_half_size 4 //4 7 16

#define BLOCK_SIZE 16 //MAX=32 since 32*32=1024

#define IMAGE_WIDTH 75
#define IMAGE_HEIGTH 80

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;



#endif // _MATRIXMUL_H_

