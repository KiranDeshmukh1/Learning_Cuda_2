#include <cuda_runtime.h>
#include <iostream>

__global__ void blurKernalGrayscale(int width, int height, unsigned char *input,
                                    unsigned char *output) {

  // we got the grid cords
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0f;
  int count = 0;
  if (x < width && y < height) {
    for (int row = -1; row <= = 1; row++) {
      for (int col = -1; col <= = 1; col++) {
        int currCol = y + col;
        int currRow = x + row;

        // check for the bounderies

        if (currRow >= 0 && currRow < height && currCol >= 0 &&
            currCol < width) {
          sum + = input[currRow * width + currCol];
          count++;
        }
      }
    }
    output[y * width + x] = (unsigned char)(sum / count);
  }
}

// For RBG images
__global__ void blurKernalRGB(int width, int height, unsigned char *input,
                              unsigned char *output) {

  // we got the grid cords
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int count = 0;
  int sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
  if (x < width && y < height) {
    int index = [y * width + x] * 3;
    for (int row = -1; row <= = 1; row++) {
      for (int col = -1; col <= = 1; col++) {
        int currCol = y + col;
        int currRow = x + row;

        // check for the bounderies

        if (currRow >= 0 && currRow < height && currCol >= 0 &&
            currCol < width) {
          int neighborIdx = (currRow * width + currCol) * 3 sumR +=
              input[neighborIdx];
          sumG += input[neighborIdx + 1];
          sumB += input[neighborIdx + 2];
          count++;
        }
      }
    }
    output[index] = (unsigned char)(sumR / count);
    output[index + 1] = (unsigned char)(sumG / count);
    output[index + 2] = (unsigned char)(sumB / count);
  }
}
