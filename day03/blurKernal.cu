#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void blurKernalGrayscale(int width, int height, unsigned char *input,
                                    unsigned char *output) {

  // we got the grid cords
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0f;
  int count = 0;
  if (x < width && y < height) {
    for (int row = -1; row <= 1; row++) {
      for (int col = -1; col <= 1; col++) {
        int currCol = y + col;
        int currRow = x + row;

        // check for the bounderies

        if (currRow >= 0 && currRow < height && currCol >= 0 &&
            currCol < width) {
          sum += input[currRow * width + currCol];
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
    int index = (y * width + x) * 3;
    for (int row = -1; row <= 1; row++) {
      for (int col = -1; col <= 1; col++) {
        int currCol = y + col;
        int currRow = x + row;

        // check for the bounderies

        if (currRow >= 0 && currRow < height && currCol >= 0 &&
            currCol < width) {
          int neighborIdx = (currRow * width + currCol) * 3;
          sumR += input[neighborIdx];
          sumG += input[neighborIdx + 1];
          sumB += input[neighborIdx + 2];
          count++;
        }
      }
    }

    if (count > 0) {
      output[index] = (unsigned char)(sumR / count);
      output[index + 1] = (unsigned char)(sumG / count);
      output[index + 2] = (unsigned char)(sumB / count);
    }
  }
}

__global__ void color2GrayKernal(int width, int height, unsigned char *input,
                                 unsigned char *output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = (y * width + x) * 3;

    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    unsigned char gray =
        static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.11f * b);

    output[y * width + x] = gray;
  }
}

int main() {

  cv::Mat image = cv::imread("input_image.jpeg");

  if (image.empty()) {
    std::cerr << "Error loading image" << std::endl;
    return -1;
  }

  // Convert BGR to RGB

  image = image.clone();

  int width = image.cols;
  int height = image.rows;

  unsigned char *h_input = image.data;
  unsigned char *h_output = new unsigned char[width * height * 3];

  unsigned char *d_input, *d_output;

  cudaMalloc((void **)&d_input, width * height * 3 * sizeof(unsigned char));
  cudaMalloc((void **)&d_output, width * height * 3 * sizeof(unsigned char));
  cudaMemcpy(d_input, h_input, width * height * 3 * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

  blurKernalRGB<<<numBlocks, threadsPerBlock>>>(width, height, d_input,
                                                d_output);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {

    std::cerr << "Cuda error" << cudaGetErrorString(err) << std::endl;
  }

  cudaDeviceSynchronize();
  cudaMemcpy(h_output, d_output, width * height * 3 * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cv::Mat blurImage(height, width, CV_8UC3, h_output);
  cv::imwrite("blur.jpeg", blurImage);

  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_output;

  std::cout << "Image successfully converted to Blurr using CUDA!" << std::endl;

  return 0;
}
