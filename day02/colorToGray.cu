#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void colorToGray(unsigned char *output, unsigned char *input,
                            int width, int height) {
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
    std::cerr << "Could not open or find image" << std::endl;
    return -1;
  }

  int width = image.cols;
  int height = image.rows;

  unsigned char *h_input = image.data;
  unsigned char *h_output = new unsigned char[width * height];

  unsigned char *d_input, *d_output;

  cudaMalloc((void **)&d_input, width * height * 3 * sizeof(unsigned char));
  cudaMalloc((void **)&d_output, width * height * sizeof(unsigned char));

  cudaMemcpy(d_input, h_input, width * height * 3 * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

  colorToGray<<<numBlocks, threadsPerBlock>>>(d_output, d_input, width, height);

  // we will also check for errors

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {

    std::cerr << "Cuda error" << cudaGetErrorString(err) << std::endl;
  }


  cudaDeviceSynchronize(); 
  cudaMemcpy(h_output, d_output, width * height * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  // lets create new image brothaa!!!

  cv::Mat grayImage(height, width, CV_8UC1, h_output);
  cv::imwrite("grayscale_cuda.jpeg", grayImage);

  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_output;

  std::cout << "Image successfully converted to grayscale using CUDA!"
            << std::endl;

  return 0;
}
