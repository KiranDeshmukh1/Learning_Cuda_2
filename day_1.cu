/*Day_1*/

#include <cuda_runtime.h>
#include <iostream>

// calling hello world with gpu
__global__ void cuda_hello() {
  printf("Output :");
  printf("Hello world from Block %d and Thread %d", blockIdx.x, threadIdx.x);
}

__global__ void vec_add(float *out, float *a, float *b, int n) {

  int idx = threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

__global__ void parallel_vec_add(float *out, float *a, float *b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    out[tid] = a[tid] + b[tid];
  }
}

int main() {
  int N = 10;
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;
  a = new float[N];
  b = new float[N];
  out = new float[N];

  for (int i = 0; i < N; i++) {
    a[i] = i * 1.0f;
    b[i] = i * 2.0f;
  }

  cudaMalloc((void **)&d_a, sizeof(float) * N);
  cudaMalloc((void **)&d_b, sizeof(float) * N);
  cudaMalloc((void **)&d_out, sizeof(float) * N);

  // copy from host to device

  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // vec_add<<<1, N>>>(d_out, d_a, d_b, N);

  //executing kernal 

  int threads = 256;
  int blocks = ((N + threads)/threads);
  
  parallel_vec_add<<<blocks,threads>>>(d_out,d_a,d_b,N);

  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  std::cout << "Results: " << "\n";

  for (int i = 0 ; i < N ; i++) {
    std::cout<<out[i]<<"\n";
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  delete[] a;
  delete[] b;
  delete[] out;

  cudaDeviceSynchronize();
  return 0;
}
