#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

int open_cuda_dev(int cudadev) {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Detected %d CUDA accelerators:\n", deviceCount);
  int dev;
  for (dev=0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n",
           dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024),
           deviceProp.major, deviceProp.minor);
  }

  if (cudadev < 0 || cudadev >= deviceCount) {
    printf("No such CUDA device %d, using device 0\n", cudadev);
    cudadev = 0;
  }
  cudaSetDevice(cudadev);
  CUERR // check and clear any existing errors

  return 0;
}

}
