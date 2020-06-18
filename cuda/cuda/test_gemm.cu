#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "util.h"

extern "C" __global__ void gemm(int M, int N, int K, float *A, float *B,
                                float *C, int alpha, int beta) {}

void test(float *A, float *B, float *blas, float *h_c, int m, int k, int n) {
  float *result = (float *)malloc(sizeof(float) * m * n);

  /*
  //column-major implementation
  for (int i=0; i<m; ++i)
          for (int j=0; j<n; ++j){
                  result[i+j*m] = (float)0.f;

                  for (int t=0; t<k; ++t)
                          result[i+j*m] += (float)A[i+m*t] * (float)B[j*k+t];
          }

  printf("\nCPU Result:\n");
  for (int i=0; i<m; ++i){
          for (int j=0; j<n; ++j)
                  printf("%.0f ", result[i+j*m]);
          printf("\n");
  }
  */

  printf("\nCUBLAS Result:\n");
  for (int i = 0; i < m; ++i) {
    if (i % 4 == 0) printf("\n");
    for (int j = 0; j < n; ++j) {
      printf("%.0f ", blas[i + j * m]);
      if ((j + 1) % 4 == 0) printf(" ");
    }
    printf("\n");
  }

  printf("\nGEMM Result:\n");
  for (int i = 0; i < m; ++i) {
    if (i % 4 == 0) printf("\n");
    for (int j = 0; j < n; ++j) {
      printf("%.0f ", h_c[i + j * m]);
      if ((j + 1) % 4 == 0) printf(" ");
    }
    printf("\n");
  }

  for (int i = 0; i < m * n; ++i)
    if (abs(blas[i] - h_c[i]) > 10e-2) {
      printf("Rejected @ %d\n", i);
      return;
    }
  printf("Passed\n");
}

void test_gemm() {
  int M = 64;
  int N = 64;
  int K = 64;
  float alpha = 1.f;
  float beta = 0.f;

  float *h_A = NULL;
  float *h_B = NULL;
  float *h_C = NULL;
  float *h_b = NULL;

  h_A = (float *)malloc(sizeof(float) * M * K);
  h_B = (float *)malloc(sizeof(float) * K * N);
  h_C = (float *)malloc(sizeof(float) * M * N);
  h_b = (float *)malloc(sizeof(float) * M * N);

  for (int i = 0; i < M * K; ++i) h_A[i] = float(i % 10);

  for (int i = 0; i < K * N; ++i) h_B[i] = float(i % 5);

  for (int i = 0; i < M * N; ++i) h_C[i] = (i % 5);

  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  CUDA_SAFE_CALL(cudaSetDevice(0));

  CUDA_SAFE_CALL(cudaMalloc((void **)&A, sizeof(float) * M * K));
  CUDA_SAFE_CALL(cudaMalloc((void **)&B, sizeof(float) * K * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&C, sizeof(float) * M * N));
  CUDA_SAFE_CALL(
      cudaMemcpy(A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(C, h_C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));

  float elapsed = 0.0f;
  double avg_ms = 0.f;
  int64_t num_flops = 0;
  double gflops_per_sec = 0.f;

  // CUBLAS
  cublasHandle_t handle;
  CUDA_SAFE_CALL(cublasCreate(&handle));
  CUDA_SAFE_CALL(cudaEventRecord(start, 0));

  int const ITERATION = 1;

  for (int i = 0; i < ITERATION; ++i)
    CUDA_SAFE_CALL(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (void *)&alpha, (void *)A,
        CUDA_R_32F, M, (void *)B, CUDA_R_32F, K, (void *)&beta, (void *)C,
        CUDA_R_32F, M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, stop));

  avg_ms = elapsed / ITERATION;
  num_flops = (2 * int64_t(M) * int64_t(N) * int64_t(K)) +
              (2 * int64_t(M) * int64_t(N));
  gflops_per_sec = double(num_flops) / avg_ms / 1.0e6;
  printf("Avg runtime: %.3f us, total flops: %ld, GFLOP/s: %.2f\n", avg_ms,
         num_flops, gflops_per_sec);

  CUDA_SAFE_CALL(cublasDestroy(handle));
  CUDA_SAFE_CALL(
      cudaMemcpy(h_b, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(
      cudaMemcpy(C, h_C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

  dim3 grid_size;
  dim3 block_size;

  grid_size.x = M / 16;
  grid_size.y = N / 16;
  grid_size.z = 1;

  block_size.x = 8;
  block_size.y = 8;
  block_size.z = 1;

  CUDA_SAFE_CALL(cudaEventRecord(start, 0));

  for (int i = 0; i < ITERATION; ++i) {
    gemm<<<grid_size, block_size>>>(M, N, K, A, B, C, alpha, beta);
    KernelErrChk
  }
  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, stop));

  avg_ms = elapsed / ITERATION;
  gflops_per_sec = double(num_flops) / avg_ms / 1.0e6;
  printf("Avg runtime: %.3f us, total flops: %ld, GFLOP/s: %.2f\n", avg_ms,
         num_flops, gflops_per_sec);

  CUDA_SAFE_CALL(
      cudaMemcpy(h_C, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
  test(h_A, h_B, h_b, h_C, M, K, N);

  return;
}
