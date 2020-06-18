#ifndef ELENA_CUDA_UTIL_H_
#define ELENA_CUDA_UTIL_H_
#include <cublas_v2.h>
#include <cuda.h>

static const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

#define CUDA_SAFE_CALL(ret) \
  { Assert((ret), __FILE__, __LINE__); }

inline void Assert(cudaError_t ret, const char *file, int line) {
  if (ret != cudaSuccess) {
    std::cerr << "\ncuda error in file " << file << " at line " << line
              << " with error " << cudaGetErrorString(ret) << '\n';
    exit(EXIT_FAILURE);
  }
}
// inline void Assert(cudnnStatus_t ret, const char *file, int line) {
//   if (ret != CUDNN_STATUS_SUCCESS) {
//     std::cerr << "\ncudnn error in file " << file << " at line " << line
//               << " with error " << cudnnGetErrorString(ret) << '\n';
//     exit(EXIT_FAILURE);
//   }
// }
inline void Assert(cublasStatus_t ret, const char *file, int line) {
  if (ret != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "\ncublas error in file " << file << " at line " << line
              << " with error " << cublasGetErrorString(ret) << '\n';
    exit(EXIT_FAILURE);
  }
}

#define KernelErrChk                                                    \
  {                                                                     \
    cudaError_t errSync = cudaGetLastError();                           \
    cudaError_t errAsync = cudaDeviceSynchronize();                     \
    if (errSync != cudaSuccess) {                                       \
      printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));   \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
    if (errAsync != cudaSuccess) {                                      \
      printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }
#endif  // ELENA_CUDA_UTIL_H_
