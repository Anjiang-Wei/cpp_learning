find_package(CUDA QUIET REQUIRED)

 
set(CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70;-O3)

cuda_add_executable(
    test_gemm
    test_gemm.cpp test_gemm.cu)
target_link_libraries(test_gemm cublas cuda)
target_include_directories(test_gemm PRIVATE ${cuda_INCLUDE_DIRS})
