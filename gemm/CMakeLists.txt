add_executable(gemm-fp16 gemm-fp16.cu)
target_link_libraries(gemm-fp16 PRIVATE cudart cublas cublasLt nvidia-ml)

add_executable(gemm-bf16 gemm-bf16.cu)
target_link_libraries(gemm-bf16 PRIVATE cudart cublas cublasLt nvidia-ml)

add_executable(gemm-fp32 gemm-fp32.cu)
target_link_libraries(gemm-fp32 PRIVATE cudart cublas cublasLt nvidia-ml)