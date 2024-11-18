#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include <iostream>

void cublas_gemm(int M, int N, int K, float *A, float *B, float *C) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K,
              &beta, C, N);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "cuBLAS GEMM time: " << milliseconds << " ms" << std::endl;

  cublasDestroy(handle);
}

void cublaslt_gemm(int M, int N, int K, float *A, float *B, float *C) {
  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
  cublasLtMatrixTransformDesc_t transformDesc = nullptr;

  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);

  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cublasLtMatmul(handle, matmulDesc, &alpha, B, Bdesc, A, Adesc, &beta, C,
                 Cdesc, C, Cdesc, nullptr, nullptr, 0, nullptr);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "cuBLASLT GEMM time: " << milliseconds << " ms" << std::endl;

  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtDestroy(handle);
}

void cudnn_gemm(int M, int N, int K, float *A, float *B, float *C) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  cudnnTensorDescriptor_t descA, descB, descC;
  cudnnCreateTensorDescriptor(&descA);
  cudnnCreateTensorDescriptor(&descB);
  cudnnCreateTensorDescriptor(&descC);

  cudnnSetTensor4dDescriptor(descA, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K,
                             1, M);
  cudnnSetTensor4dDescriptor(descB, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N,
                             1, K);
  cudnnSetTensor4dDescriptor(descC, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, N,
                             1, M);

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudnnConvolutionForward(handle, &alpha, descA, A, descB, B, nullptr, 0, &beta,
                          descC, C);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "cuDNN GEMM time: " << milliseconds << " ms" << std::endl;

  cudnnDestroyTensorDescriptor(descA);
  cudnnDestroyTensorDescriptor(descB);
  cudnnDestroyTensorDescriptor(descC);
  cudnnDestroy(handle);
}

void cutlass_gemm(int M, int N, int K, float *A, float *B, float *C) {
  using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor,
                                           float, cutlass::layout::RowMajor,
                                           float, cutlass::layout::RowMajor>;

  Gemm gemm_op;
  cutlass::Status status;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  status = gemm_op({{M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1, 0}, {0, 0}});
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "CUTLASS GEMM time: " << milliseconds << " ms" << std::endl;

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM failed" << std::endl;
  }
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  float *A, *B, *C;
  cudaMallocManaged(&A, M * K * sizeof(float));
  cudaMallocManaged(&B, K * N * sizeof(float));
  cudaMallocManaged(&C, M * N * sizeof(float));

  // Initialize matrices A and B
  for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
  for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

  // Run GEMM with cuBLAS
  cublas_gemm(M, N, K, A, B, C);

  // Run GEMM with CUTLASS
  cutlass_gemm(M, N, K, A, B, C);

  // Run GEMM with cuDNN
  cudnn_gemm(M, N, K, A, B, C);

  // Run GEMM with cuBLASLT
  cublaslt_gemm(M, N, K, A, B, C);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}