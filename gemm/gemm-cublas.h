#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>

#include "gemm-common.h"

class GemmCublas {
 public:
  /*
  关于 cudaDataType_t
  https://docs.nvidia.com/cuda/cublas/index.html#cudadatatype-t
  CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUDA_R_8I, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2

  关于 cublasGemmAlgo_t
  https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmalgo-t
  CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_ALGO0, ..., CUBLAS_GEMM_ALGO23

  关于 cublasOperation_t
  https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation-t
  CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C

  关于 alpha beta
  https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmex
  */
  GemmCublas(int m, int n, int k, cudaDataType_t data_type,
             cublasGemmAlgo_t algo, cublasOperation_t transa,
             cublasOperation_t transb, float alpha = 1.0, float beta = 0.0);
  ~GemmCublas();
  void run_gemm();

 private:
  // alpha beta的意义见文档
  // 一般alpha取1.0, beta取0.0
  float alpha_, beta_;
  // operation
  cublasOperation_t transa_, transb_;
  // 矩阵的大小
  int m_, n_, k_;
  // host内存
  void *h_A, *h_B, *h_C;
  // device 内存
  void *d_A, *d_B, *d_C;
  unsigned int element_size_;  // 数据大小, 单位是字节
  cudaDataType_t data_type_;   // 数据类型
  cublasGemmAlgo_t algo_;      // 算法
  cudaEvent_t start_, stop_;   // 用于记录时间
};

#endif