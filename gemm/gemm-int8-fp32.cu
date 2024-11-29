// Copyright (c) 2024 BAAI. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License")

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <argparse.hpp>
#include <chrono>
#include <iostream>

// 3072 6400 7168
int M = 3072;
int N = 6400;
int K = 7168;

int device_id = 0;

// A40的性能
double int8_gflops = 299.3 * 1024.0;
double int8_tflops = 299.3;

// 记录最好的算法, layout
double best_tflops = 0.0;
int best_algo, best_layout_a, best_layout_b;

struct PrecisionConfig {
  std::string func_name;
  cudaDataType_t cudaType;
  cublasComputeType_t cublasType;
  int bytesPerElement;
  std::string type_name;
  int NUM_ITERATIONS;
  int WARMUP_ITERATIONS = 10;
  cublasOperation_t transa;
  cublasOperation_t transb;
  cublasGemmAlgo_t algo;
};

void test(const PrecisionConfig& config) {
  cudaSetDevice(device_id);
  int8_t *d_A, *d_B;
  float* d_C;
  cudaMallocManaged(&d_A, M * K * config.bytesPerElement);
  cudaMallocManaged(&d_B, K * N * config.bytesPerElement);
  cudaMallocManaged(&d_C, M * N * sizeof(float));

  cublasHandle_t handle;
  cublasCreate(&handle);

  int alpha = 1;
  int beta = 0;

  for (int i = 0; i < config.WARMUP_ITERATIONS; ++i) {
    cublasGemmEx(handle, config.transa, config.transb, M, N, K, &alpha, d_A,
                 config.cudaType, (config.transa == CUBLAS_OP_N ? M : K), d_B,
                 config.cudaType, (config.transb == CUBLAS_OP_N ? K : N), &beta,
                 d_C, CUDA_R_32F, M, config.cublasType, config.algo);
  }

  cudaError_t syncError = cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  if (syncError != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(syncError) << std::endl;
  }

  for (int i = 0; i < config.NUM_ITERATIONS; ++i) {
    cublasGemmEx(handle, config.transa, config.transb, M, N, K, &alpha, d_A,
                 config.cudaType, (config.transa == CUBLAS_OP_N ? M : K), d_B,
                 config.cudaType, (config.transb == CUBLAS_OP_N ? K : N), &beta,
                 d_C, CUDA_R_32F, M, config.cublasType, config.algo);
  }
  syncError = cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  if (syncError != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(syncError) << std::endl;
  }
  auto dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // std::cout << "Average " << config.name << " Single Op Duration: "
  // << dt.count() / config.NUM_ITERATIONS << " us" << std::endl;

  double time_second = dt.count() / 1.0e6;
  double ops = 2.0 * M * N * K * config.NUM_ITERATIONS;
  double OPS = ops / time_second;
  double TOPS = OPS / 1.0e12;
  double ratio = TOPS / int8_tflops;

  if (TOPS > best_tflops) {
    best_tflops = TOPS;
    best_algo = (int)config.algo;
    best_layout_a = (int)config.transa;
    best_layout_b = (int)config.transb;
  }

  std::cout << config.func_name << "," << config.type_name << "," << config.algo
            << "," << config.transa << "," << config.transb << "," << TOPS
            << "," << ratio << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cublasDestroy(handle);
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("gemm-int8-fp32");

  // append模式不会输出csv第一行
  program.add_argument("--append")
      .flag()
      .help("output the first line of csv file?");
  program.add_argument("-m").default_value(512).store_into(M).help("set m");
  program.add_argument("-k").default_value(512).store_into(K).help("set k");
  program.add_argument("-n").default_value(512).store_into(N).help("set n");
  program.add_argument("--device_id")
      .default_value(0)
      .store_into(device_id)
      .help("set the device id");
  program.parse_args(argc, argv);

  PrecisionConfig int8 = {
      .func_name = "cublasGemmEx",
      .cudaType = CUDA_R_8I,
      .cublasType = CUBLAS_COMPUTE_32F,
      .bytesPerElement = sizeof(int8_t),
      .type_name = "int8",
      .NUM_ITERATIONS = 100,
      .WARMUP_ITERATIONS = 10,
      // .transa = CUBLAS_OP_N,
      // .transb = CUBLAS_OP_T,
      // .algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP
  };

  if (program["--append"] == false)
    std::cout << "func_name,dtype,algo,layout_a,layout_b,TOPS,ratio"
              << std::endl;

  for (int layout_a = 0; layout_a <= 1; layout_a++) {
    int8.transa = (cublasOperation_t)layout_a;
    for (int layout_b = 0; layout_b <= 1; layout_b++) {
      int8.transb = (cublasOperation_t)layout_b;
      for (int i = -1; i <= 23; i++) {
        int8.algo = (cublasGemmAlgo_t)i;
        test(int8);
      }

      for (int i = 99; i <= 115; i++) {
        int8.algo = (cublasGemmAlgo_t)i;
        test(int8);
      }
    }
  }

  std::cout << "best tflops: " << best_tflops << "("
            << best_tflops / int8_tflops << ")" << ", best algo: " << best_algo
            << ", best layout a: " << best_layout_a
            << ", best layout b: " << best_layout_b << std::endl;

  return 0;
}