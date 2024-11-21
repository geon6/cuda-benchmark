#ifndef __GEMM_COMMON_H
#define __GEMM_COMMON_H

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// 检查CUDA错误
#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cout << "CUDA API failed at " << __FILE__ << ":" << __LINE__        \
                << " with error: " << cudaGetErrorString(status) << std::endl; \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  }

// 检查cuBLAS错误
#define CHECK_CUBLAS(func)                                                \
  {                                                                       \
    cublasStatus_t status = (func);                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                                \
      std::cout << "cuBLAS API failed at " << __FILE__ << ":" << __LINE__ \
                << " with error: " << status << std::endl;                \
      return EXIT_FAILURE;                                                \
    }                                                                     \
  }

// 检查NVML错误
#define CHECK_NVML(func)                                                    \
  {                                                                         \
    nvmlReturn_t status = (func);                                           \
    if (status != NVML_SUCCESS) {                                           \
      std::cout << "NVML API failed at " << __FILE__ << ":" << __LINE__     \
                << " with error: " << nvmlErrorString(status) << std::endl; \
      return EXIT_FAILURE;                                                  \
    }                                                                       \
  }

// 初始化数据
using fp8 = __nv_fp8_e4m3;
// using fp8_e5m2 = __nv_fp8_e5m2;
using fp16 = __half;
using fp32 = float;
using bf16 = __nv_bfloat16;
using int8 = int8_t;

template <cudaDataType_t T>
struct GetType {
  using type = void;
};

template <>
struct GetType<CUDA_R_8F_E4M3> {
  using type = fp8;
};

// 不使用e5m2
// template <>
// struct GetType<CUDA_R_8F_E5M2> {
//   using type = fp8_e5m2;
// };

template <>
struct GetType<CUDA_R_16F> {
  using type = fp16;
};

template <>
struct GetType<CUDA_R_32F> {
  using type = fp32;
};

template <>
struct GetType<CUDA_R_16BF> {
  using type = bf16;
};

template <>
struct GetType<CUDA_R_8I> {
  using type = int8;
};

template <typename T>
void randomInit(T* data, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<T>(dis(gen));
  }
}

template <>
void randomInit<fp8>(fp8* data, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < size; ++i) {
    data[i] = __float2fp8_e4m3(dis(gen));
  }
}

template <>
void randomInit<fp16>(fp16* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(dis(gen));
    }
}

template <>
void randomInit<bf16>(bf16* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2bfloat16(dis(gen));
    }
}

template <>
void randomInit<int8>(int8* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int8> dis(std::numeric_limits<int8>::min(), std::numeric_limits<int8>::max());

    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// 用RAII进行cuda context管理
struct GemmContext {
  GemmContext() {
    CHECK_CUDA(cublasCreate(&handle_));
    CHECK_NVML(nvmlInit());
  }
  ~GemmContext() {
    CHECK_CUDA(cublasDestroy(handle_));
    CHECK_NVML(nvmlShutdown());
  }
  cublasHandle_t handle_;
};

// 暂时没用
struct GemmData {
  GemmData(int m, int n, int k, cudaDataType_t data_type)
      : m_(m), n_(n), k_(k), data_type_(data_type), {
    element_size_ = sizeof(data_type_);

    // // 申请空间
    // unsigned int mem_size_A = m * k * element_size_,
    //              mem_size_B = k * n * element_size_,
    //              mem_size_C = m * n * element_size_;
    // // host memory
    // void *h_A = malloc(mem_size_A), *h_B = malloc(mem_size_B),
    //      *h_C = malloc(mem_size_C);
    // // device memory
    // void *d_A = cudaMalloc((void**)&d_A, mem_size_A),
    //      *d_B = cudaMalloc((void**)&d_B, mem_size_B),
    //      *d_C = cudaMalloc((void**)&d_C, mem_size_C);

    // // 对AB进行初始化, 并拷贝到GPU上 TODO
    // randomInit(h_A, mem_size_A);
    // randomInit(h_B, mem_size_B);
    // cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
  }

  GemmData::~GemmData() {
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }

  // 矩阵的大小
  int m_, n_, k_;
  // host内存
  void *h_A, *h_B, *h_C;
  // device 内存
  void *d_A, *d_B, *d_C;
  unsigned int element_size_;  // 数据大小, 单位是字节
  cudaDataType_t data_type_;   // 数据类型
};

namespace cublas_iter {

// CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUDA_R_8I, CUDA_R_8F_E4M3,
// CUDA_R_8F_E5M2
std::vector<std::pair<std::string, cudaDataType_t>> data_types{
    {"CUDA_R_16F", CUDA_R_16F},         {"CUDA_R_16BF", CUDA_R_16BF},
    {"CUDA_R_32F", CUDA_R_32F},         {"CUDA_R_8I", CUDA_R_8I},
    {"CUDA_R_8F_E4M3", CUDA_R_8F_E4M3},
    //  {"CUDA_R_8F_E5M2", CUDA_R_8F_E5M2}
};

std::vector<std::pair<std::string, cublasGemmAlgo_t>> algos{
    {"CUBLAS_GEMM_DEFAULT", CUBLAS_GEMM_DEFAULT},
    {"CUBLAS_GEMM_ALGO0", CUBLAS_GEMM_ALGO0},
    {"CUBLAS_GEMM_ALGO1", CUBLAS_GEMM_ALGO1},
    {"CUBLAS_GEMM_ALGO2", CUBLAS_GEMM_ALGO2},
    {"CUBLAS_GEMM_ALGO3", CUBLAS_GEMM_ALGO3},
    {"CUBLAS_GEMM_ALGO4", CUBLAS_GEMM_ALGO4},
    {"CUBLAS_GEMM_ALGO5", CUBLAS_GEMM_ALGO5},
    {"CUBLAS_GEMM_ALGO6", CUBLAS_GEMM_ALGO6},
    {"CUBLAS_GEMM_ALGO7", CUBLAS_GEMM_ALGO7},
    {"CUBLAS_GEMM_ALGO8", CUBLAS_GEMM_ALGO8},
    {"CUBLAS_GEMM_ALGO9", CUBLAS_GEMM_ALGO9},
    {"CUBLAS_GEMM_ALGO10", CUBLAS_GEMM_ALGO10},
    {"CUBLAS_GEMM_ALGO11", CUBLAS_GEMM_ALGO11},
    {"CUBLAS_GEMM_ALGO12", CUBLAS_GEMM_ALGO12},
    {"CUBLAS_GEMM_ALGO13", CUBLAS_GEMM_ALGO13},
    {"CUBLAS_GEMM_ALGO14", CUBLAS_GEMM_ALGO14},
    {"CUBLAS_GEMM_ALGO15", CUBLAS_GEMM_ALGO15},
    {"CUBLAS_GEMM_ALGO16", CUBLAS_GEMM_ALGO16},
    {"CUBLAS_GEMM_ALGO17", CUBLAS_GEMM_ALGO17},
    {"CUBLAS_GEMM_ALGO18", CUBLAS_GEMM_ALGO18},
    {"CUBLAS_GEMM_ALGO19", CUBLAS_GEMM_ALGO19},
    {"CUBLAS_GEMM_ALGO20", CUBLAS_GEMM_ALGO20},
    {"CUBLAS_GEMM_ALGO21", CUBLAS_GEMM_ALGO21},
    {"CUBLAS_GEMM_ALGO22", CUBLAS_GEMM_ALGO22},
    {"CUBLAS_GEMM_ALGO23", CUBLAS_GEMM_ALGO23}};

}  // namespace cublas_iter

#endif