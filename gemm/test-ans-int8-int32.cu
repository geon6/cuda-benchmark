#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "helper_cuda.h"  // 假设包含了checkCudaErrors函数定义的头文件

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

void transposeMatrix(int32_t* matrix, int numRows, int numCols) {
  int32_t* transposedMatrix = new int32_t[numRows * numCols];
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      transposedMatrix[j * numRows + i] = matrix[i * numCols + j];
    }
  }
  // 将转置后的矩阵数据复制回原矩阵存储空间
  for (int i = 0; i < numRows * numCols; ++i) {
    matrix[i] = transposedMatrix[i];
  }
  delete[] transposedMatrix;
}

// CPU上实现矩阵乘法的函数，这里简单实现两个矩阵相乘，你可以根据实际情况优化或者替换更高效的实现方式
void cpuMatrixMultiply(int8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int32_t sum = 0;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// 检查两个int32_t类型矩阵乘法结果之间的误差是否在可接受范围内
bool checkMatrixMultiplyError(const int32_t* matrix1, const int32_t* matrix2,
                              int numRows, int numCols, double threshold) {
  double sumSquaredError = 0.0;
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      int32_t diff = matrix1[i * numCols + j] - matrix2[i * numCols + j];
      sumSquaredError += static_cast<double>(diff * diff);
    }
  }
  double rmse = std::sqrt(sumSquaredError / (numRows * numCols));
  return rmse <= threshold;
}

// 输出int32_t类型矩阵的函数
void printMatrix(const int32_t* matrix, int numRows, int numCols) {
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      printf("%d ", matrix[i * numCols + j]);
    }
    printf("\n");
  }
}

int main() {
  // 定义矩阵维度
  int m = 16;
  int n = 16;
  int k = 16;

  // 分配主机内存用于存储矩阵数据
  int8_t* h_A = (int8_t*)malloc(m * k * sizeof(int8_t));
  int8_t* h_B = (int8_t*)malloc(k * n * sizeof(int8_t));
  // 用于存放GPU计算结果
  int32_t* h_C_gpu = (int32_t*)malloc(m * n * sizeof(int32_t));
  // 用于存放CPU计算结果
  int32_t* h_C_cpu = (int32_t*)malloc(m * n * sizeof(int32_t));

  // 初始化矩阵数据（这里简单示例为随机赋值，范围在int8_t的取值区间内）
  // 生成 -64 到 63 之间的随机数作为int8_t的值
  for (int i = 0; i < m * k; i++) {
    h_A[i] = (int8_t)(rand() % 128 - 64);
  }
  for (int i = 0; i < k * n; i++) {
    h_B[i] = (int8_t)(rand() % 128 - 64);
  }

  // 分配设备内存
  int8_t *d_A, *d_B;
  int32_t* d_C;
  checkCudaErrors(cudaMalloc((void**)&d_A, m * k * sizeof(int8_t)));
  checkCudaErrors(cudaMalloc((void**)&d_B, k * n * sizeof(int8_t)));
  checkCudaErrors(cudaMalloc((void**)&d_C, m * n * sizeof(int32_t)));

  // 将数据从主机内存复制到设备内存
  checkCudaErrors(
      cudaMemcpy(d_A, h_A, m * k * sizeof(int8_t), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_B, h_B, k * n * sizeof(int8_t), cudaMemcpyHostToDevice));

  // 创建CUBLAS句柄
  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));

  // 执行矩阵乘法（GPU上）
  int32_t alpha = 1;
  int32_t beta = 0;
  PrecisionConfig int8 = {
      .func_name = "cublasGemmEx",
      .cudaType = CUDA_R_8I,
      .cublasType = CUBLAS_COMPUTE_32I,
      .bytesPerElement = sizeof(int8_t),
      .type_name = "int8",
      .NUM_ITERATIONS = 100,
      .WARMUP_ITERATIONS = 10,
      .transa = CUBLAS_OP_N,
      //  .transa = CUBLAS_OP_T,
      //  .transb = CUBLAS_OP_N,
      .transb = CUBLAS_OP_T,
      .algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP
      //  .algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP
  };
  checkCudaErrors(
      cublasGemmEx(handle, int8.transa, int8.transb, m, n, k, &alpha, d_A,
                   int8.cudaType, (int8.transa == CUBLAS_OP_N ? m : k), d_B,
                   int8.cudaType, (int8.transb == CUBLAS_OP_N ? k : n), &beta,
                   d_C, CUDA_R_32I, m, int8.cublasType, int8.algo));

  // 进行设备同步，确保GPU上的矩阵乘法操作完成
  checkCudaErrors(cudaDeviceSynchronize());

  // 将结果从设备内存复制回主机内存
  checkCudaErrors(cudaMemcpy(h_C_gpu, d_C, m * n * sizeof(int32_t),
                             cudaMemcpyDeviceToHost));

  // 在CPU上执行矩阵乘法
  cpuMatrixMultiply(h_A, h_B, h_C_cpu, m, n, k);
  checkCudaErrors(cudaDeviceSynchronize());

  // 输出CPU计算得到的矩阵结果
  printf("CPU计算结果矩阵 h_C_cpu:\n");
  printMatrix(h_C_cpu, m, n);

  // 输出GPU计算得到的矩阵结果
  printf("GPU计算结果矩阵 h_C_gpu:\n");
  printMatrix(h_C_gpu, m, n);

  transposeMatrix(h_C_cpu, h_C_gpu, m, n);
  // 检查GPU和CPU计算结果的正确性，假设epsilon设为一个较小的值，比如1e-5，你可以根据实际精度需求调整
  bool resultCompare = checkMatrixMultiplyError(h_C_cpu, h_C_gpu, m, n, 1.0);
  if (resultCompare) {
    printf("GPU和CPU计算结果匹配，矩阵乘法计算正确。\n");
  } else {
    printf("GPU和CPU计算结果不匹配，矩阵乘法计算可能存在问题。\n");
  }

  // 释放设备内存
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  // 释放主机内存
  free(h_A);
  free(h_B);
  free(h_C_gpu);
  free(h_C_cpu);
  // 销毁CUBLAS句柄
  checkCudaErrors(cublasDestroy(handle));

  return 0;
}