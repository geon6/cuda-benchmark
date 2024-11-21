#include <iostream>

#include "gemm-common.h"
#include "gemm.h"

int main(int argc, char** argv) {
  GemmContext context;
  int m = 1024, n = 1024, k = 1024;
  // cudaDataType_t data_type = CUDA_R_16F;
  // cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
  for (auto [dtype_str, dtype] : cublas_iter::data_types) {
    for (auto [algo_str, algo] : cublas_iter::algos) {
      std::cout << "---- " << "dtype: " << dtype_str << ", algo: " << algo_str << " ----\n";
      GemmCublas gemm_cublas(m, n, k, dtype, algo, transa, transb);
      gemm_cublas.run_gemm();
    }
  }
}