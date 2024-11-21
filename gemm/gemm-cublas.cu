#include "gemm-cublas.h"

GemmCublas::GemmCublas(int m, int n, int k, cudaDataType_t data_type,
                       cublasGemmAlgo_t algo, cublasOperation_t transa,
                       cublasOperation_t transb, float alpha, float beta)
    : m_(m),
      n_(n),
      k_(k),
      data_type_(data_type),
      algo_(algo),
      transa_(transa),
      transb_(transb),
      alpha_(alpha),
      beta_(beta) {
  element_size_ = sizeof(data_type_);

  // 申请空间
  unsigned int mem_size_A = m * k * element_size_,
               mem_size_B = k * n * element_size_,
               mem_size_C = m * n * element_size_;
  // host memory
  void *h_A = malloc(mem_size_A), *h_B = malloc(mem_size_B),
       *h_C = malloc(mem_size_C);
  // device memory
  void *d_A = cudaMalloc((void**)&d_A, mem_size_A),
       *d_B = cudaMalloc((void**)&d_B, mem_size_B),
       *d_C = cudaMalloc((void**)&d_C, mem_size_C);

  // 对AB进行初始化, 并拷贝到GPU上 TODO
  size_t A_size = m * k, B_size = k * n;
  randomInit((GetType<data_type>::type*)h_A, A_size);
  randomInit((GetType<data_type>::type*)h_B, B_size);
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
}

GemmCublas::~GemmCublas() {
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void GemmCublas::run_gemm() {
  // warmup, 10次
  for (int i = 0; i < 10; i++) {
    cublasGemmEx(handle, transa_, transb_, m_, n_, k_, &alpha_, d_A, data_type_,
                 transa_ ? k_ : m_, d_B, data_type_, tranb_ ? n_ : k_, &beta_,
                 d_C, data_type_, m_, data_type_, algo_);
  }

  // benckmark, 100次取平均
  cudaEventRecord(start_, NULL);
  for (int i = 0; i < 100; i++) {
    cublasGemmEx(handle, transa_, transb_, m_, n_, k_, &alpha_, d_A, data_type_,
                 transa_ ? k_ : m_, d_B, data_type_, tranb_ ? n_ : k_, &beta_,
                 d_C, data_type_, m_, data_type_, algo_);
  }
  cudaEventRecord(stop_, NULL);
  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

  // 运行时间, gflops, 设备利用率
  float ms_total = 0.0f;
  cudaEventElapsedTime(&ms_total, start_, stop_);
  float ms_per_mm = ms_total / 100.0;  // 100次取平均
  double flops_per_mm = 2.0 * m * n * k;
  double gigaFlops = (flops_per_mm * 1.0e-9f) / (ms_per_mm / 1000.0f);

  std::cout << "gflops: " << gigaFlops << std::endl;
  std::cout << "msec per mm: " << msecPerMatrixMul << std::endl;
  std::cout << "flops per mm: " << flops_per_mm << std::endl;

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  std::cout << "device: " << prop.name << std::endl;
  std::cout << "compute capability: " << prop.major << "." << prop.minor
            << std::endl;

  unsigned int gpuUtil, memoryUtil;
  getDeviceUtilization(device, gpuUtil, memoryUtil);
  std::cout << "GPU Utilization: " << gpuUtil << "%" << std::endl;
  std::cout << "Memory Utilization: " << memoryUtil << "%" << std::endl;
}

void getDeviceUtilization(int deviceIndex, unsigned int& gpuUtil,
                          unsigned int& memoryUtil) {
  nvmlDevice_t device;
  nvmlReturn_t result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
  if (NVML_SUCCESS != result) {
    throw std::runtime_error("Failed to get device handle: " +
                             std::string(nvmlErrorString(result)));
  }

  nvmlUtilization_t utilization;
  result = nvmlDeviceGetUtilizationRates(device, &utilization);
  if (NVML_SUCCESS != result) {
    throw std::runtime_error("Failed to get utilization rates: " +
                             std::string(nvmlErrorString(result)));
  }

  gpuUtil = utilization.gpu;
  memoryUtil = utilization.memory;
}