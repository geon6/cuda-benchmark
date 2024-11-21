#ifndef __GEMM_CUBLASLT_H
#define __GEMM_CUBLASLT_H

void gemm_cublaslt_fp32();
void gemm_cublaslt_fp16();
void gemm_cublaslt_fp8();
void gemm_cublaslt_int8();
void gemm_cublaslt_bf16();

#endif