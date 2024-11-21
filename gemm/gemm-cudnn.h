#ifndef __GEMM_CUDNN_H
#define __GEMM_CUDNN_H

void gemm_cudnn_fp32();
void gemm_cudnn_fp16();
void gemm_cudnn_fp8();
void gemm_cudnn_int8();
void gemm_cudnn_bf16();

#endif