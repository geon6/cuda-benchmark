# cuda-benchmark
用来测试GPU速度峰值. 在A40上进行测试, A40的标称TFLOPS为: 
* bf16: 149.7
* fp16: 149.7
* fp32: 37.4
* int8: 299.3

> 数据来源于[NVIDIA A40 data sheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)

## Executable
* gemm-bf16: 输入数据, 输出数据都是bf16
* gemm-fp16: 输入数据, 输出数据都是bf16
* gemm-fp32: 输入数据, 输出数据都是bf16
* gemm-int8-fp32: 输入数据是int8, 输出数据是fp32
* gemm-int8-int32: 输入数据是int8, 输出数据是int32