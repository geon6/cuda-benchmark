import subprocess
import argparse
import os
from tqdm import tqdm

# command: python range-test.py -m 0 5376 256 -n 0 5376 256 -k 0 5376 256 --device_id 5

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Run gemm-fp16 with input file, range parameters, and device ID, and redirect output to a file.")
    
    # 添加命令行参数
    parser.add_argument("-m", "--m_range", nargs=3, type=int, required=True, metavar=('start', 'stop', 'step'), help="Range for parameter m (start, stop, step)")
    parser.add_argument("-n", "--n_range", nargs=3, type=int, required=True, metavar=('start', 'stop', 'step'), help="Range for parameter n (start, stop, step)")
    parser.add_argument("-k", "--k_range", nargs=3, type=int, required=True, metavar=('start', 'stop', 'step'), help="Range for parameter k (start, stop, step)")
    parser.add_argument("--device_id", type=int, required=True, help="Device ID for gemm-fp16")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 定义可执行文件的路径
    executable_path = "./build/gemm/gemm-fp16"
    
    # 解析 range 参数
    m_start, m_stop, m_step = args.m_range
    n_start, n_stop, n_step = args.n_range
    k_start, k_stop, k_step = args.k_range

    # 计算总的迭代次数
    total_iterations = ((m_stop - m_start) // m_step) * ((n_stop - n_start) // n_step) * ((k_stop - k_start) // k_step)

    # 使用 tqdm 显示进度条
    with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
        for m in range(m_start, m_stop, m_step):
            for n in range(n_start, n_stop, n_step):
                for k in range(k_start, k_stop, k_step):
                    output_file = f"./output/gemm-fp16-m{m}-k{k}-n{n}.csv"
                    with open(output_file, "w") as f:
                        subprocess.run(
                            [executable_path, 
                             "-m", str(m), 
                             "-n", str(n), 
                             "-k", str(k), 
                             "--device_id", str(args.device_id)], 
                             stdout=f, stderr=subprocess.STDOUT)
                    pbar.update(1)
        print(f"所有命令成功执行，输出已保存到相应的文件中")

if __name__ == "__main__":
    main()