import os
import subprocess
import argparse
from tqdm import tqdm


def format_files(directory, extensions, config_file, clang_format_path):
    # 获取所有需要格式化的文件
    files_to_format = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                files_to_format.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for file_path in tqdm(files_to_format, desc="Formatting files", unit="file"):
        # 调用 clang-format 命令，并指定配置文件
        try:
            subprocess.run(
                [clang_format_path, "-i", "--style=file", file_path],
                check=True,
                env={"CLANG_FORMAT_STYLE_FILE": config_file},
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to format: {file_path} - {e}")


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Format C/C++/CUDA files using clang-format."
    )

    # 添加命令行参数
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default=".",
        help="Directory to format (default: current directory)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=os.path.join(os.getcwd(), ".clang-format"),
        help="Path to the .clang-format configuration file (default: ./.clang-format)",
    )
    parser.add_argument(
        "--clang-format-path",
        type=str,
        default="/opt/homebrew/bin/clang-format",
        help="Path to the clang-format executable (default: clang-format)",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 指定需要处理的文件扩展名
    file_extensions = [".c", ".cpp", ".cc", ".h", ".hpp", ".cu"]

    # 调用格式化函数
    format_files(
        args.directory, file_extensions, args.config_file, args.clang_format_path
    )
