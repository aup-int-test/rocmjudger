#!/usr/bin/env python3
import random
import os
import sys

def generate_test_case(n, output_file=None, test_type="random"):
    """
    生成測試數據
    
    Args:
        n: 數組大小
        output_file: 輸出文件名，如果為 None 則輸出到標準輸出
        test_type: 測試類型 ("random", "sequential", "same", "edge")
    """
    
    if test_type == "random":
        # 隨機浮點數 (-100.0 到 100.0)
        data = [random.uniform(-100.0, 100.0) for _ in range(n)]
    elif test_type == "sequential":
        # 連續整數轉浮點數
        data = [float(i + 1) for i in range(n)]
    elif test_type == "same":
        # 所有相同的值
        value = 0
        data = [value] * n
    elif test_type == "edge":
        # 邊界情況：包含很大和很小的數
        data = []
        for i in range(n):
            if i % 3 == 0:
                data.append(1e6)  # 大數
            elif i % 3 == 1:
                data.append(-1e6)  # 負大數
            else:
                data.append(random.uniform(-1.0, 1.0))  # 小數
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # 準備輸出 (單行格式)
    output_parts = [str(n)] + [f"{x:.6f}" for x in data]
    output_content = " ".join(output_parts)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_content + "\n")
        print(f"測試數據已生成: {output_file}")
        print(f"數組大小: {n}")
        print(f"測試類型: {test_type}")
        print(f"預期結果: {sum(data):.6f}")
    else:
        print(output_content)
    
    return sum(data)

def generate_multiple_tests():
    """生成多個不同規模的測試檔案"""
    
    test_cases = [
        (10, "small", "random"),
        (256, "medium_block", "sequential"),  # 正好一個 block
        (1000, "medium", "random"),
        (2560, "large_10blocks", "same"),     # 10 個 blocks
        (10000, "large", "edge"),
        (100000, "huge", "random"),
    ]
    
    os.makedirs("test_data", exist_ok=True)
    
    for size, name, test_type in test_cases:
        filename = f"test_data/test_{name}_{size}.txt"
        expected = generate_test_case(size, filename, test_type)
        
        # 同時生成答案檔案
        answer_file = f"test_data/answer_{name}_{size}.txt"
        with open(answer_file, 'w') as f:
            f.write(f"{expected:.6f}\n")

def main():
    if len(sys.argv) == 1:
        print("HIP 並行歸約測試數據生成器")
        print("\n使用方式:")
        print(f"  {sys.argv[0]} <size>                    # 生成指定大小的隨機測試數據到標準輸出")
        print(f"  {sys.argv[0]} <size> <type>             # 指定測試類型")
        print(f"  {sys.argv[0]} <size> <type> <filename>  # 輸出到文件")
        print(f"  {sys.argv[0]} batch                     # 生成多個測試檔案")
        print("\n測試類型:")
        print("  random     - 隨機浮點數")
        print("  sequential - 連續整數")
        print("  same       - 所有相同值")
        print("  edge       - 邊界情況")
        print("\n範例:")
        print(f"  {sys.argv[0]} 1000 > input.txt")
        print(f"  {sys.argv[0]} 256 sequential test_seq.txt")
        print(f"  {sys.argv[0]} batch")
        return
    
    if sys.argv[1] == "batch":
        generate_multiple_tests()
        print("\n所有測試檔案已生成在 test_data/ 目錄中")
        return
    
    try:
        n = int(sys.argv[1])
        test_type = sys.argv[2] if len(sys.argv) > 2 else "random"
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        if n <= 0:
            print("錯誤: 數組大小必須為正整數", file=sys.stderr)
            sys.exit(1)
            
        generate_test_case(n, output_file, test_type)
        
    except ValueError as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"未知錯誤: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()