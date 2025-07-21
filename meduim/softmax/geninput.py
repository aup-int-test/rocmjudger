import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_array_to_file(filename, N, array_data):
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_1():
    N = 100
    array_data = [round(random.uniform(0.1, 10.0), 4) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 10000
    array_data = [round(random.uniform(-50.0, 50.0), 4) for _ in range(N)]
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 1048576
    array_data = [-1.0] * N
    write_array_to_file('testcases/3.in', N, array_data)

def generate_testcase_4():
    N = 1048576
    array_data = [round(random.uniform(-10.0, 10.0), 4) for _ in range(N-1)]
    array_data.append(100.0)  # 單一最大值
    random.shuffle(array_data)
    write_array_to_file('testcases/4.in', N, array_data)

def generate_testcase_5():
    N = 1048576
    common_value = 5.0
    array_data = [common_value] * (N - 100)
    array_data.extend([round(random.uniform(-5.0, 15.0), 4) for _ in range(100)])
    random.shuffle(array_data)
    write_array_to_file('testcases/5.in', N, array_data)

def generate_testcase_6():
    N = 1048576
    array_data = [round(random.uniform(-50.0, -1.0), 4) for _ in range(int(N*0.9))]
    array_data.extend([round(random.uniform(-1.0, 10.0), 4) for _ in range(int(N*0.1))])
    random.shuffle(array_data)
    write_array_to_file('testcases/6.in', N, array_data)

def generate_testcase_7():
    N = 16777216
    
    filename = 'testcases/7.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [round(random.uniform(0.1, 10.0), 4) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_8():
    N = 134217728
    
    filename = 'testcases/8.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [round(random.uniform(-200.0, 200.0), 6) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 100 全正數 ===")
    generate_testcase_1()
    
    print("=== 2.in: 10000 隨機 ===")
    generate_testcase_2()
    
    print("=== 3.in: 1048576 全-1向量 ===")
    generate_testcase_3()
    
    print("=== 4.in: 1048576 單一最大值明顯 ===")
    generate_testcase_4()
    
    print("=== 5.in: 1048576 大量相同值 ===")
    generate_testcase_5()
    
    print("=== 6.in: 1048576 大多負數 ===")
    generate_testcase_6()
    
    print("=== 7.in: 16777216 ===")
    generate_testcase_7()
    
    print("=== 8.in: 134217728 ===")
    generate_testcase_8()
    
    for i in range(1, 9):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()