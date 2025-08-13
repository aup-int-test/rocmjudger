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
    N = 1
    array_data = [round(random.uniform(0, 10), 2) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 10
    array_data = [round(random.uniform(0, 10), 2) for _ in range(N)]
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 10
    array_data = [round(random.uniform(-10, 10), 2) for _ in range(N)]
    write_array_to_file('testcases/3.in', N, array_data)

def generate_testcase_4():
    N = 10
    array_data = [round(random.uniform(-10, 0), 2) for _ in range(N)]
    write_array_to_file('testcases/4.in', N, array_data)

def generate_testcase_5():
    N = 9
    array_data = [round(random.uniform(-10, 10), 2) for _ in range(N)]
    write_array_to_file('testcases/5.in', N, array_data)

def generate_testcase_6():
    N = 1024
    array_data = [round(random.uniform(-50.0, 50.0), 2) for _ in range(N)]
    write_array_to_file('testcases/6.in', N, array_data)

def generate_testcase_7():
    N = 1023
    array_data = [round(random.uniform(-50.0, 50.0), 2) for _ in range(N)]
    write_array_to_file('testcases/7.in', N, array_data)

def generate_testcase_8():
    N = 10000
    array_data = [round(random.uniform(-50.0, 0), 2) for _ in range(N)]
    write_array_to_file('testcases/8.in', N, array_data)

def generate_testcase_9():
    N = 1000000
    array_data = [round(random.uniform(-10, 10), 2) for _ in range(N)]
    write_array_to_file('testcases/9.in', N, array_data)

def generate_testcase_10():
    N = 1000000
    array_data = [round(random.uniform(-10, -1), 3) for _ in range(N)]
    write_array_to_file('testcases/10.in', N, array_data)

def generate_testcase_11():
    N = 10000000
    array_data = [-1.0] * N
    write_array_to_file('testcases/11.in', N, array_data)

def generate_testcase_12():
    N = 10000000
    array_data = [round(random.uniform(-10, 0), 3) for _ in range(N)]
    write_array_to_file('testcases/12.in', N, array_data)

def generate_testcase_13():
    N = 10000000
    array_data = [round(random.uniform(0, 10), 3) for _ in range(N)]
    write_array_to_file('testcases/13.in', N, array_data)

def generate_testcase_14():
    N = 100000000
    array_data = [round(random.uniform(-10.0, 10.0), 3) for _ in range(N-1)]
    array_data.append(100.0)  
    random.shuffle(array_data)
    write_array_to_file('testcases/14.in', N, array_data)

def generate_testcase_15():
    N = 100000000
    common_value = 15.0
    array_data = [common_value] * (N - 100)
    array_data.extend([round(random.uniform(-5.0, 15.0), 3) for _ in range(100)])
    random.shuffle(array_data)
    write_array_to_file('testcases/15.in', N, array_data)

def main():
    random.seed(2025)
    
    create_testcases_dir()
    
    generate_testcase_1()
    
    generate_testcase_2()
    
    generate_testcase_3()
    
    generate_testcase_4()
    
    generate_testcase_5()
    
    generate_testcase_6()
    
    generate_testcase_7()

    generate_testcase_8()

    generate_testcase_9()

    generate_testcase_10()

    generate_testcase_11()

    generate_testcase_12()

    generate_testcase_13()

    generate_testcase_14()

    generate_testcase_15()

    for i in range(1, 16):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()