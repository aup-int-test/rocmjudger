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
    N = 10
    array_data = [random.randint(-10, 10) for _ in range(N)]
    write_array_to_file('testcases/1.in', N, array_data)

def generate_testcase_2():
    N = 1000000
    array_data = [random.randint(-100, -50) for _ in range(N)]
    write_array_to_file('testcases/2.in', N, array_data)

def generate_testcase_3():
    N = 100000000
    
    filename = 'testcases/3.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [random.randint(-1000, -500) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_4():
    N = 1000000000
    
    filename = 'testcases/4.in'
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        
        array_data = [random.randint(-10000, 10000) for _ in range(N)]
        f.write(" ".join(map(str, array_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    generate_testcase_1()
    
    generate_testcase_2()
    
    generate_testcase_3()
    
    generate_testcase_4()
    
    for i in range(1, 5):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()