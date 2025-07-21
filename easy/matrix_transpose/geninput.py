import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_matrix_to_file(filename, rows, cols, matrix):
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        for i in range(rows):
            for j in range(cols):
                f.write(f"{matrix[i * cols + j]} ")
            f.write("\n")

def generate_testcase_1():
    rows, cols = 16, 16
    matrix = [round(random.uniform(-10.0, 10.0), 2) for _ in range(rows * cols)]
    write_matrix_to_file('testcases/1.in', rows, cols, matrix)

def generate_testcase_2():
    rows, cols = 128, 32
    matrix = [round(random.uniform(-100.0, 100.0), 2) for _ in range(rows * cols)]
    write_matrix_to_file('testcases/2.in', rows, cols, matrix)

def generate_testcase_3():
    rows, cols = 1, 1024
    matrix = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(rows * cols)]
    write_matrix_to_file('testcases/3.in', rows, cols, matrix)

def generate_testcase_4():
    rows, cols = 1001, 2001
    
    filename = 'testcases/4.in'
    with open(filename, 'w') as f:
        f.write(f"{rows} {cols}\n")
        
        for i in range(rows):
            row_data = [round(random.uniform(-1000.0, 1000.0), 6) for _ in range(cols)]
            f.write(" ".join(map(str, row_data)) + "\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 16 * 16 ===")
    generate_testcase_1()
    
    print("=== 2.in: 128 * 32 ===")
    generate_testcase_2()
    
    print("=== 3.in: 1 * 1024 ===")
    generate_testcase_3()
    
    print("=== 4.in: 1001 * 2001 ===")
    generate_testcase_4()
    
    for i in range(1, 5):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()