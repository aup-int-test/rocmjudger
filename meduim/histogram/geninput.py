import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_histogram_to_file(filename, N, num_bins, array_data):
    with open(filename, 'w') as f:
        f.write(f"{N} {num_bins}\n")
        f.write(" ".join(map(str, array_data)) + "\n")

def generate_testcase_1():
    N = 1
    num_bins = 10
    array_data = [random.randint(0, num_bins-1)]
    write_histogram_to_file('testcases/1.in', N, num_bins, array_data)

def generate_testcase_2():
    N = 10000000
    num_bins = 64
    array_data = [random.randint(0, num_bins-1) for _ in range(N)]
    write_histogram_to_file('testcases/2.in', N, num_bins, array_data)

def generate_testcase_3():
    N = 10000000
    num_bins = 512
    
    filename = 'testcases/3.in'
    with open(filename, 'w') as f:
        f.write(f"{N} {num_bins}\n")
        
        batch_size = 100000
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_data = [random.randint(0, num_bins-1) for _ in range(batch_end - batch_start)]
            if batch_start == 0:
                f.write(" ".join(map(str, batch_data)))
            else:
                f.write(" " + " ".join(map(str, batch_data)))
        f.write("\n")

def generate_testcase_4():
    N = 100000000
    num_bins = 512
    
    filename = 'testcases/4.in'
    with open(filename, 'w') as f:
        f.write(f"{N} {num_bins}\n")
        
        batch_size = 100000
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_data = [random.randint(0, num_bins-1) for _ in range(batch_end - batch_start)]
            if batch_start == 0:
                f.write(" ".join(map(str, batch_data)))
            else:
                f.write(" " + " ".join(map(str, batch_data)))
        f.write("\n")

def generate_testcase_5():
    N = 100000000
    num_bins = 256
    same_value = 256
    array_data = [same_value] * N
    write_histogram_to_file('testcases/5.in', N, num_bins, array_data)

def generate_testcase_6():
    N = 100000000  
    num_bins = 1024
    array_data = [i % num_bins for i in range(N)]  
    write_histogram_to_file('testcases/6.in', N, num_bins, array_data)


def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    generate_testcase_1()
    
    generate_testcase_2()

    generate_testcase_3()
    
    generate_testcase_4()
    
    generate_testcase_5()
    
    generate_testcase_6()
    
    
    # 顯示檔案大小
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")
            
            # 簡單驗證檔案格式
            with open(filename, 'r') as f:
                first_line = f.readline().strip().split()
                N, num_bins = int(first_line[0]), int(first_line[1])
                print(f"    N={N:,}, num_bins={num_bins}")

if __name__ == "__main__":
    main()