import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_vectors_to_file(filename, vector_a, vector_b):
    with open(filename, 'w') as f:
        f.write(f"{len(vector_a)}\n")
        f.write(" ".join(map(str, vector_a)) + "\n")
        f.write(" ".join(map(str, vector_b)) + "\n")

def generate_testcase_1():
    size = 100
    vector_a = [random.randint(-1000, 1000) for _ in range(size)]
    vector_b = [random.randint(-1000, 1000) for _ in range(size)]
    write_vectors_to_file('testcases/1.in', vector_a, vector_b)

def generate_testcase_2():
    size = 10000
    vector_a = [round(random.uniform(-100.0, 100.0), 2) for _ in range(size)]
    vector_b = [round(random.uniform(-100.0, 100.0), 2) for _ in range(size)]
    write_vectors_to_file('testcases/2.in', vector_a, vector_b)

def generate_testcase_3():
    size = 10000
    vector_a = [0.0] * size
    vector_b = [0.0] * size
    write_vectors_to_file('testcases/3.in', vector_a, vector_b)

def generate_testcase_4():
    size = 10000
    vector_a = [round(random.uniform(0.1, 100.0), 2) for _ in range(size)]
    vector_b = [round(random.uniform(-100.0, -0.1), 2) for _ in range(size)]
    write_vectors_to_file('testcases/4.in', vector_a, vector_b)

def generate_testcase_5():
    size = 1000000
    
    filename = 'testcases/5.in'
    with open(filename, 'w') as f:
        f.write(f"{size}\n")
        
        batch_size = 1000000  
        batches = size // batch_size
        remaining = size % batch_size
        
        vector_a_parts = []
        for i in range(batches):
            batch = [round(random.uniform(-999999.0, 999999.0), 6) for _ in range(batch_size)]
            vector_a_parts.extend(batch)
        
        if remaining > 0:
            batch = [round(random.uniform(-999999.0, 999999.0), 6) for _ in range(remaining)]
            vector_a_parts.extend(batch)
        
        f.write(" ".join(map(str, vector_a_parts)) + "\n")
        
        vector_b_parts = []
        for i in range(batches):
            batch = [round(random.uniform(-999999.0, 999999.0), 6) for _ in range(batch_size)]
            vector_b_parts.extend(batch)
        
        if remaining > 0:
            batch = [round(random.uniform(-999999.0, 999999.0), 6) for _ in range(remaining)]
            vector_b_parts.extend(batch)
        
        f.write(" ".join(map(str, vector_b_parts)) + "\n")

def generate_testcase_6():
    size = 1000
    
    vector_a = []
    vector_b = []
    
    special_values = [0.0, 1.0, -1.0, 0.000001, -0.000001, 
                     123.456789, -123.456789, 999999.999999, -999999.999999]
    
    for i in range(size):
        if i < len(special_values):
            vector_a.append(special_values[i])
            vector_b.append(special_values[-(i+1)])  
        else:
            vector_a.append(round(random.uniform(-1000.0, 1000.0), 6))
            vector_b.append(round(random.uniform(-1000.0, 1000.0), 6))
    
    write_vectors_to_file('testcases/6.in', vector_a, vector_b)

def main():
    
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 100個整數 ===")
    generate_testcase_1()
    
    print("=== 2.in: 10000個兩位浮點數 ===")
    generate_testcase_2()
    
    print("=== 3.in: 10000個零向量 ===")
    generate_testcase_3()
    
    print("=== 4.in: 10000個正負向量 ===")
    generate_testcase_4()
    
    print("=== 5.in: 1000000個六位浮點數 ===")
    generate_testcase_5()
    
    print("=== 6.in: 極大/極小數 ===")
    generate_testcase_6()
    
    print("done!")
    for i in range(1, 7):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()