import random
import os

def create_testcases_dir():
    if not os.path.exists('testcases'):
        os.makedirs('testcases')

def write_image_to_file(filename, width, height, image_data):
    with open(filename, 'w') as f:
        f.write(f"{width} {height}\n")
        for i in range(width * height * 4):
            f.write(f"{image_data[i]} ")
            if (i + 1) % (width * 4) == 0:
                f.write("\n")

def generate_testcase_1():
    width, height = 1, 1
    image_data = [random.randint(0, 255) for _ in range(width * height * 4)]
    write_image_to_file('testcases/1.in', width, height, image_data)

def generate_testcase_2():
    width, height = 64, 64
    image_data = [random.randint(0, 255) for _ in range(width * height * 4)]
    write_image_to_file('testcases/2.in', width, height, image_data)

def generate_testcase_3():
    width, height = 257, 257
    image_data = [random.randint(0, 255) for _ in range(width * height * 4)]
    write_image_to_file('testcases/3.in', width, height, image_data)

def generate_testcase_4():
    width, height = 1001, 2001
    
    filename = 'testcases/4.in'
    with open(filename, 'w') as f:
        f.write(f"{width} {height}\n")
        
        for i in range(width * height):
            pixel_data = [random.randint(0, 255) for _ in range(4)]
            f.write(" ".join(map(str, pixel_data)) + " ")
            if (i + 1) % width == 0:
                f.write("\n")

def generate_testcase_5():
    width, height = 4096, 2048
    
    filename = 'testcases/5.in'
    with open(filename, 'w') as f:
        f.write(f"{width} {height}\n")
        
        for i in range(width * height):
            pixel_data = [random.randint(0, 255) for _ in range(4)]
            f.write(" ".join(map(str, pixel_data)) + " ")
            if (i + 1) % width == 0:
                f.write("\n")

def main():
    random.seed(77777)
    
    create_testcases_dir()
    
    print("=== 1.in: 1*1 ===")
    generate_testcase_1()
    
    print("=== 2.in: 64*64 ===")
    generate_testcase_2()
    
    print("=== 3.in: 257*257 ===")
    generate_testcase_3()
    
    print("=== 4.in: 1001*2001 ===")
    generate_testcase_4()
    
    print("=== 5.in: 4096*2048 ===")
    generate_testcase_5()

    for i in range(1, 6):
        filename = f"testcases/{i}.in"
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"  {filename}: {file_size:,} bytes")

if __name__ == "__main__":
    main()