# ROCm Judger Challenge Bank

## Overview
This repository contains a collection of benchmark and challenge problems designed for future ROCm HIP Online Judge (OJ) use.  
It provides a standardized set of solutions and build scripts to help evaluate HIP-based GPU programming tasks.

## Repository Structure
- **main.hip**  
  The official reference solution for each challenge, serving as the standard implementation.

- **fs_main.hip**  
  A version of the solution with explicit file I/O handling.

- **fs_xxx.hip**  
  Alternative or experimental solutions (e.g., optimized variants or different algorithmic approaches).  

- **Makefile**  
  Build script to compile the HIP source files and generate executables.  
  Supports common targets such as `make`, `make clean`, etc.

- **geninput.py**
  Python script for input generation.

- **README.md**
  Challenge description.

3. Run the compiled executable with appropriate input files or test cases.
