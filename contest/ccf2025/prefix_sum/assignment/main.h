#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <fstream>

extern "C" void solve(const int* input, int* output, int N);

#endif 