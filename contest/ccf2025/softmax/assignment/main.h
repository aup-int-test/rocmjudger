#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <float.h>
#include <fstream>

extern "C" void solve(const float* input, float* output, int N);

#endif 