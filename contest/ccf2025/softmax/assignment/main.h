#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <vector>
#include <float.h>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <fstream>

extern "C" void solve(const float* input, float* output, int N);

#endif 