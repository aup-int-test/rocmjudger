// fast.hip — Blocked Floyd–Warshall (HIP)
// Input:  text integers: V E, then E lines of "u v w"
// Output: stdout prints VxV ints (space-separated), one row per line

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE   64
#define THREAD_SIZE  32
#define INF ((1 << 30) - 1)

#define HIP_CHECK(cmd) do { \
  hipError_t _e = (cmd); \
  if (_e != hipSuccess) { \
    std::fprintf(stderr, "%s failed: %s (%d)\n", #cmd, hipGetErrorString(_e), (int)_e); \
    std::exit(1); \
  } \
} while(0)

int V, E, N;
int *deviceDist = nullptr;

__global__ void initializeDeviceDist(int *deviceDist, int N) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N && j < N) {
    deviceDist[i * N + j] = (i == j) ? 0 : INF;
  }
}

__global__ void updateDeviceDist(int *deviceDist, const int *deviceLocalEdges, int E, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < E) {
    int base = idx * 3;
    int src = deviceLocalEdges[base + 0];
    int dst = deviceLocalEdges[base + 1];
    int w   = deviceLocalEdges[base + 2];
    deviceDist[src * N + dst] = w;
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void phase1(int r, int *deviceDist, int N) {
  int i = threadIdx.y, j = threadIdx.x;
  int offset = r * BLOCK_SIZE;
  __shared__ int share[BLOCK_SIZE][BLOCK_SIZE];
  int i32 = i + 32;
  int j32 = j + 32;

  int tl = (i   + offset) * N + (j   + offset);
  int tr = (i   + offset) * N + (j32 + offset);
  int bl = (i32 + offset) * N + (j   + offset);
  int br = (i32 + offset) * N + (j32 + offset);

  share[i][j]     = deviceDist[tl];
  share[i][j32]   = deviceDist[tr];
  share[i32][j]   = deviceDist[bl];
  share[i32][j32] = deviceDist[br];

#pragma unroll 32
  for (int k = 0; k < BLOCK_SIZE; k++) {
    __syncthreads();
    int a0 = share[i   ][k], a1 = share[i32][k];
    int b0 = share[k][j   ], b1 = share[k][j32];

    int t00 = a0 + b0;
    int t10 = a1 + b0;
    int t01 = a0 + b1;
    int t11 = a1 + b1;

    if (t00 < share[i   ][j   ]) share[i   ][j   ] = t00;
    if (t10 < share[i32][j   ]) share[i32][j   ] = t10;
    if (t01 < share[i   ][j32 ]) share[i   ][j32 ] = t01;
    if (t11 < share[i32][j32 ]) share[i32][j32 ] = t11;
  }

  deviceDist[tl] = share[i][j];
  deviceDist[tr] = share[i][j32];
  deviceDist[bl] = share[i32][j];
  deviceDist[br] = share[i32][j32];
}

__global__ void phase2(int r, int *deviceDist, int N) {
  int i = threadIdx.y, j = threadIdx.x;
  __shared__ int pivot_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int ver_block  [BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int hor_block  [BLOCK_SIZE][BLOCK_SIZE];

  int offset = BLOCK_SIZE * r;
  int i32 = i + 32, j32 = j + 32;

  int ver_i_offset = (blockIdx.x < r) ? blockIdx.x * BLOCK_SIZE : (blockIdx.x + 1) * BLOCK_SIZE;
  int hor_j_offset = ver_i_offset;

  int p_tl = (i   + offset) * N + (j   + offset);
  int p_tr = (i   + offset) * N + (j32 + offset);
  int p_bl = (i32 + offset) * N + (j   + offset);
  int p_br = (i32 + offset) * N + (j32 + offset);

  int v_tl = (i   + ver_i_offset) * N + (j   + offset);
  int v_tr = (i   + ver_i_offset) * N + (j32 + offset);
  int v_bl = (i32 + ver_i_offset) * N + (j   + offset);
  int v_br = (i32 + ver_i_offset) * N + (j32 + offset);

  int h_tl = (i   + offset) * N + (j   + hor_j_offset);
  int h_tr = (i   + offset) * N + (j32 + hor_j_offset);
  int h_bl = (i32 + offset) * N + (j   + hor_j_offset);
  int h_br = (i32 + offset) * N + (j32 + hor_j_offset);

  pivot_block[i ][j ] = deviceDist[p_tl];
  pivot_block[i ][j32] = deviceDist[p_tr];
  pivot_block[i32][j ] = deviceDist[p_bl];
  pivot_block[i32][j32] = deviceDist[p_br];

  ver_block[i ][j ] = deviceDist[v_tl];
  ver_block[i ][j32] = deviceDist[v_tr];
  ver_block[i32][j ] = deviceDist[v_bl];
  ver_block[i32][j32] = deviceDist[v_br];

  hor_block[i ][j ] = deviceDist[h_tl];
  hor_block[i ][j32] = deviceDist[h_tr];
  hor_block[i32][j ] = deviceDist[h_bl];
  hor_block[i32][j32] = deviceDist[h_br];

  __syncthreads();

#pragma unroll 32
  for (int k = 0; k < BLOCK_SIZE; k++) {
    // vertical updates
    int pkj   = pivot_block[k][j];
    int pkj32 = pivot_block[k][j32];

    int vik   = ver_block[i   ][k];
    int vi32k = ver_block[i32][k];

    int t00 = vik   + pkj;
    int t10 = vi32k + pkj;
    int t01 = vik   + pkj32;
    int t11 = vi32k + pkj32;

    if (t00 < ver_block[i   ][j   ]) ver_block[i   ][j   ] = t00;
    if (t10 < ver_block[i32][j   ]) ver_block[i32][j   ] = t10;
    if (t01 < ver_block[i   ][j32 ]) ver_block[i   ][j32 ] = t01;
    if (t11 < ver_block[i32][j32 ]) ver_block[i32][j32 ] = t11;

    // horizontal updates
    int pik   = pivot_block[i   ][k];
    int pi32k = pivot_block[i32][k];

    int hkj   = hor_block[k][j];
    int hkj32 = hor_block[k][j32];

    int s00 = pik   + hkj;
    int s10 = pi32k + hkj;
    int s01 = pik   + hkj32;
    int s11 = pi32k + hkj32;

    if (s00 < hor_block[i   ][j   ]) hor_block[i   ][j   ] = s00;
    if (s10 < hor_block[i32][j   ]) hor_block[i32][j   ] = s10;
    if (s01 < hor_block[i   ][j32 ]) hor_block[i   ][j32 ] = s01;
    if (s11 < hor_block[i32][j32 ]) hor_block[i32][j32 ] = s11;
  }

  deviceDist[v_tl] = ver_block[i][j];
  deviceDist[v_tr] = ver_block[i][j32];
  deviceDist[v_bl] = ver_block[i32][j];
  deviceDist[v_br] = ver_block[i32][j32];

  deviceDist[h_tl] = hor_block[i][j];
  deviceDist[h_tr] = hor_block[i][j32];
  deviceDist[h_bl] = hor_block[i32][j];
  deviceDist[h_br] = hor_block[i32][j32];
}

__global__ void phase3(int r, int *deviceDist, int N) {
  int i = threadIdx.y;
  int j = threadIdx.x;
  int round = N / BLOCK_SIZE;

  int i_block_base = blockIdx.y * 2;
  int i_block = (i_block_base < r) ? i_block_base : (i_block_base + 1);
  int i_block_next = i_block + 1;
  if (i_block_next == r) i_block_next++;

  int i_offset       = i_block * BLOCK_SIZE;
  int i_offset_next  = i_block_next * BLOCK_SIZE;
  int offset         = r * BLOCK_SIZE;

  __shared__ int hor_block[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int ver_block1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int ver_block2[BLOCK_SIZE][BLOCK_SIZE];

  // preload two vertical blocks (col = r)
  for (int di = 0; di < 4; ++di) {
    int row = i + di * 16 + i_offset;
    ver_block1[i + di * 16][j     ] = deviceDist[row * N + (offset + j)];
    ver_block1[i + di * 16][j + 32] = deviceDist[row * N + (offset + j + 32)];
  }
  for (int di = 0; di < 4; ++di) {
    int row = i + di * 16 + i_offset_next;
    ver_block2[i + di * 16][j     ] = deviceDist[row * N + (offset + j)];
    ver_block2[i + di * 16][j + 32] = deviceDist[row * N + (offset + j + 32)];
  }

  int core_ij, core_ij32, core_i16j, core_i16j32, core_i32j, core_i32j32, core_i48j, core_i48j32;

  for (int j_block = 0; j_block < round; j_block++) {
    if (j_block == r) continue;
    int j_offset   = j_block * BLOCK_SIZE;
    int global_j   = j + j_offset;
    int global_j32 = j + 32 + j_offset;

    // load horizontal block (row = r)
    for (int di = 0; di < 4; ++di) {
      int row = offset + i + di * 16;
      hor_block[i + di * 16][j     ] = deviceDist[row * N + global_j];
      hor_block[i + di * 16][j + 32] = deviceDist[row * N + global_j32];
    }

    // first target row block
    {
      core_ij     = deviceDist[(i +       i_offset) * N + global_j];
      core_ij32   = deviceDist[(i +       i_offset) * N + global_j32];
      core_i16j   = deviceDist[(i + 16  + i_offset) * N + global_j];
      core_i16j32 = deviceDist[(i + 16  + i_offset) * N + global_j32];
      core_i32j   = deviceDist[(i + 32  + i_offset) * N + global_j];
      core_i32j32 = deviceDist[(i + 32  + i_offset) * N + global_j32];
      core_i48j   = deviceDist[(i + 48  + i_offset) * N + global_j];
      core_i48j32 = deviceDist[(i + 48  + i_offset) * N + global_j32];

      int old_core_ij     = core_ij;
      int old_core_ij32   = core_ij32;
      int old_core_i16j   = core_i16j;
      int old_core_i16j32 = core_i16j32;
      int old_core_i32j   = core_i32j;
      int old_core_i32j32 = core_i32j32;
      int old_core_i48j   = core_i48j;
      int old_core_i48j32 = core_i48j32;

      __syncthreads();
#pragma unroll 32
      for (int k = 0; k < BLOCK_SIZE; k++) {
        int hk    = hor_block[k][j];
        int hk32  = hor_block[k][j + 32];
        int v0k   = ver_block1[i      ][k];
        int v16k  = ver_block1[i + 16 ][k];
        int v32k  = ver_block1[i + 32 ][k];
        int v48k  = ver_block1[i + 48 ][k];

        core_ij     = min(core_ij,     v0k  + hk);
        core_ij32   = min(core_ij32,   v0k  + hk32);
        core_i16j   = min(core_i16j,   v16k + hk);
        core_i16j32 = min(core_i16j32, v16k + hk32);
        core_i32j   = min(core_i32j,   v32k + hk);
        core_i32j32 = min(core_i32j32, v32k + hk32);
        core_i48j   = min(core_i48j,   v48k + hk);
        core_i48j32 = min(core_i48j32, v48k + hk32);
      }

      if (core_ij     != old_core_ij)     deviceDist[(i +      i_offset) * N + global_j   ] = core_ij;
      if (core_ij32   != old_core_ij32)   deviceDist[(i +      i_offset) * N + global_j32 ] = core_ij32;
      if (core_i16j   != old_core_i16j)   deviceDist[(i + 16 + i_offset) * N + global_j   ] = core_i16j;
      if (core_i16j32 != old_core_i16j32) deviceDist[(i + 16 + i_offset) * N + global_j32 ] = core_i16j32;
      if (core_i32j   != old_core_i32j)   deviceDist[(i + 32 + i_offset) * N + global_j   ] = core_i32j;
      if (core_i32j32 != old_core_i32j32) deviceDist[(i + 32 + i_offset) * N + global_j32 ] = core_i32j32;
      if (core_i48j   != old_core_i48j)   deviceDist[(i + 48 + i_offset) * N + global_j   ] = core_i48j;
      if (core_i48j32 != old_core_i48j32) deviceDist[(i + 48 + i_offset) * N + global_j32 ] = core_i48j32;
    }

    // second target row block
    {
      core_ij     = deviceDist[(i +       i_offset_next) * N + global_j];
      core_ij32   = deviceDist[(i +       i_offset_next) * N + global_j32];
      core_i16j   = deviceDist[(i + 16  + i_offset_next) * N + global_j];
      core_i16j32 = deviceDist[(i + 16  + i_offset_next) * N + global_j32];
      core_i32j   = deviceDist[(i + 32  + i_offset_next) * N + global_j];
      core_i32j32 = deviceDist[(i + 32  + i_offset_next) * N + global_j32];
      core_i48j   = deviceDist[(i + 48  + i_offset_next) * N + global_j];
      core_i48j32 = deviceDist[(i + 48  + i_offset_next) * N + global_j32];

      int old_core_ij     = core_ij;
      int old_core_ij32   = core_ij32;
      int old_core_i16j   = core_i16j;
      int old_core_i16j32 = core_i16j32;
      int old_core_i32j   = core_i32j;
      int old_core_i32j32 = core_i32j32;
      int old_core_i48j   = core_i48j;
      int old_core_i48j32 = core_i48j32;

#pragma unroll 32
      for (int k = 0; k < BLOCK_SIZE; k++) {
        int hk    = hor_block[k][j];
        int hk32  = hor_block[k][j + 32];
        int v0k   = ver_block2[i      ][k];
        int v16k  = ver_block2[i + 16 ][k];
        int v32k  = ver_block2[i + 32 ][k];
        int v48k  = ver_block2[i + 48 ][k];

        core_ij     = min(core_ij,     v0k  + hk);
        core_ij32   = min(core_ij32,   v0k  + hk32);
        core_i16j   = min(core_i16j,   v16k + hk);
        core_i16j32 = min(core_i16j32, v16k + hk32);
        core_i32j   = min(core_i32j,   v32k + hk);
        core_i32j32 = min(core_i32j32, v32k + hk32);
        core_i48j   = min(core_i48j,   v48k + hk);
        core_i48j32 = min(core_i48j32, v48k + hk32);
      }

      if (core_ij     != old_core_ij)     deviceDist[(i +      i_offset_next) * N + global_j   ] = core_ij;
      if (core_ij32   != old_core_ij32)   deviceDist[(i +      i_offset_next) * N + global_j32 ] = core_ij32;
      if (core_i16j   != old_core_i16j)   deviceDist[(i + 16 + i_offset_next) * N + global_j   ] = core_i16j;
      if (core_i16j32 != old_core_i16j32) deviceDist[(i + 16 + i_offset_next) * N + global_j32 ] = core_i16j32;
      if (core_i32j   != old_core_i32j)   deviceDist[(i + 32 + i_offset_next) * N + global_j   ] = core_i32j;
      if (core_i32j32 != old_core_i32j32) deviceDist[(i + 32 + i_offset_next) * N + global_j32 ] = core_i32j32;
      if (core_i48j   != old_core_i48j)   deviceDist[(i + 48 + i_offset_next) * N + global_j   ] = core_i48j;
      if (core_i48j32 != old_core_i48j32) deviceDist[(i + 48 + i_offset_next) * N + global_j32 ] = core_i48j32;
    }
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input.txt>\n";
    return 1;
  }

  // ---- Read text input ----
  std::ifstream fin(argv[1]);
  if (!fin) {
    std::cerr << "Failed to open input file: " << argv[1] << "\n";
    return 1;
  }
  std::ios::sync_with_stdio(false);
  fin.tie(nullptr);

  if (!(fin >> V >> E)) {
    std::cerr << "Invalid header. Expecting: V E\n";
    return 1;
  }

  // Pad N to a multiple of BLOCK_SIZE; ensure (N/BLOCK_SIZE) is odd as in your logic
  N = (V % BLOCK_SIZE) ? (V + (BLOCK_SIZE - V % BLOCK_SIZE)) : V;
  if ((N / BLOCK_SIZE) % 2 == 0) N += BLOCK_SIZE;

  // Allocate and initialize dist
  HIP_CHECK(hipMalloc((void**)&deviceDist, (size_t)N * (size_t)N * sizeof(int)));
  {
    dim3 block(THREAD_SIZE, THREAD_SIZE);
    dim3 grid((N + THREAD_SIZE - 1) / THREAD_SIZE,
              (N + THREAD_SIZE - 1) / THREAD_SIZE);
    initializeDeviceDist<<<grid, block>>>(deviceDist, N);
    HIP_CHECK(hipGetLastError());
  }

  // Read edges to host
  std::vector<int> hostEdges((size_t)E * 3);
  for (int e = 0; e < E; ++e) {
    int u, v, w;
    if (!(fin >> u >> v >> w)) {
      std::cerr << "Invalid edge at line " << (e + 2) << " (expect: u v w)\n";
      return 1;
    }
    hostEdges[(size_t)3 * e + 0] = u;
    hostEdges[(size_t)3 * e + 1] = v;
    hostEdges[(size_t)3 * e + 2] = w;
  }
  fin.close();

  // Copy edges to device and apply
  int *deviceLocalEdges = nullptr;
  HIP_CHECK(hipMalloc((void**)&deviceLocalEdges, hostEdges.size() * sizeof(int)));
  HIP_CHECK(hipMemcpy(deviceLocalEdges, hostEdges.data(),
                      hostEdges.size() * sizeof(int), hipMemcpyHostToDevice));

  {
    int threadsPerBlock = 1024;
    int blocksPerGrid   = (E + threadsPerBlock - 1) / threadsPerBlock;
    updateDeviceDist<<<blocksPerGrid, threadsPerBlock>>>(deviceDist, deviceLocalEdges, E, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipFree(deviceLocalEdges));
  }

  // ---- Blocked Floyd–Warshall ----
  int round = N / BLOCK_SIZE;                  
  dim3 block(THREAD_SIZE, THREAD_SIZE);
  dim3 block3(1, (round - 1) / 2);

  for (int r = 0; r < round; r++) {
    phase1<<<1, block>>>(r, deviceDist, N);
    HIP_CHECK(hipGetLastError());

    if (round - 1 > 0) {
      phase2<<<round - 1, block>>>(r, deviceDist, N);
      HIP_CHECK(hipGetLastError());
    }
    if (round > 1) {
      dim3 threads_phase3(THREAD_SIZE, THREAD_SIZE / 2);
      phase3<<<block3, threads_phase3>>>(r, deviceDist, N);
      HIP_CHECK(hipGetLastError());
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  // ---- Copy back and print only the top-left VxV ----
  std::vector<int> hostDist((size_t)N * (size_t)N);
  HIP_CHECK(hipMemcpy(hostDist.data(), deviceDist, (size_t)N * (size_t)N * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(deviceDist));

  // stdout: V rows, each row V ints
  for (int i = 0; i < V; ++i) {
    int base = i * N;
    for (int j = 0; j < V; ++j) {
      if (j) std::cout << ' ';
      std::cout << hostDist[base + j];
    }
    std::cout << '\n';
  }

  return 0;
}
