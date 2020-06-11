#include <iostream>
#include <memory>
#include <chrono>
#include <cstdlib> 
#include <vector>
#include <string>
#include "sparseMatrix.h"

template<typename T>
__device__ T warp_reduction(T val)
{
#define warpSize 32

    for (auto offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0, val, offset, warpSize);
    }
    return val;
}

template<typename T>
__global__ void spMulAdd_scalar(const int * __restrict__ row, const int * __restrict__ col, const T * __restrict__ val, const T * __restrict__ dx, T * __restrict__ dy, int n, int nnz)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
    T y_val = 0.0;
    if (tid < n)
    {
         #pragma unroll
         for (auto j = row[tid]; j < row[tid + 1]; ++j) 
         {
              y_val += val[j] * dx[col[j]];
         }
         dy[tid] = y_val;
    }
}

//kernel
template<typename T>
__global__ void spMulAdd_vector(const int * __restrict__ row, const int * __restrict__ col, const T * __restrict__ val, const T * __restrict__ dx, T * __restrict__ dy, int n, int nnz)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
    auto rowid = tid / warpSize;
    auto lane = tid % warpSize;
    T y_val = 0;
    
    if (rowid < n)
    {
         for (auto i = row[rowid] + lane; i < row[rowid + 1]; i += warpSize) 
         {
              y_val += val[i] * dx[col[i]];
         }
         y_val = warp_reduction<T>(y_val);
    }

    if (lane == 0 && rowid < n)
    { 
         dy[rowid] = y_val;
    }
}

int main(int args, char *argv[])
{
    // 読み込みたい行列は実行時引数で与える
    std::string fname;
    fname = argv[1];

    sparseMatrix sp(fname);

    const auto n = sp.n;
    const auto nnz = sp.nnz;

    // ベクトルxとベクトルyを作るところ
    std::unique_ptr<float[]> host_x(new float[n]);
    std::unique_ptr<float[]> host_y(new float[n]);

    for (auto i = 0; i < n; i++)
    {
        host_x[i] = static_cast<float>(rand()) / RAND_MAX;
        host_y[i] = 0;
    }

    // gpu に渡すところ
    int *row, *col; 
    float *val, *vec_x, *vec_y;

    cudaMalloc((void**)&row, (n + 1) * sizeof(int));
    cudaMalloc((void**)&col, nnz * sizeof(int));
    cudaMalloc((void**)&val, nnz * sizeof(float));
    cudaMalloc((void**)&vec_x, n * sizeof(float));
    cudaMalloc((void**)&vec_y, n * sizeof(float));

    cudaMemcpy(row, sp.row.get(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col, sp.col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(val, sp.val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_x, host_x.get(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_y, host_y.get(), n * sizeof(float), cudaMemcpyHostToDevice);

    // スレッドサイズはどう決めるのがよいのだろうか?
    const auto blocksize = 64;
    const dim3 block(blocksize, 1, 1);
    const dim3 grid(warpSize * std::ceil(n / static_cast<float>(block.x)), 1, 1);
    
    // 時間計測するところ、データ転送は含まなくてok?
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // 計算するところ
    spMulAdd_vector<float> <<<grid, block>>>(row, col, val, vec_x, vec_y, n, nnz);

    end = std::chrono::system_clock::now();

    // 結果があっているかcpuでも計算して確認するところ
    std::unique_ptr<float[]> result(new float[n]);
    cudaMemcpy(result.get(), vec_y, n * sizeof(n), cudaMemcpyDeviceToHost);

    std::unique_ptr<float[]> host_result(new float[n]);
    for (auto i = 0; i < n; i++)
    {
        host_result[i] = 0;
        for (auto j = sp.row[i]; j < sp.row[i + 1]; j++)
        {
           host_result[i] += sp.val[j] * host_x[sp.col[j]]; 
        }
    }

    auto checker = 0;
    for (auto i = 0; i < n; i++)
    {
        // float で誤差含めてだいたいこのくらい合ってれば正しい？
        auto m = 7 - std::log10(n);
        if (fabs(host_result[i] - result[i]) > std::pow(10, -2))
        {
            // 基準を満たさなかったら NG
            std::cout << "ng: " << fabs(host_result[i] - result[i]) << std::endl;
            checker++;
        }
    }
    
    if (checker == 0)
    {
        std::cout << "ok" << std::endl;
    }

    // 計算時間(データ転送含めない？)や次数、実効性能を出力
    const auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

    std::cout << "n: " << n << ", nnz: " << nnz << ", threads: " << blocksize << std::endl;
    std::cout << "time: " << time << " [ms]" << std::endl;
    std::cout << "perf: " << 2 * n * n / time / 1e6 << " [Gflops/sec]" << std::endl;

    cudaFree(row);
    cudaFree(col);
    cudaFree(val);
    cudaFree(vec_x);
    cudaFree(vec_y);

    return 0;
}

