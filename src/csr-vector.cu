#include <iostream>
#include <memory>
#include <chrono>
#include <cstdlib> 
#include <vector>
#include <string>
#include "readMatrix.h"

template<typename T>
__device__ T warp_reduction(T val)
{
#define warpSize 32

    for (auto offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
    }
    return val;
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
    fname = atoi(argv[1]);

    sparseMatrix sp(fname);

    // ベクトルxとベクトルyを作るところ
    std::unique_ptr<double[]> host_x(new double[sp.n]);
    std::unique_ptr<double[]> host_y(new double[sp.n]);

    for (auto i = 0; i < n; i++)
    {
        host_x[i] = static_cast<double>(rand()) / RAND_MAX;
        host_y[i] = 0;
    }

    // gpu に渡すところ
    int *row, *col; 
    double *val, *vec_x, *vec_y;

    cudaMalloc((void**)&row, (sp.n + 1) * sizeof(int));
    cudaMalloc((void**)&col, sp.nnz * sizeof(int));
    cudaMalloc((void**)&val, sp.nnz * sizeof(double));
    cudaMalloc((void**)&vec_x, sp.n * sizeof(double));
    cudaMalloc((void**)&vec_y, sp.n * sizeof(double));

    cudaMemcpy(row, sp.row.get(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col, sp.col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(val, sp.val.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_x, host_x.get(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_y, host_y.get(), n * sizeof(float), cudaMemcpyHostToDevice);

    // スレッドサイズはどう決めるのがよいのだろうか?
    auto blocksize = 32;
    dim3 block (blocksize, 1, 1);
    dim3 grid  (warpSize * std::ceil(n / static_cast<double>(block.x)), 1, 1);
    
    // 時間計測するところ、データ転送は含まなくてok?
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    // 計算するところ
    spMulAdd_vector<double> <<<grid, block>>>(row, col, val, vec_x, vec_y, n, nnz);

    end = std::chrono::system_clock::now();

    // 結果があっているかcpuでも計算して確認するところ
    std::unique_ptr<double[]> result(new double[n]);
    cudaMemcpy(result.get(), vec_y, n * sizeof(n), cudaMemcpyDeviceToHost);
    std::unique_ptr<double[]> residual(new double[n]);

    sp.residual(host_x.get(), result.get(), residual.get());

    auto checker = 0;
    for (auto i = 0; i < n; i++)
    {
        // float で誤差含めてだいたいこのくらい合ってれば正しい？
        auto m = 7 - std::log10(n);
        if (fabs(residual[i]) > std::pow(10, -m))
        {
            // 基準を満たさなかったら NG
            std::cout << "ng: " << std::endl;
            checker++;
            break;
        }
    }
    
    if (checker == 0)
    {
        std::cout << "ok" << std::endl;
    }

    // 計算時間(データ転送含めない？)や次数、実効性能を出力
    auto time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

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

