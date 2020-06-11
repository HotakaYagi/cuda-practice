#include <iostream>
#include <memory>
#include <chrono>
#include <cstdlib> 
#include <vector>

//kernel
template<typename T>
__global__ void spMulAdd(const int * __restrict__ row, const int * __restrict__ col, const T * __restrict__ val, const T * __restrict__ dx, T * __restrict__ dy, int n, int nnz)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
    T y_val = 0.0;
    
    // Ax=yにおいて、ベクトルyの成分を各スレッドが計算するように並列化
    if (tid < n)
    {
         // C++ は列優先だから、各スレッドは行列Aの各列のデータが読めれば良い
         #pragma unroll
         for (auto j = row[tid]; j < row[tid + 1]; ++j) 
         {
              y_val += val[j] * dx[col[j]];
         }
         dy[tid] = y_val;
         // スレッド番号がnになるまで(yの全要素計算するまで)インクリメント
         tid += blockIdx.x * blockDim.x;
    }
}

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
    // n は実行時引数で与える
    int n;
    n = atoi(argv[1]);

    // 疎行列を作るところ
    int *row, *col; 
    double *val, *vec_x, *vec_y;

    std::unique_ptr<double[]> host_a(new double[n * n]);

    for (auto i = 0; i < n * n; i++)
    {
        if (static_cast<double>(std::rand()) / RAND_MAX < 0.5)
        {
             //host_a[i] = static_cast<double>(std::rand()) / RAND_MAX;
             host_a[i] = 1;
        }
        else
        {
             host_a[i] = 0;
        }
    }
    std::unique_ptr<int[]> host_row(new int[n + 1]);
    std::vector<int> host_col;
    std::vector<double> host_val;

    auto nnz = 0;
    host_row[0] = nnz;
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            if (host_a[i * n + j] != 0)
            {
                host_val.push_back(host_a[i * n + j]);
                host_col.push_back(j);
                nnz++;
            }
        }
        host_row[i + 1] = nnz;
    }

    // ベクトルxとベクトルyを作るところ
    std::unique_ptr<double[]> host_x(new double[n]);
    std::unique_ptr<double[]> host_y(new double[n]);

    for (auto i = 0; i < n; i++)
    {
        //host_x[i] = static_cast<double>(rand()) / RAND_MAX;
        host_x[i] = 1;
        host_y[i] = 0;
    }

    // gpu に渡すところ
    cudaMalloc((void**)&row, (n + 1) * sizeof(int));
    cudaMalloc((void**)&col, nnz * sizeof(int));
    cudaMalloc((void**)&val, nnz * sizeof(double));
    cudaMalloc((void**)&vec_x, n * sizeof(double));
    cudaMalloc((void**)&vec_y, n * sizeof(double));

    cudaMemcpy(row, host_row.get(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    auto* p_host_col = host_col.data();
    cudaMemcpy(col, p_host_col, nnz * sizeof(int), cudaMemcpyHostToDevice);
    auto* p_host_val = host_val.data();
    cudaMemcpy(val, p_host_val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_x, host_x.get(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_y, host_y.get(), n * sizeof(double), cudaMemcpyHostToDevice);

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

    std::unique_ptr<double[]> host_result(new double[n]);
    for (auto i = 0; i < n; i++)
    {
        host_result[i] = 0;
    }

    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
           host_result[i] += host_a[i * n + j] * host_x[j]; 
        }
    }

    auto checker = 0;
    for (auto i = 0; i < n; i++)
    {
        // double で誤差含めてだいたいこのくらい合ってれば正しい？
        auto m = 7 - std::log10(n);
        if (fabs(host_result[i] - result[i]) > std::pow(10, -m))
        {
            // 基準を満たさなかったら NG
            std::cout << "ng: " << result[i] << std::endl;
            checker++;
        }
    }
    
    if (checker == 0)
    {
        std::cout << "ok" << std::endl;
    }
    else
    {
        std::cout << checker << std::endl;
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

