#include <iostream>
#include <memory>
#include <chrono>
#include <cstdlib> 
#include <vector>
#include <string>
#include "sparseMatrix.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/device_vector.h>
#include <iomanip>

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

template<typename T>
__global__ void spMulAdd_vector(const int * __restrict__ row, const int * __restrict__ col, const T * __restrict__ val, const T * __restrict__ dx, T * __restrict__ dy, int n, int nnz)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x; 
    auto rowid = tid / warpSize;
    auto lane = tid % warpSize;
    T y_val = 0;
    
    if (rowid < n)
    {
         #pragma unroll
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
    std::unique_ptr<double[]> host_x(new double[n]);
    std::unique_ptr<double[]> host_y(new double[n]);

    for (auto i = 0; i < n; i++)
    {
        //host_x[i] = static_cast<float>(rand()) / RAND_MAX;
        host_x[i] = 1;
        host_y[i] = 0;
    }

    // gpu用ので配列を生成
    thrust::device_vector<int> row(n + 1);
    thrust::device_vector<int> col(nnz);
    thrust::device_vector<double> val(nnz);
    thrust::device_vector<double> vec_x(n);
    thrust::device_vector<double> vec_y(n);

    thrust::copy_n(sp.row.begin(), n + 1, row.begin());
    thrust::copy_n(sp.col.begin(), nnz, col.begin());
    thrust::copy_n(sp.val.begin(), nnz, val.begin());
    thrust::copy_n(host_x.get(), n, vec_x.begin());
    thrust::copy_n(host_y.get(), n, vec_y.begin());

    int* rowPtr = thrust::raw_pointer_cast(&(row[0]));
    int* colPtr = thrust::raw_pointer_cast(&(col[0]));
    double* valPtr = thrust::raw_pointer_cast(&(val[0]));
    double* vec_xPtr = thrust::raw_pointer_cast(&(vec_x[0]));
    double* vec_yPtr = thrust::raw_pointer_cast(&(vec_y[0]));

    // スレッドサイズはどう決めるのがよいのだろうか?
    int thread_size = atoi(argv[2]);
    const auto blocksize = thread_size;
    const dim3 block(blocksize, 1, 1);
    const dim3 grid(std::ceil(n / static_cast<double>(block.x)), 1, 1);
    
    // 時間計測するところ
    const auto num_iter = 30;
    std::vector<double> time_stamp;

    for (auto i = 0; i < num_iter; i++)
    {
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();

        // 計算するところ
        spMulAdd_vector<double> <<<grid, block>>>(rowPtr, colPtr, valPtr, vec_xPtr, vec_yPtr, n, nnz);

        end = std::chrono::system_clock::now();

        time_stamp.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())/1000/1000);
    }

    // 結果があっているかcpuでも計算して確認するところ
    std::unique_ptr<double[]> result(new double[n]);
    thrust::copy_n(vec_y.begin(), n, result.get());

    std::unique_ptr<double[]> host_result(new double[n]);
    for (auto i = 0; i < n; i++)
    {
        host_result[i] = 0;
        for (auto j = sp.row[i]; j < sp.row[i + 1]; j++)
        {
           host_result[i] += sp.val[j] * host_x[sp.col[j]]; 
        }
    }

    auto residual = 0;
    auto y_norm = 0;
    for (auto i = 0; i < n; i++)
    {
        residual += std::pow(host_result[i] - result[i], 2);
        y_norm += std::pow(result[i], 2);
    }
    
    residual = std::sqrt(residual);
    y_norm = std::sqrt(y_norm);

/*
    // float で誤差含めてだいたいこのくらい合ってれば正しい？
    const auto m = 14 - std::log10(n);
    if (residual / y_norm < m)
    {
        std::cout << "ok" << std::endl;
    }
    else
    {
        std::cout << "ng" << std::endl;
    }
*/
    // cuSPARSE
    
    ::cusparseHandle_t cusparse;
    ::cusparseCreate(&cusparse);

    ::cusparseMatDescr_t matDescr;
    ::cusparseCreateMatDescr(&matDescr);
    ::cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
    ::cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);

    thrust::device_vector<double> result_cu(n);
    thrust::copy_n(host_y.get(), n, result_cu.begin());
    double* result_cuPtr = thrust::raw_pointer_cast(&(result_cu[0]));
    const double ALPHA = 1;
    const double BETA = 0;

    std::vector<double> time_stamp_cublas;
    for (auto i = 0; i < num_iter; i++)
    {
        std::chrono::system_clock::time_point start_cublas, end_cublas;
        start_cublas = std::chrono::system_clock::now();

        ::cusparseDcsrmv(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            n, n, nnz,
            &ALPHA, matDescr, valPtr, rowPtr, colPtr,
            vec_xPtr,
            &BETA, result_cuPtr);

        end_cublas = std::chrono::system_clock::now();
        time_stamp_cublas.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_cublas - start_cublas).count())/1000/1000);
     }
    std::unique_ptr<double[]> result_cu_host(new double[n]);
    thrust::copy_n(result_cu.begin(), n, result_cu_host.get());
    
    std::vector<double> time_stamp_scl;
    for (auto i = 0; i < num_iter; i++)
    {
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();

        // 計算するところ
        spMulAdd_scalar<double> <<<grid, block>>>(rowPtr, colPtr, valPtr, vec_xPtr, vec_yPtr, n, nnz);

        end = std::chrono::system_clock::now();

        time_stamp_scl.push_back(static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())/1000/1000);
    }
 
    // 計算時間や次数、実効性能を出力
    const auto median_it = time_stamp.begin() + time_stamp.size() / 2;
    std::nth_element(time_stamp.begin(), median_it, time_stamp.end());
    const auto time = *median_it;
    const auto median_it_scl = time_stamp_scl.begin() + time_stamp_scl.size() / 2;
    std::nth_element(time_stamp_scl.begin(), median_it_scl, time_stamp_scl.end());
    const auto time_scl = *median_it_scl;
    const auto median_it_cublas = time_stamp_cublas.begin() + time_stamp_cublas.size() / 2;
    std::nth_element(time_stamp_cublas.begin(), median_it_cublas, time_stamp_cublas.end());
    const auto time_cublas = *median_it_cublas;

    const auto flops = 2 * nnz;
    const auto bytes = (n + 1) * sizeof(int) + nnz * sizeof(double) + nnz * sizeof(int) + 3 * n * sizeof(double);
    //std::cout << "matrix: " << fname << std::endl;
    //std::cout << "n: " << n << ", nnz: " << nnz << ", threads: " << blocksize << std::endl;
    //std::cout << "time: " << time << " [msec]" << std::endl;
    //std::cout << "time(cublas): " << time_cublas << " [msec]" << std::endl;
    //std::cout << "perf: " << flops / time / 1e6 << " [Gflops/sec]" << std::endl;
    //std::cout << "perf(cublas): " << flops / time_cublas / 1e6 << " [Gflops/sec]" << std::endl;
    //std::cout << "perf: " << bytes / time / 1e6 << " [Gbytes/sec]" << std::endl;
    //std::cout << "perf(cublas): " << bytes / time_cublas / 1e6 << " [Gbytes/sec]" << std::endl;
    //std::cout << "residual norm 2: " << residual / y_norm << std::endl;

    std::cout << fname << "," << std::fixed << std::setprecision(15) << time_scl << "," << time << "," << time_cublas << "," << flops / time_scl / 1e9 << "," << flops / time / 1e9 << "," << flops / time_cublas / 1e9 << "," << bytes / time_scl / 1e9 << "," << bytes / time / 1e9 << "," << bytes / time_cublas / 1e9 << std::endl; 
    return 0;
}

