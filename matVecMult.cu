#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#define BLOCK_SIZE 32

template<typename T>
__global__ void mulAdd(const T * __restrict__ dA, const T * __restrict__ dx, T * __restrict__ dy, int nRows, int nCols)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    T y_val = 0.0;

    if (tid < nRows)
    {
         #pragma unroll
         for (int m = 0; m < nCols; ++m)
         {
              y_val += dA[tid * nRows + m] * dx[m];
              __syncthreads();
         }
         dy[tid] = y_val;
         tid += blockIdx.x * blockDim.x;
    }
}

template<typename T>
__global__ void mulAdd_block(const T * __restrict__ dA, const T * __restrict__ dx, T * __restrict__ dy, int nRows, int nCols)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    __shared__ T x_shared[BLOCK_SIZE];
    T y_val = 0.0;

    #pragma unroll
    for (int m = 0; m < (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m)
    {
         if ((m * BLOCK_SIZE + threadIdx.x) < nCols)
         {
             x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
         }
         else
         {
             x_shared[threadIdx.x] = 0.f;
         }
         __syncthreads();

         #pragma unroll
         for (int e = 0; e < BLOCK_SIZE; ++e) {
             // --- Column-major ordering - faster
             //y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
             // --- Row-major ordering - slower
             y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
         }

         __syncthreads();
    }
    if (tid < nRows)
    {
         dy[tid] = y_val;
    }
}

int main(int args, char *argv[])
{
    int n;
    n = atoi(argv[1]);
    float *mat_a, *vec_x, *vec_y;

    std::unique_ptr<float[]> host_a(new float[n * n]);
    std::unique_ptr<float[]> host_x(new float[n]);
    std::unique_ptr<float[]> host_y(new float[n]);

    cudaMalloc((void**)&mat_a, n * n * sizeof(float));
    cudaMalloc((void**)&vec_x, n * sizeof(float));
    cudaMalloc((void**)&vec_y, n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        std::random_device rand{};
        host_a[i] = rand() / 1e9;
        host_x[i] = rand() / 1e9;
        host_y[i] = 0;
    }
    for (int i = n; i < n * n; i++)
    {
        std::random_device rand{};
        host_a[i] = rand() / 1e9;
    }

    cudaMemcpy(mat_a, host_a.get(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_x, host_x.get(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_y, host_y.get(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 32;
    dim3 block (blocksize, 1, 1);
    dim3 grid  ((n + blocksize + 1) / block.x, 1, 1);
    
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    mulAdd_block<float> <<<grid, block>>>(mat_a, vec_x, vec_y, n, n);

    end = std::chrono::system_clock::now();

    std::unique_ptr<float[]> result(new float[n]);
    cudaMemcpy(result.get(), vec_y, n * sizeof(n), cudaMemcpyDeviceToHost);

    std::unique_ptr<float[]> host_result(new float[n]);
    for (int i = 0; i < n; i++)
    {
        host_result[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
           host_result[i] += host_a[i * n + j] * host_x[j]; 
        }
    }

    int checker = 0;
    for (int i = 0; i < n; i++)
    {
        auto m = 8 - std::log10(n);
        if (fabs(host_result[i] - result[i]) > std::pow(10, -m))
        {
            std::cout << "ng: " << host_result[i] - result[i] << std::endl;
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

    double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

    std::cout << "n: " << n << " threads: " << blocksize << std::endl;
    std::cout << "time: " << time << " [ms]" << std::endl;
    std::cout << "perf: " << 2 * n * n / time / 1e6 << " [Gflops/sec]" << std::endl;

    cudaFree(mat_a);
    cudaFree(vec_x);
    cudaFree(vec_y);
    return 0;
}

