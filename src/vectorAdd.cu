#include <iostream>
#include <memory>
#include <chrono>
#include <random>

__global__ void add(float* vec_a, float* vec_b, float* vec_c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        vec_c[i] = vec_a[i] + vec_b[i];
        i += blockDim.x * gridDim.x;
    }
}

int main(int args, char *argv[])
{
    int n;
    n = atoi(argv[1]);
    float *vec_a, *vec_b, *vec_c;

    std::unique_ptr<float[]> host_a(new float[n]);
    std::unique_ptr<float[]> host_b(new float[n]);
    std::unique_ptr<float[]> host_c(new float[n]);

    cudaMalloc((void**)&vec_a, n * sizeof(float));
    cudaMalloc((void**)&vec_b, n * sizeof(float));
    cudaMalloc((void**)&vec_c, n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        std::random_device rand{};
        host_a[i] = rand();
        host_b[i] = rand();
        host_c[i] = 0;
    }

    cudaMemcpy(vec_a, host_a.get(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_b, host_b.get(), n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec_c, host_c.get(), n*sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 128;
    dim3 block (blocksize, 1, 1);
    dim3 grid  ((n + blocksize + 1) / block.x, 1, 1);
    
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    add<<<grid, block>>>(vec_a, vec_b, vec_c, n);

    end = std::chrono::system_clock::now();

    std::unique_ptr<float[]> host_result(new float[n]);

    cudaMemcpy(host_result.get(), vec_c, n * sizeof(n), cudaMemcpyDeviceToHost);

    int checker = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabs(host_result[i] - (host_a[i] + host_b[i])) > 10e-8)
        {
            std::cout << "ng: " << host_result[i] << std::endl;
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
    std::cout << "perf: " << n / time / 1e6 << " [Gflops/sec]" << std::endl;

    cudaFree(vec_a);
    cudaFree(vec_b);
    cudaFree(vec_c);
    return 0;
}

