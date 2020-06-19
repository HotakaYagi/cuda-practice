#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

class sparseMatrix
{
public:
  int m, n, nnz;
  std::unique_ptr<int[]> row;
  std::vector<int> col;
  std::vector<float> val;
  std::unique_ptr<float[]> matrix;

  sparseMatrix(std::string fname);
  void residual(const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ answer);
  void spMulAdd_vector(const T * __restrict__ dx, T * __restrict__ dy);
  void spMulAdd_scalar(const T * __restrict__ dx, T * __restrict__ dy);
  //TODO: 行列ベクトル積とか基本演算実装しとく？
  };
  
  sparseMatrix::sparseMatrix(std::string fname)
  {
    // TODO: 長方形の行列や複素数もパースできると良い
    // mtx file を読む
    std::ifstream fin(fname);
    
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    fin >> m >> n >> nnz;
  
    matrix.reset(new float[m * n]);
    for (auto i = 0; i < m*n; i++)
    {
        matrix[i] = 0;
    }
    // dense matrix を作成しておく
    for (auto i = 0; i < nnz; i++)
    {
      int ind_m, ind_n;
      double data;
      fin >> ind_m >> ind_n >> data;
      matrix[(ind_m-1) + (ind_n-1) * m] = static_cast<float>(data);
    }
    fin.close();

    // sparse (csr) を作るところ
    row.reset(new int[n + 1]);
    auto nnz_count = 0;
    row[0] = nnz_count;
    for (auto i = 0; i < n; i++)
    {
      for (auto j = 0; j < n; j++)
      {
         if (matrix[i * n + j] != 0)
         {
            val.push_back(matrix[i * n + j]);
            col.push_back(j);
            nnz_count++;
         }
      } 
      row[i + 1] = nnz_count;
      }
  }
}

public:
{
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
}

void sparseMatrix::residual(const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ answer)
{
  for (auto i = 0; i < n; i++)
  {
    auto y_val = 0;
    for (auto j = row[i]; j < row[i+1]; j++)
    {
      y_val += val[j] * x[col[j]];
    }
    answer[i] = y_val - y[i];
  }
}

template<typename T>
__global__ void sparseMatrix::spMulAdd_scalar(const T * __restrict__ dx, T * __restrict__ dy)
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
__global__ void sparseMatrix::spMulAdd_vector(const T * __restrict__ dx, T * __restrict__ dy)
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

