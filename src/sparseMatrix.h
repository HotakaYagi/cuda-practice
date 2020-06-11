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
