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
  std::unique_ptr<double[]> matrix;

  sparseMatrix(std::string fname);
   
  //TODO: 行列ベクトル積とか基本演算実装しとく？
};

sparseMatrix::sparseMatrix(std::string fname)
{
  // mtx file を読む
  std::ifstream fin(fname);

  while (fin.peek() == '%') fin.ignore(2048, '\n');

  fin >> m >> n >> nnz;

  matrix.reset(new double[m * n]);

  // dense matrix を作成しておく
  for (auto l = 0; l < nnz; l++)
  {
    int ind_m, ind_n;
    double data;
    fin >> ind_m >> ind_n >> data;
    matrix[(ind_m-1) + (ind_n-1) * m] = data;
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
