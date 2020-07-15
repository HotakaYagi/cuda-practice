#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cstdlib>
#include <memory>

class sparseMatrix
{
public:
  int m, n, nnz;
  //std::unique_ptr<int[]> row;
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double> val;

  sparseMatrix(std::string fname);
  //void residual(const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ answer);
   
  //TODO: 行列ベクトル積とか基本演算実装しとく？
};
  
sparseMatrix::sparseMatrix(std::string fname)
{
    // TODO: 長方形の行列や複素数もパースできると良い
    // mtx file を読む
  if (fname.find("mtx") != std::string::npos)
  {
    std::ifstream fin(fname);
    
    while (fin.peek() == '%') fin.ignore(2048, '\n');
    fin >> m >> n >> nnz;
    if (m != n)
    {
        std::cout << "EROOR!" << std::endl;
    }
  
    // sparse (csr) を作るところ
    //row.reset(new int[n + 1]);
    auto current_n = 1;
    row.push_back(0);
    for (auto i = decltype(nnz)(0); i < nnz; i++)
    {
      int ind_m, ind_n;
      double data;
      fin >> ind_m >> ind_n >> data;

      col.push_back(ind_m - 1);
      val.push_back(data);
      if (ind_n == current_n + 1)
      {
          row.push_back(i);
          current_n++;  
      }
    }
    row[n] = nnz;
    fin.close();
  }
  else
  {
    n = 20000;
    m = 20000;
    nnz = m * n;

    std::unique_ptr<double[]> matrix(new double[m * n]);
    for (auto i = 0; i < m * n; i++)
    { 
      matrix[i] = 1.0;
    }

    auto nnz_count = 0;
    row.push_back(0);
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
      row.push_back(nnz_count);
    }	      
    row[n] = nnz;
  }
}

/*
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
*/
