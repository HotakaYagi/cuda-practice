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
  int m, n, nnz, row_block_num_;
  //std::unique_ptr<int[]> row;
  std::vector<int> row;
  std::vector<int> row_blocks;
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
    
    auto num_entries_in_current_batch = 0;
    const auto shared_mem_size = 1024; // number of column indices loaded to shared memory, number of floating point values loaded to shared memory

    row_block_num_ = 0;
    row_blocks.push_back(0);
    for (auto i = 0; i < n; ++i)
    {
      auto entries_in_row = row[i + 1] - row[i];
      num_entries_in_current_batch += entries_in_row;

      if (num_entries_in_current_batch > shared_mem_size)
      {
        auto rows_in_batch = i - row_blocks[row_block_num_];
        if (rows_in_batch > 0) // at least one full row is in the batch. Use current row in next batch.
        {
          //row_blocks.set(++row_block_num_, i--);
          row_blocks.push_back(i--);
        }
        else // row is larger than buffer in shared memory
          row_blocks.push_back(i + 1);
          //row_blocks.set(++row_block_num_, i+1);
        ++row_block_num_;
        num_entries_in_current_batch = 0;
      }
    }
    if (num_entries_in_current_batch > 0)
    {
      row_blocks.push_back(n);
      ++row_block_num_;
      //row_blocks.set(++row_block_num_, rows_);
    }
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
