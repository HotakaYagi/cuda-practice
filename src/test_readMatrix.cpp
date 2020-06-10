#include "readMatrix.h"
#include <iostream>

int main()
{
  sparseMatrix sp("nos5.mtx");

  std::cout << sp.m << std::endl;
  std::cout << sp.n << std::endl;
  std::cout << sp.nnz << std::endl;
  for (auto i = 0; i < sp.m * sp.n; i++)
  {
    if (sp.matrix[i] != 0)
    {
      printf("%16e\n", sp.matrix[i]);
    }
  }

  return 0;
}
