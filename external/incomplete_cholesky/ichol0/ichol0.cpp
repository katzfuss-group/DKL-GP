#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <math.h> 
#include <string>
#include <stdexcept>


namespace py = pybind11;


#define ALMOSTZERO 1E-10


float dot_prod(int l1, int u1, int l2, int u2, const std::vector<int> &row_inds, 
    const std::vector<float> &vals){

  float result = 0.0;
  while(l1 < u1 && l2 < u2){
    if(row_inds[l1] == row_inds[l2]) {
      result += vals[l1] * vals[l2];
      l1++; l2++;
    }
    else if(row_inds[l1] < row_inds[l2])
      l1++;
    else
      l2++;
  }
  return result;
}


//' Incomplete Cholesky decomposition of a sparse matrix passed in
//' the compressed sparse row format
//'
//' @param crow_inds, col_inds, vals: the CSR-format storage of the lower-triangular 
//     part of the covariance matrix
//  @param nrow: number of rows
//' @return the Cholesky factor of the same sparsity pattern stored in vals
std::vector<float> ichol0(std::vector<int> crow_inds, std::vector<int> col_inds,
    std::vector<float> vals, int nrow){
    for(int i = 0; i < nrow; ++i) {
        if(col_inds[crow_inds[i + 1] - 1] != i) {
            throw std::invalid_argument("Diagonal " + std::to_string(i) + 
                "should be a non-zero entry\n");
        }
        for(int u1 = crow_inds[i]; u1 < crow_inds[i + 1]; ++u1){
            int j = col_inds[u1];
            if(j > i) {
                throw std::invalid_argument("The input sparsity is not "
                    "lower-triangular\n");
            }
            int l1 = crow_inds[i];
            int l2 = crow_inds[j];
            int u2 = crow_inds[j + 1] - 1;
            float dp = dot_prod(l1, u1, l2, u2, col_inds, vals);
            if(j < i) {
                vals[u1] = (vals[u1] - dp) / vals[crow_inds[j + 1] - 1];
            }
            else if(j == i) {
                if(vals[u1] - dp < ALMOSTZERO) {
                    throw std::runtime_error("The " + std::to_string(i) + 
                        "-th diagonal coefficient is smaller than " + 
                        std::to_string(ALMOSTZERO) + "\n");
                }
                vals[u1] = sqrt(vals[u1] - dp);
            }
        }
    }
  return vals;
}


PYBIND11_MODULE(ichol0, m) {
    m.doc() = "Imcomplete Cholesky factorization";
    m.def("ichol0", &ichol0);
}









