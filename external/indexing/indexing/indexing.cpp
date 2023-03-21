#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <stdexcept>
 #ifdef _OPENMP
 # include <omp.h>
 #endif

namespace py = pybind11;


template <class ForwardIterator, class T>
  int my_binary_search (ForwardIterator first, ForwardIterator last, const T& val)
{
    auto first_cp = first;
    first = std::lower_bound(first,last,val);
    if (first!=last && !(val<*first))
        return first - first_cp;
    else
        return last - first_cp;
}


struct Ancestor {
    Ancestor(int n, const std::vector<int> &spCidxCSR, 
        const std::vector<int> &spIdxCSR, const std::vector<int> &ancCidx,
        const std::vector<int> &ancIdx);
    std::tuple<std::vector<int>, std::vector<int>> query(
        const std::vector<int> &batIdx);
    std::tuple<std::vector<int>, std::vector<int>> query_row(
        const std::vector<int> &batIdx);
    int find_max_anc_size(const std::vector<int> &batIdx);

    int n;
    std::vector<int> sp_cidx_csr;
    std::vector<int> sp_idx_csr;
    std::vector<int> anc_cidx;
    std::vector<int> anc_idx;
    std::map<int, int> ds_idx_to_mem_idx;
};

Ancestor::Ancestor(int nOther, const std::vector<int> &spCidxCSR, 
    const std::vector<int> &spIdxCSR, const std::vector<int> &ancCidx,
    const std::vector<int> &ancIdx){
    /*
    Caution:
      * ancCidx must be in ascending order  
      * anc_idx[ancCidx[i]:ancCidx[i + 1]] must be in ascending order
    */
    n = nOther;
    if((int) spCidxCSR.size() != n + 1)
        throw std::invalid_argument("Mismatch between nOther and spCidxCSR\n");
    int idx_mem = 0;  // index in the storage memory
    for(int i = 0; i < n; i++)  // row index
        for(int j = spCidxCSR[i]; j < spCidxCSR[i + 1]; j++){
            int idx_col = spIdxCSR[j];  // col index
            ds_idx_to_mem_idx.insert({i * n + idx_col, idx_mem});
            idx_mem++;
        }
    sp_cidx_csr = spCidxCSR;
    sp_idx_csr = spIdxCSR;
    anc_cidx = ancCidx;
    anc_idx = ancIdx; 
}

int Ancestor::find_max_anc_size(const std::vector<int> &batIdx){
    int bat_size = batIdx.size();
    std::vector<int> bat_anc_size(bat_size);
    for(int i = 0; i < bat_size; i++){
        int idx = batIdx[i];
        bat_anc_size[i] = anc_cidx[idx + 1] - anc_cidx[idx];
    }
    return *std::max_element(bat_anc_size.begin(), bat_anc_size.end());
}

std::tuple<std::vector<int>, std::vector<int>> Ancestor::query(
        const std::vector<int> &batIdx){
    /*
    Input:
      * batIdx: mini-batch indices
    Return (idxs_flat_bat_mat_sub, idxs_mem), where
      * idxs_flat_bat_mat_sub: is the 1D index of the non-zero coefficients in the 
      batch-matrix of shape `(bat_size, max_anc_size, max_anc_size)`. Matrices 
      smaller than `max_anc_size` are stored in the bottom-right corners.
      * idxs_mem: has the same length as `idxs_flat_bat_mat_sub`, corresponding to
      the non-zero entries in memory. The non-zeros entries should be stored in 
      the CSR order.
    */
    int bat_size = batIdx.size();  // batch size
    int max_anc_size = find_max_anc_size(batIdx);  // max ancestor size for the batch  
    // Each thread writes on a different vector<int> to avoid conflict 
    std::vector<std::vector<int>> idxs_flat_bat_mat_sub_2d(bat_size);
    std::vector<std::vector<int>> idxs_mem_2d(bat_size);
    #pragma omp parallel for
    for(int k = 0; k < bat_size; k++){
        int idx = batIdx[k]; 
        const auto anc_bgn = anc_idx.begin() + anc_cidx[idx];
        const auto anc_end = anc_idx.begin() + anc_cidx[idx + 1];
        int anc_size = anc_end - anc_bgn;
        // num of rows/cols for padding
        int offset_mat_sub = max_anc_size - anc_size;  
        // offset for previous k sub-matrices and the first `offset_mat_sub` 
        // rows/cols
        int offset_idx_flat_k = k * max_anc_size * max_anc_size + 
            offset_mat_sub * max_anc_size + offset_mat_sub;
        for(int i = 0; i < anc_size; i++){
            /*
            `i` or `row_idx_mat_sub` is the row index in the 
            `anc_size` X `anc_size` sub-matrix.
            `col_idx_mat_sub` is the row index in the `anc_size` X `anc_size` 
            sub-matrix.
            `row_idx_mat` and `col_idx_mat` are row/col indices in the 
            n X n matrix.
            */
            int row_idx_mat = anc_bgn[i];
            int row_idx_mat_sub = i; 
            for(int j = sp_cidx_csr[row_idx_mat]; 
                j < sp_cidx_csr[row_idx_mat + 1]; j++){
                /*
                The `j`-th non-zero coefficient in the sparse n X n matrix 
                in the CSR order.
                */
                int col_idx_mat = sp_idx_csr[j];  
                int col_idx_mat_sub = my_binary_search(anc_bgn, anc_end, 
                    col_idx_mat);
                if(col_idx_mat_sub < anc_size){
                    int idx_flat_bat_mat_sub = offset_idx_flat_k + 
                        row_idx_mat_sub * max_anc_size + col_idx_mat_sub;
                    idxs_flat_bat_mat_sub_2d[k].push_back(idx_flat_bat_mat_sub);
                    idxs_mem_2d[k].push_back(j);
                }
            }
        }
    }
    // cumulative length of each thread's rslt
    std::vector<int> cidx_mem(bat_size + 1, 0);  
    for(int k = 0; k < bat_size; k++)
        cidx_mem[k + 1] = cidx_mem[k] + idxs_mem_2d[k].size();
    // flatten idxs_flat_bat_mat_sub_2d and idxs_mem_2d
    std::vector<int> idxs_flat_bat_mat_sub(cidx_mem[bat_size]);
    std::vector<int> idxs_mem(cidx_mem[bat_size]);
    for(int k = 0; k < bat_size; k++){
        std::copy(idxs_flat_bat_mat_sub_2d[k].begin(), 
            idxs_flat_bat_mat_sub_2d[k].end(), 
            idxs_flat_bat_mat_sub.begin() + cidx_mem[k]);
        std::copy(idxs_mem_2d[k].begin(), idxs_mem_2d[k].end(), 
            idxs_mem.begin() + cidx_mem[k]);
    }

    return std::make_tuple(idxs_flat_bat_mat_sub, idxs_mem);
}

std::tuple<std::vector<int>, std::vector<int>> Ancestor::query_row(
        const std::vector<int> &batIdx){
    /*
    Input:
      * batIdx: mini-batch indices
    Return (idxs_flat_bat_row_sub, idxs_mem), where
      * idxs_flat_bat_row_sub: is the 1D index of the non-zero coefficients in the 
      batch-row of shape `(bat_size, max_anc_size)`. Rows 
      smaller than `max_anc_size` are stored in the right segments.
      * idxs_mem: has the same length as `idxs_flat_bat_row_sub`, corresponding to
      the non-zero entries in memory. The non-zeros entries should be stored in 
      the CSR order.
    */
    int bat_size = batIdx.size();  // batch size
    int max_anc_size = find_max_anc_size(batIdx);  // max ancestor size for the batch  
    // Each thread writes on a different vector<int> to avoid conflict 
    std::vector<std::vector<int>> idxs_flat_bat_row_sub_2d(bat_size);
    std::vector<std::vector<int>> idxs_mem_2d(bat_size);
    #pragma omp parallel for
    for(int k = 0; k < bat_size; k++){
        /*
        `k` is the row index in the `bat_size` X `max_anc_size` batch-row.
        `idx` is the row index in the `n` X `n` matrix.
        */
        int idx = batIdx[k]; 
        const auto anc_bgn = anc_idx.begin() + anc_cidx[idx];
        const auto anc_end = anc_idx.begin() + anc_cidx[idx + 1];
        int anc_size = anc_end - anc_bgn;
        // num of entries for padding
        int offset_row_sub = max_anc_size - anc_size;  
        // offset for previous k sub-rows and the first `offset_row_sub` entries
        int offset_idx_flat_k = k * max_anc_size + offset_row_sub;
        for(int j = sp_cidx_csr[idx]; j < sp_cidx_csr[idx + 1]; j++){
            /*
            The `j`-th non-zero coefficient in the sparse n X n matrix 
                in the CSR order.
            `col_idx_row` is the col index of `j`-th non-zero coef in the 
                n X n matrix.
            `col_idx_row_sub` is the col index in the `anc_size`-sized sub-row.
            */
            int col_idx_row = sp_idx_csr[j];  
            int col_idx_row_sub = my_binary_search(anc_bgn, anc_end, 
                col_idx_row);
            if(col_idx_row_sub < anc_size){
                int idx_flat_bat_row_sub = offset_idx_flat_k + col_idx_row_sub;
                idxs_flat_bat_row_sub_2d[k].push_back(idx_flat_bat_row_sub);
                idxs_mem_2d[k].push_back(j);
            }
        }
    }
    // cumulative length of each thread's rslt
    std::vector<int> cidx_mem(bat_size + 1, 0);  
    for(int k = 0; k < bat_size; k++)
        cidx_mem[k + 1] = cidx_mem[k] + idxs_mem_2d[k].size();
    // flatten idxs_flat_bat_row_sub_2d and idxs_mem_2d
    std::vector<int> idxs_flat_bat_row_sub(cidx_mem[bat_size]);
    std::vector<int> idxs_mem(cidx_mem[bat_size]);
    for(int k = 0; k < bat_size; k++){
        std::copy(idxs_flat_bat_row_sub_2d[k].begin(), 
            idxs_flat_bat_row_sub_2d[k].end(), 
            idxs_flat_bat_row_sub.begin() + cidx_mem[k]);
        std::copy(idxs_mem_2d[k].begin(), idxs_mem_2d[k].end(), 
            idxs_mem.begin() + cidx_mem[k]);
    }

    return std::make_tuple(idxs_flat_bat_row_sub, idxs_mem);
}


PYBIND11_MODULE(indexing, m) {
    py::class_<Ancestor>(m, "Ancestor")
        .def(py::init<int, const std::vector<int> &, const std::vector<int> &, 
            const std::vector<int> &, const std::vector<int> &>())
        .def("query", &Ancestor::query)
        .def("query_row", &Ancestor::query_row);
}










