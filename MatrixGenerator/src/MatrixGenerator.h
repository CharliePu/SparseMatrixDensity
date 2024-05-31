#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <Eigen/SparseCore>
#include <functional>
#include <string>
#include <random>
#include <boost/python.hpp>

struct DataSetEntry
{
    Eigen::SparseMatrix<bool, 0, int64_t> m1, m2, prod;
    float m1_nnz_density, m2_nnz_density, product_nnz_density;
};

std::function<int64_t(std::default_random_engine &)> select_random_generator(std::default_random_engine &gen, int64_t min_val, int64_t max_val, std::string debug_name = "");

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity, float col_sparsity, float diag_sparsity, bool symmetric);

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_one_row(int64_t size, int64_t max_nnz, float nnz_sparsity);

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_one_col(int64_t size, int64_t max_nnz, float nnz_sparsity);

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_multiple_cols(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float col_sparsity);

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_multiple_rows(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity);

#endif // MATRIX_GENERATOR_H
