#ifndef ENTRY_GENERATOR_H
#define ENTRY_GENERATOR_H

#include <boost/python.hpp>
#include <string>
#include <functional>

#include "MatrixGenerator.h"

DataSetEntry generate_entry_helper(int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, 
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m1_matrix_generator,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m2_matrix_generator);

boost::python::tuple generate_entry(std::string path, int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m1_matrix_generator,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m2_matrix_generator);

boost::python::tuple generate_entry_rectangle_matrices(std::string path, int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, int64_t max_nnz, 
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric);

boost::python::tuple generate_entry_square_matrices(std::string path, int64_t size, int64_t max_nnz,
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric);

boost::python::tuple generate_entry_horizontal_vertical_product(std::string path, int64_t size, int64_t max_nnz, float m1_nnz_sparsity, float m2_nnz_sparsity);

boost::python::tuple generate_entry_inner_product(std::string path, int64_t size, float m1_nnz_sparsity, float m2_nnz_sparsity);

boost::python::tuple generate_entry_outer_product(std::string path, int64_t size, float m1_nnz_sparsity, float m2_nnz_sparsity);

boost::python::tuple generate_entry_extreme_cases(std::string path, int64_t size, int64_t max_nnz, float m1_nnz_sparsity, float m1_row_col_sparsity, float m2_nnz_sparsity, float m2_row_col_sparsity);

#endif // ENTRY_GENERATOR_H
