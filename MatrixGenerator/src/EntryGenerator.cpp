#include "EntryGenerator.h"
#include <filesystem>
#include <iostream>

#include "Utilities.h"

DataSetEntry generate_entry_helper(int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, 
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m1_matrix_generator,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m2_matrix_generator)
{
    DataSetEntry entry;
    entry.m1 = m1_matrix_generator();
    entry.m2 = m2_matrix_generator();

    entry.m1_nnz_density = static_cast<float>(entry.m1.nonZeros()) / (m1_rows * m1_cols_and_m2_rows);
    entry.m2_nnz_density = static_cast<float>(entry.m2.nonZeros()) / (m1_cols_and_m2_rows * m2_cols);

    entry.prod = entry.m1 * entry.m2;

    entry.product_nnz_density = static_cast<float>(entry.prod.nonZeros()) / (m1_rows * m2_cols);

    return entry;
}

boost::python::tuple generate_entry(std::string path, int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m1_matrix_generator,
                                    const std::function<Eigen::SparseMatrix<bool, 0, int64_t>()> &m2_matrix_generator)
{
    std::filesystem::create_directories(path);

    auto entry = generate_entry_helper(m1_rows, m1_cols_and_m2_rows, m2_cols, m1_matrix_generator, m2_matrix_generator);

    std::string timestamp = current_timestamp();

    std::filesystem::path m1_path = path + "/" + timestamp + "_m1.mtx";
    if (!save_matrix(m1_path, entry.m1))
    {
        std::cerr << "Failed to save matrix 1 to " << m1_path << std::endl;
    }

    std::filesystem::path m2_path = path + "/" + timestamp + "_m2.mtx";
    if (!save_matrix(m2_path, entry.m2))
    {
        std::cerr << "Failed to save matrix 2 to " << m2_path << std::endl;
    }

    std::filesystem::path product_path = path + "/" + timestamp + "_product.mtx";
    if (!save_matrix(product_path, entry.prod))
    {
        std::cerr << "Failed to save product to " << product_path << std::endl;
    }

    return boost::python::make_tuple(
        timestamp,
        m1_path.string(), 
        entry.m1.rows(),
        entry.m1.cols(),
        entry.m1.nonZeros(),
        m2_path.string(), 
        entry.m2.rows(),
        entry.m2.cols(),
        entry.m2.nonZeros(),
        product_path.string(), 
        entry.prod.rows(),
        entry.prod.cols(),
        entry.prod.nonZeros(),
        entry.product_nnz_density);
}

boost::python::tuple generate_entry_rectangle_matrices(std::string path, int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, int64_t max_nnz, 
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    auto m1_generator = [=]() { 
        return generate_matrix(m1_rows, m1_cols_and_m2_rows, max_nnz, m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric);
    };
    auto m2_generator = [=]() {
        return generate_matrix(m1_cols_and_m2_rows, m2_cols, max_nnz, m2_nnz_sparsity, m2_row_sparsity, m2_col_sparsity, m2_diag_sparsity, m2_symmetric);
    };

    return generate_entry(path, m1_rows,m1_cols_and_m2_rows, m2_cols, m1_generator, m2_generator);
}

boost::python::tuple generate_entry_square_matrices(std::string path, int64_t size, int64_t max_nnz,
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    return generate_entry_rectangle_matrices(path, size, size, size, max_nnz, 
                                    m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric,
                                    m2_nnz_sparsity, m2_row_sparsity, m2_col_sparsity, m2_diag_sparsity, m2_symmetric);
}

boost::python::tuple generate_entry_horizontal_vertical_product(std::string path, int64_t size, int64_t max_nnz, float m1_nnz_sparsity, float m2_nnz_sparsity)
{
    auto m1_generator = [=]() { 
        return generate_matrix(size, size, max_nnz, m1_nnz_sparsity, 0.0, 1.0, 0.0, false);
    };

    auto m2_generator = [=]() {
        return generate_matrix_one_col(size, size, m2_nnz_sparsity);
    };

    return generate_entry(path, size, size, size, m1_generator, m2_generator);
}

boost::python::tuple generate_entry_inner_product(std::string path, int64_t size, float m1_nnz_sparsity, float m2_nnz_sparsity)
{
    auto m1_generator = [=]() { 
        return generate_matrix_one_row(size, size, m1_nnz_sparsity);
    };

    auto m2_generator = [=]() {
        return generate_matrix_one_col(size, size, m2_nnz_sparsity);
    };

    return generate_entry(path, size, size, size, m1_generator, m2_generator);
} 

boost::python::tuple generate_entry_outer_product(std::string path, int64_t size, float m1_nnz_sparsity, float m2_nnz_sparsity)
{
    auto m1_generator = [=]() { 
        return generate_matrix_one_col(size, size, m1_nnz_sparsity);
    };

    auto m2_generator = [=]() {
        return generate_matrix_one_row(size, size, m2_nnz_sparsity);
    };

    return generate_entry(path, size, size, size, m1_generator, m2_generator);
}

boost::python::tuple generate_entry_extreme_cases(std::string path, int64_t size, int64_t max_nnz, float m1_nnz_sparsity, float m1_row_col_sparsity, float m2_nnz_sparsity, float m2_row_col_sparsity)
{   
    auto gen1 = [=]() { 
        return generate_matrix_multiple_cols(size, size, max_nnz, m1_nnz_sparsity, m1_row_col_sparsity);
    };

    auto gen2 = [=]() {
        return generate_matrix_multiple_rows(size, size, max_nnz, m2_nnz_sparsity, m2_row_col_sparsity);
    };

    std::array<std::function<Eigen::SparseMatrix<bool, 0, int64_t>()>, 2> generators = {gen1, gen2};

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    // Randomly select the generators
    auto m1_generator = generators[dis(gen)];
    auto m2_generator = generators[dis(gen)];

    return generate_entry(path, size, size, size, m1_generator, m2_generator);
}
