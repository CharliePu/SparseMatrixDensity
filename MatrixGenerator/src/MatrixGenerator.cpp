#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <fstream>
#include <chrono>
#include <unordered_set>

#include <boost/python.hpp>
#include <Eigen/SparseCore>

using namespace Eigen;

std::string current_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto seconds_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return std::to_string(seconds_since_epoch);
}

bool save_matrix(std::filesystem::path path, const Eigen::SparseMatrix<bool, 0, int64_t> &matrix)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        return false;
    }

    file << "%%MatrixMarket matrix coordinate real general" << std::endl;

    file << matrix.rows() << " " << matrix.cols() << " " << matrix.nonZeros() << std::endl;

    for (int k = 0; k < matrix.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<bool, 0, int64_t>::InnerIterator it(matrix, k); it; ++it)
        {
            file << it.row() + 1 << " " << it.col() + 1 << " " << it.value() << std::endl;
        }
    }

    return true;
}

struct DataSetEntry
{
    Eigen::SparseMatrix<bool, 0, int64_t> m1, m2, prod;
    float m1_nn_density, m2_nn_density, product_nnz_density;
};

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix(int64_t size, int64_t max_nnz, float nnz_sparsity, float row_sparsity, float col_sparsity, float diag_sparsity, bool symmetric)
{

    std::random_device rd;
    std::default_random_engine gen(rd());

    // Setup constraints
    std::unordered_set<int64_t> selected_rows_set;
    int64_t target_rows_count = size * (1.0 - row_sparsity);
    for (int64_t i = 0; i < target_rows_count * 100; i++)
    {
        if (selected_rows_set.size() >= target_rows_count)
        {
            break;
        }
        selected_rows_set.insert(std::uniform_int_distribution<int64_t>(0, size - 1)(gen));
    }
    std::vector<int64_t> selected_rows(selected_rows_set.begin(), selected_rows_set.end());
    std::cout << "Selected rows: " << selected_rows.size() << std::endl;

    std::unordered_set<int64_t> selected_cols_set;
    int64_t target_cols_count = size * (1.0 - col_sparsity);
    for (int64_t i = 0; i < target_cols_count * 100; i++)
    {
        if (selected_cols_set.size() >= target_cols_count)
        {
            break;
        }
        selected_cols_set.insert(std::uniform_int_distribution<int64_t>(0, size - 1)(gen));
    }
    std::vector<int64_t> selected_cols(selected_cols_set.begin(), selected_cols_set.end());
    std::cout << "Selected cols: " << selected_cols.size() << std::endl;

    std::unordered_set<int64_t> excluded_diags;
    int64_t target_diags_count = size * diag_sparsity * 2;
    for (int64_t i = 0; i < target_diags_count * 100; i++)
    {
        if (excluded_diags.size() >= target_diags_count)
        {
            break;
        }
        excluded_diags.insert(std::uniform_int_distribution<int64_t>(-size + 1, size - 1)(gen));
    }
    std::cout << "Excluded diags: " << excluded_diags.size() << std::endl;

    // Generate all possibilities
    std::vector<std::pair<int64_t, int64_t>> all_elements;

    // Fill in elements that satisfy the constraints
    for (int64_t row = 0; row < size; row++)
    {
        for (int64_t col = 0; col < (symmetric ? row : size); col++)
        {
            if (excluded_diags.find(row - col) != excluded_diags.end())
            {
                continue;
            }

            if (selected_rows_set.find(row) == selected_rows_set.end() || selected_cols_set.find(col) == selected_cols_set.end())
            {
                continue;
            }

            all_elements.push_back({row, col});
            all_elements.push_back({col, row});
        }
    }

    // If no elements are selected, lose the constraints and try again
    if (all_elements.size() == 0)
    {
        return generate_matrix(size, max_nnz, nnz_sparsity * 0.9, row_sparsity * 0.9, col_sparsity * 0.9, diag_sparsity * 0.9, symmetric);
    }

    // Randomly remove some elements if there are too many
    std::shuffle(all_elements.begin(), all_elements.end(), gen);
    max_nnz = std::min(max_nnz, std::min(static_cast<int64_t>(all_elements.size()), static_cast<int64_t>(size * size * (1.0 - nnz_sparsity))));
    all_elements.resize(max_nnz);

    // Sort elements
    std::sort(all_elements.begin(), all_elements.end(), [](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
              { return a.first < b.first || (a.first == b.first && a.second < b.second); });
    std::cout << "Selected elements: " << all_elements.size() << std::endl;

    // Matrix data structure
    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (auto &pair : all_elements)
    {
        triplets.emplace_back(pair.first, pair.second, 1);
    }

    std::cout << "Generating matrix with size " << size << "x" << size << " and " << max_nnz << " non-zero elements." << std::endl<<std::endl;

    // Create matrix instance
    Eigen::SparseMatrix<bool, 0, int64_t> matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

DataSetEntry generate_entry_helper(int64_t size, int64_t max_nnz,
                                   float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,

                                   float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    DataSetEntry entry;
    entry.m1 = generate_matrix(size, max_nnz, m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric);
    entry.m2 = generate_matrix(size, max_nnz, m2_nnz_sparsity, m2_row_sparsity, m2_col_sparsity, m2_diag_sparsity, m2_symmetric);

    entry.m1_nn_density = static_cast<float>(entry.m1.nonZeros()) / (size * size);
    entry.m2_nn_density = static_cast<float>(entry.m2.nonZeros()) / (size * size);

    Eigen::SparseMatrix<bool, 0, int64_t> m1_dense(entry.m1), m2_dense(entry.m2);

    entry.prod = entry.m1 * entry.m2;

    entry.product_nnz_density = static_cast<float>(entry.prod.nonZeros()) / (size * size);

    return entry;
}

boost::python::tuple generate_entry(std::string path, int64_t size, int64_t max_nnz,
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    // Create paths if not exist
    std::filesystem::create_directories(path);

    auto entry = generate_entry_helper(size, max_nnz, m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric,
                                       m2_nnz_sparsity, m2_row_sparsity, m2_col_sparsity, m2_diag_sparsity, m2_symmetric);

    std::string timestamp = current_timestamp();

    std::filesystem::path m1_path = path + "/" + current_timestamp() + "_m1.mtx";
    if (!save_matrix(m1_path, entry.m1))
    {
        std::cerr << "Failed to save matrix 1 to " << m1_path << std::endl;
    }

    std::filesystem::path m2_path = path + "/" + current_timestamp() + "_m2.mtx";
    if (!save_matrix(m2_path, entry.m2))
    {
        std::cerr << "Failed to save matrix 2 to " << m2_path << std::endl;
    }

    std::filesystem::path product_path = path + "/" + current_timestamp() + "_product.mtx";
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

BOOST_PYTHON_MODULE(MatrixGenerator)
{
    using namespace boost::python;
    def("generate_entry", generate_entry);
}