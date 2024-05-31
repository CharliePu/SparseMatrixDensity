#include "MatrixGenerator.h"
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>

using namespace Eigen;

std::function<int64_t(std::default_random_engine &)> select_random_generator(std::default_random_engine &gen, int64_t min_val, int64_t max_val, std::string debug_name)
{
    std::uniform_int_distribution<int> dist_type(0, 2);

    switch (dist_type(gen))
    {
    case 0:
    {
        std::uniform_int_distribution<int64_t> dist(min_val, max_val);
        return [dist](std::default_random_engine &gen) mutable
        { return dist(gen); };
    }
    case 1:
    {
        double mean = (max_val + min_val) / 2;
        double stddev = (max_val - min_val) / 4;
        std::normal_distribution<double> dist(mean, stddev);
        return [dist, min_val, max_val](std::default_random_engine &gen) mutable
        {
            return std::max(min_val, std::min(max_val, static_cast<int64_t>(dist(gen))));
        };
    }
    case 2:
    {
        double desired_coverage = 0.95;
        double p = 1 - std::pow(1.0 - desired_coverage, 1.0 / (max_val - min_val + 1));

        std::geometric_distribution<int64_t> dist(p);
        return [dist, min_val, max_val](std::default_random_engine &gen) mutable
        {
            return std::max(min_val, std::min(max_val, static_cast<int64_t>(dist(gen))));
        };
    }
    }

    std::cerr << "Invalid distribution type selected." << std::endl;
    return nullptr;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_helper(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity, float col_sparsity, float diag_sparsity, bool symmetric, int recursion_count)
{
    std::random_device rd;
    std::default_random_engine gen(rd());

    auto rows_gen = select_random_generator(gen, 0, rows - 1, "row_gen");
    auto cols_gen = select_random_generator(gen, 0, cols - 1, "col_gen");
    auto excluded_diags_gen = select_random_generator(gen, -rows + 1, cols - 1, "excluded_diags_gen");

    std::unordered_set<int64_t> selected_rows_set;
    int64_t target_rows_count = std::round(rows * (1.0 - row_sparsity));
    target_rows_count = std::max(target_rows_count, 1l);
    for (int64_t i = 0; i < target_rows_count * 100; i++)
    {
        if (selected_rows_set.size() >= target_rows_count)
        {
            break;
        }
        selected_rows_set.insert(rows_gen(gen));
    }
    std::vector<int64_t> selected_rows(selected_rows_set.begin(), selected_rows_set.end());

    std::unordered_set<int64_t> selected_cols_set;
    int64_t target_cols_count = std::round(cols * (1.0 - col_sparsity));
    target_cols_count = std::max(target_cols_count, 1l);
    for (int64_t i = 0; i < target_cols_count * 100; i++)
    {
        if (selected_cols_set.size() >= target_cols_count)
        {
            break;
        }
        selected_cols_set.insert(cols_gen(gen));
    }
    std::vector<int64_t> selected_cols(selected_cols_set.begin(), selected_cols_set.end());

    std::unordered_set<int64_t> excluded_diags;
    int64_t target_diags_exclude_count = std::round((rows + cols - 1) * diag_sparsity);
    target_diags_exclude_count = std::min(target_diags_exclude_count, rows + cols - 2);

    for (int64_t i = 0; i < target_diags_exclude_count * 100; i++)
    {
        if (excluded_diags.size() >= target_diags_exclude_count)
        {
            break;
        }
        excluded_diags.insert(excluded_diags_gen(gen));
    }

    std::vector<std::pair<int64_t, int64_t>> all_elements;

    for (int64_t row = 0; row < rows; row++)
    {
        for (int64_t col = 0; col < (symmetric ? std::min(row + 1, cols) : cols); col++)
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
            if (symmetric && row != col && col < rows && row < cols)
            {
                all_elements.push_back({col, row});
            }
        }
    }

    if (all_elements.size() == 0)
    {
        if (recursion_count > 10)
        {
            std::cerr << "Failed to generate matrix after 10 attempts." << std::endl;
            return Eigen::SparseMatrix<bool, 0, int64_t>(rows, cols);
        }
        return generate_matrix_helper(rows, cols, max_nnz, nnz_sparsity * 0.9, row_sparsity * 0.9, col_sparsity * 0.9, diag_sparsity * 0.9, symmetric, recursion_count + 1);
    }

    std::shuffle(all_elements.begin(), all_elements.end(), std::default_random_engine());
    max_nnz = std::min(max_nnz, std::min(static_cast<int64_t>(all_elements.size()), static_cast<int64_t>(rows * cols * (1.0 - nnz_sparsity))));
    all_elements.resize(max_nnz);

    std::sort(all_elements.begin(), all_elements.end(), [](const std::pair<int64_t, int64_t> &a, const std::pair<int64_t, int64_t> &b)
              { return a.first < b.first || (a.first == b.first && a.second < b.second); });

    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (auto &pair : all_elements)
    {
        triplets.emplace_back(pair.first, pair.second, 1);
    }

    Eigen::SparseMatrix<bool, 0, int64_t> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity, float col_sparsity, float diag_sparsity, bool symmetric)
{
    return generate_matrix_helper(rows, cols, max_nnz, nnz_sparsity, row_sparsity, col_sparsity, diag_sparsity, symmetric, 0);
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_one_row(int64_t size, int64_t max_nnz, float nnz_sparsity)
{
    auto engine = std::default_random_engine{};
    std::bernoulli_distribution dist(1.0 - nnz_sparsity);
    
    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (int64_t i = 0; i < size; i++)
    {
        if (dist(engine))
        {
            triplets.emplace_back(0, i, 1);
        }
    }

    Eigen::SparseMatrix<bool, 0, int64_t> matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_one_col(int64_t size, int64_t max_nnz, float nnz_sparsity)
{
    auto engine = std::default_random_engine{};
    std::bernoulli_distribution dist(1.0 - nnz_sparsity);
    
    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (int64_t i = 0; i < size; i++)
    {
        if (!dist(engine))
        {
            triplets.emplace_back(i, 0, 1);
        }
    }

    Eigen::SparseMatrix<bool, 0, int64_t> matrix(size, size);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_multiple_cols(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::bernoulli_distribution nnz_dist(1.0 - nnz_sparsity);

    std::unordered_set<int64_t> selected_rows_set;
    int64_t target_rows_count = std::round(rows * (1.0 - row_sparsity));
    target_rows_count = std::max(target_rows_count, int64_t(1));

    for (int64_t i = 0; i < target_rows_count * 100; ++i)
    {
        if (selected_rows_set.size() >= target_rows_count)
        {
            break;
        }
        selected_rows_set.insert(std::uniform_int_distribution<int64_t>(0, rows - 1)(gen));
    }

    std::vector<int64_t> selected_rows(selected_rows_set.begin(), selected_rows_set.end());

    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (const auto& row : selected_rows)
    {
        for (int64_t col = 0; col < cols; ++col)
        {
            if (nnz_dist(gen))
            {
                triplets.emplace_back(row, col, true);
            }
        }
    }

    if (triplets.size() > max_nnz)
    {
        std::shuffle(triplets.begin(), triplets.end(), gen);
        triplets.resize(max_nnz);
    }

    Eigen::SparseMatrix<bool, 0, int64_t> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix_multiple_rows(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float col_sparsity)
{
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::bernoulli_distribution nnz_dist(1.0 - nnz_sparsity);

    std::unordered_set<int64_t> selected_cols_set;
    int64_t target_cols_count = std::round(cols * (1.0 - col_sparsity));
    target_cols_count = std::max(target_cols_count, int64_t(1));

    for (int64_t i = 0; i < target_cols_count * 100; ++i)
    {
        if (selected_cols_set.size() >= target_cols_count)
        {
            break;
        }
        selected_cols_set.insert(std::uniform_int_distribution<int64_t>(0, cols - 1)(gen));
    }

    std::vector<int64_t> selected_cols(selected_cols_set.begin(), selected_cols_set.end());

    std::vector<Eigen::Triplet<bool, int64_t>> triplets;
    for (int64_t row = 0; row < rows; ++row)
    {
        for (const auto& col : selected_cols)
        {
            if (nnz_dist(gen))
            {
                triplets.emplace_back(row, col, true);
            }
        }
    }

    if (triplets.size() > max_nnz)
    {
        std::shuffle(triplets.begin(), triplets.end(), gen);
        triplets.resize(max_nnz);
    }

    Eigen::SparseMatrix<bool, 0, int64_t> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}