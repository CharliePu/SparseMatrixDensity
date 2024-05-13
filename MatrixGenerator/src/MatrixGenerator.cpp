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

std::function<int64_t(std::default_random_engine &)> select_random_generator(std::default_random_engine &gen, int64_t min_val, int64_t max_val, std::string debug_name="")
{
    std::uniform_int_distribution<int> dist_type(0, 2);

    std::cout<<debug_name<<": ";

    switch (dist_type(gen))
    {
    case 0:
    {
        std::uniform_int_distribution<int64_t> dist(min_val, max_val);
        std::cout << "Uniform Distribution: Min=" << min_val << ", Max=" << max_val << std::endl;
        return [dist](std::default_random_engine &gen) mutable
        { return dist(gen); };
    }
    case 1:
    {
        double mean = (max_val + min_val) / 2;
        double stddev = (max_val - min_val) / 4;
        std::normal_distribution<double> dist(mean, stddev);
        std::cout << "Normal Distribution: Mean=" << mean << ", StdDev=" << stddev << std::endl;
        return [dist, min_val, max_val](std::default_random_engine &gen) mutable
        {
            return std::max(min_val, std::min(max_val, static_cast<int64_t>(dist(gen))));
        };
    }
    case 2:
    {
        // Calculate p based on covering 95% within the interval (not sure if this is correct)
        double desired_coverage = 0.95;
        double p = 1 - std::pow(1.0 - desired_coverage, 1.0 / (max_val - min_val + 1));

        std::geometric_distribution<int64_t> dist(p);
        std::cout << "Geometric Distribution: p=" << p << std::endl;
        return [dist, min_val, max_val](std::default_random_engine &gen) mutable
        {
            return std::max(min_val, std::min(max_val, static_cast<int64_t>(dist(gen))));
        };
    }
    }

    std::cerr << "Invalid distribution type selected." << std::endl;
    return nullptr;
}

Eigen::SparseMatrix<bool, 0, int64_t> generate_matrix(int64_t rows, int64_t cols, int64_t max_nnz, float nnz_sparsity, float row_sparsity, float col_sparsity, float diag_sparsity, bool symmetric)
{
    std::random_device rd;
    std::default_random_engine gen(rd());

    std::cout<<"Generating matrix with size "<<rows<<"x"<<cols<<", max_nnz="<<max_nnz<<", nnz_sparsity="<<nnz_sparsity<<", row_sparsity="<<row_sparsity<<", col_sparsity="<<col_sparsity<<", diag_sparsity="<<diag_sparsity<<", symmetric="<<symmetric<<std::endl;

    auto rows_gen = select_random_generator(gen, 0, rows - 1, "row_gen");
    auto cols_gen = select_random_generator(gen, 0, cols - 1, "col_gen");
    auto excluded_diags_gen = select_random_generator(gen, -rows + 1, cols - 1, "excluded_diags_gen");

    // Setup constraints
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
    std::cout << "Selected rows: " << selected_rows.size() << std::endl;

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
    std::cout << "Selected cols: " << selected_cols.size() << std::endl;

    std::unordered_set<int64_t> excluded_diags;
    int64_t target_diags_exclude_count = std::round((rows + cols - 1) * diag_sparsity);
    target_diags_exclude_count = std::min(target_diags_exclude_count, rows + cols - 2); // At least one diag should be included

    for (int64_t i = 0; i < target_diags_exclude_count * 100; i++)
    {
        if (excluded_diags.size() >= target_diags_exclude_count)
        {
            break;
        }
        excluded_diags.insert(excluded_diags_gen(gen));
    }
    std::cout << "Excluded diags: " << excluded_diags.size() << std::endl;

    // Generate all possibilities
    std::vector<std::pair<int64_t, int64_t>> all_elements;

    // Fill in elements that satisfy the constraints
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

    // If no elements are selected, lose the constraints and try again
    if (all_elements.size() == 0)
    {
        return generate_matrix(rows, cols, max_nnz, nnz_sparsity * 0.9, row_sparsity * 0.9, col_sparsity * 0.9, diag_sparsity * 0.9, symmetric);
    }

    // Randomly remove some elements if there are too many
    std::shuffle(all_elements.begin(), all_elements.end(), std::default_random_engine());
    max_nnz = std::min(max_nnz, std::min(static_cast<int64_t>(all_elements.size()), static_cast<int64_t>(rows * cols * (1.0 - nnz_sparsity))));
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

    std::cout << "Matrix generated with size " << rows << "x" << cols << " and " << max_nnz << " non-zero elements." << std::endl<<std::endl;

    // Print a few elements
    for (int i = 0; i < std::min(10l, static_cast<int64_t>(triplets.size())); i++)
    {
        std::cout << "Element " << i << ": (" << triplets[i].row() << ", " << triplets[i].col() << ")" << std::endl;
    }

    // Create matrix instance
    Eigen::SparseMatrix<bool, 0, int64_t> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    return matrix;
}

DataSetEntry generate_entry_helper(int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, int64_t max_nnz,
                                   float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,

                                   float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    DataSetEntry entry;
    entry.m1 = generate_matrix(m1_rows, m1_cols_and_m2_rows, max_nnz, m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric);
    entry.m2 = generate_matrix(m1_cols_and_m2_rows, m2_cols, max_nnz, m2_nnz_sparsity, m2_row_sparsity, m2_col_sparsity, m2_diag_sparsity, m2_symmetric);

    entry.m1_nn_density = static_cast<float>(entry.m1.nonZeros()) / (m1_rows * m1_cols_and_m2_rows);
    entry.m2_nn_density = static_cast<float>(entry.m2.nonZeros()) / (m1_cols_and_m2_rows * m2_cols);

    Eigen::SparseMatrix<bool, 0, int64_t> m1_dense(entry.m1), m2_dense(entry.m2);

    entry.prod = entry.m1 * entry.m2;

    entry.product_nnz_density = static_cast<float>(entry.prod.nonZeros()) / (m1_rows * m2_cols);

    return entry;
}

boost::python::tuple generate_entry(std::string path, int64_t m1_rows, int64_t m1_cols_and_m2_rows, int64_t m2_cols, int64_t max_nnz,
                                    float m1_nnz_sparsity, float m1_row_sparsity, float m1_col_sparsity, float m1_diag_sparsity, bool m1_symmetric,
                                    float m2_nnz_sparsity, float m2_row_sparsity, float m2_col_sparsity, float m2_diag_sparsity, bool m2_symmetric)
{
    // Create paths if not exist
    std::filesystem::create_directories(path);

    auto entry = generate_entry_helper(m1_rows, m1_cols_and_m2_rows, m2_cols, max_nnz, m1_nnz_sparsity, m1_row_sparsity, m1_col_sparsity, m1_diag_sparsity, m1_symmetric,
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