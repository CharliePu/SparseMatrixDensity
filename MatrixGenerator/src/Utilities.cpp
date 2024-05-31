#include "Utilities.h"


std::string current_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto seconds_since_epoch = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    auto random_number = std::rand() % 1000;
    return std::to_string(seconds_since_epoch) + std::to_string(random_number);
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
