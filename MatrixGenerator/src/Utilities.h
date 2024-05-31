#ifndef UTILITIES_H
#define UTILITIES_H

#include <Eigen/Sparse>
#include <filesystem>
#include <fstream>

std::string current_timestamp();

bool save_matrix(std::filesystem::path path, const Eigen::SparseMatrix<bool, 0, int64_t> &matrix);

#endif // UTILITIES_H