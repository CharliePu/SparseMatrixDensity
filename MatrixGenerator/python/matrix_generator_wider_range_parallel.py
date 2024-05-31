import sys
import random
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append('./MatrixGenerator/lib')
from MatrixGenerator import generate_entry_square_matrices

dataset_name = 'wider_range'
dataset_path = './dataset/' + dataset_name
total_matrices = 50000

# Random matrix combinations
max_nnz = 100000
matrix_size_range = [1.8, 3.5]
nnz_sparsity_range = [-10.0, 0.0]
row_sparsity_range = [-10.0, 0.0]
col_sparsity_range = [-10.0, 0.0]
diag_sparsity_range = [-10.0, 0.0]
symmetric = [True, False]

def generate_dataset_entry(dataset_path, max_nnz, matrix_size_range, nnz_sparsity_range, row_sparsity_range, col_sparsity_range, diag_sparsity_range, symmetric):
    matrix_size = 10 ** random.uniform(matrix_size_range[0], matrix_size_range[1])

    nnz_sparsity_1 = 1.0 - 10 ** random.uniform(nnz_sparsity_range[0], nnz_sparsity_range[1])
    row_sparsity_1 = 1.0 - 10.0 ** random.uniform(row_sparsity_range[0], row_sparsity_range[1])
    col_sparsity_1 = 1.0 - 10.0 ** random.uniform(col_sparsity_range[0], col_sparsity_range[1])
    diag_sparsity_1 = 1.0 - 10.0 ** random.uniform(diag_sparsity_range[0], diag_sparsity_range[1])

    nnz_sparsity_2 = 1.0 - 10.0 ** random.uniform(nnz_sparsity_range[0], nnz_sparsity_range[1])
    row_sparsity_2 = 1.0 - 10.0 ** random.uniform(row_sparsity_range[0], row_sparsity_range[1])
    col_sparsity_2 = 1.0 - 10.0 ** random.uniform(col_sparsity_range[0], col_sparsity_range[1])
    diag_sparsity_2 = 1.0 - 10.0 ** random.uniform(diag_sparsity_range[0], diag_sparsity_range[1])

    results = generate_entry_square_matrices(dataset_path, 
                    int(matrix_size),
                    max_nnz,
                    # matrix 1
                    nnz_sparsity_1,
                    row_sparsity_1,
                    col_sparsity_1,
                    diag_sparsity_1,
                    random.choice(symmetric),
                    # matrix 2
                    nnz_sparsity_2,
                    row_sparsity_2,
                    col_sparsity_2,
                    diag_sparsity_2,
                    random.choice(symmetric))

    timestamp = results[0]
    m1_path = results[1]
    m1_rows = results[2]
    m1_cols = results[3]
    m1_nnz = results[4]
    m1_nnz_density = m1_nnz / (m1_rows * m1_cols)
    m2_path = results[5]
    m2_rows = results[6]
    m2_cols = results[7]
    m2_nnz = results[8]
    m2_nnz_density = m2_nnz / (m2_rows * m2_cols)
    prod_path = results[9]
    prod_rows = results[10]
    prod_cols = results[11]
    prod_nnz = results[12]
    prod_nnz_density = results[13]

    # use str(timestamp) to avoid scientific notation
    return [str(timestamp), m1_rows, m1_cols, m1_nnz, m1_nnz_density, m2_rows, m2_cols, m2_nnz, m2_nnz_density, prod_rows, prod_cols, prod_nnz, prod_nnz_density, m1_path, m2_path, prod_path]

dataset_entries = []

# If directory does not exist, create it
if not os.path.exists('./dataset/csv'):
    os.makedirs('./dataset/csv')

# Use ProcessPoolExecutor for parallel execution
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(generate_dataset_entry, dataset_path, max_nnz, matrix_size_range, nnz_sparsity_range, row_sparsity_range, col_sparsity_range, diag_sparsity_range, symmetric) for _ in range(total_matrices)]

    with tqdm(total=total_matrices, file=sys.stdout) as pbar:
        for future in as_completed(futures):
            dataset_entries.append(future.result())
            pbar.update(1)

with open('./dataset/csv/' + dataset_name + '.csv', mode='w') as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    dataset_writer.writerow(['timestamp', 'matrix 1 rows', 'matrix 1 cols', 'matrix 1 nnz', 'matrix 1 nnz density', 'matrix 2 rows', 'matrix 2 cols', 'matrix 2 nnz', 'matrix 2 nnz density', 'product rows', 'product cols', 'product nnz', 'product nnz density', 'matrix 1 path', 'matrix 2 path', 'product path'])
    for entry in dataset_entries:
        dataset_writer.writerow(entry)

print('Done!')
