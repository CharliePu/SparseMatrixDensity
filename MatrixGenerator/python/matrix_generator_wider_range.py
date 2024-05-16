import sys
sys.path.append('./MatrixGenerator/lib')
from MatrixGenerator import generate_entry_square_matrix

import random
import csv
import os


dataset_name = 'wider_range'
dataset_path = './dataset/' + dataset_name
total_matrices = 5000

# Random matrix combinations
max_nnz = 100000
matrix_size_range = [2.0, 5.0]
nnz_sparsity_range = [-4.0, -1.0]
row_sparsity_range = [-4.0, 0.0]
col_sparsity_range = [-4.0, 0.0]
diag_sparsity_range = [-4.0, 0.0]
symmetric = [True, False]

dataset_entries = []

for i in range(total_matrices):
    matrix_size = 10 ** random.uniform(matrix_size_range[0], matrix_size_range[1]) # 10^2 to 10^5

    nnz_density_1 = 5 * (10 ** random.uniform(nnz_sparsity_range[0], nnz_sparsity_range[1])) # 5*10^-4 to 5*10^-1
    row_sparsity_1 = 10.0 ** random.uniform(row_sparsity_range[0], row_sparsity_range[1]) # 10^-4 to 10^0
    col_sparsity_1 = 10.0 ** random.uniform(col_sparsity_range[0], col_sparsity_range[1]) # 10^-4 to 10^0
    diag_sparsity_1 = 10.0 ** random.uniform(diag_sparsity_range[0], diag_sparsity_range[1]) # 10^-4 to 10^0

    nnz_density_2 = 5.0 * (10.0 ** random.uniform(nnz_sparsity_range[0], nnz_sparsity_range[1])) # 5*10^-4 to 5*10^-1
    row_sparsity_2 = 10.0 ** random.uniform(row_sparsity_range[0], row_sparsity_range[1]) # 10^-4 to 10^0
    col_sparsity_2 = 10.0 ** random.uniform(col_sparsity_range[0], col_sparsity_range[1]) # 10^-4 to 10^0
    diag_sparsity_2 = 10.0 ** random.uniform(diag_sparsity_range[0], diag_sparsity_range[1]) # 10^-4 to 10^0

    results = generate_entry_square_matrix(dataset_path, 
                    int(matrix_size),
                    max_nnz,
                    # matrix 1
                    nnz_density_1,
                    row_sparsity_1,
                    col_sparsity_1,
                    diag_sparsity_1,
                    random.choice(symmetric),
                    # matrix 2
                    nnz_density_2,
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
    dataset_entries.append([str(timestamp), m1_rows, m1_cols, m1_nnz, m1_nnz_density, m2_rows, m2_cols, m2_nnz, m2_nnz_density, prod_rows, prod_cols, prod_nnz, prod_nnz_density, m1_path, m2_path, prod_path])
    print(i+1, 'of', total_matrices, 'done')


# If directory does not exist, create it
if not os.path.exists('./dataset/csv'):
    os.makedirs('./dataset/csv')

with open('./dataset/csv/'+dataset_name+'.csv', mode='w') as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    dataset_writer.writerow(['timestamp', 'matrix 1 rows', 'matrix 1 cols', 'matrix 1 nnz', 'matrix 1 nnz density', 'matrix 2 rows', 'matrix 2 cols', 'matrix 2 nnz', 'matrix 2 nnz density', 'product rows', 'product cols', 'product nnz', 'product nnz density', 'matrix 1 path', 'matrix 2 path', 'product path'])
    for entry in dataset_entries:
        dataset_writer.writerow(entry)

print('Done!')


    
