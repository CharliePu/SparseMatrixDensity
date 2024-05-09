import sys
sys.path.append('./MatrixGenerator/lib')
from MatrixGenerator import generate_entry

import random
import csv
import os


dataset_name = 'matrices_more_distributions'
dataset_path = './dataset/' + dataset_name
total_matrices = 1000

# Random matrix combinations
max_nnz = 20000
nnz_density = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
matrix_size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
row_sparsity = [0.0, 0.5, 0.9]
col_sparsity = [0.0, 0.5, 0.9]
diag_sparsity = [0.0, 0.5, 0.9]
symmetric = [True, False]

dataset_entries = []

for i in range(total_matrices):
    results = generate_entry(dataset_path, 
                    random.choice(matrix_size), 
                    max_nnz,
                    # matrix 1
                    random.choice(nnz_density), 
                    random.choice(row_sparsity), 
                    random.choice(col_sparsity), 
                    random.choice(diag_sparsity), 
                    random.choice(symmetric),
                    # matrix 2
                    random.choice(nnz_density),
                    random.choice(row_sparsity),
                    random.choice(col_sparsity),
                    random.choice(diag_sparsity),
                    random.choice(symmetric))
    
        # return boost::python::make_tuple(
        # timestamp,
        # m1_path.string(), 
        # entry.m1.rows(),
        # entry.m1.cols(),
        # entry.m1.nonZeros(),
        # m2_path.string(), 
        # entry.m2.rows(),
        # entry.m2.cols(),
        # entry.m2.nonZeros(),
        # product_path.string(), 
        # entry.prod.rows(),
        # entry.prod.cols(),
        # entry.prod.nonZeros(),
        # entry.product_nnz_density);
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


    
