import os.path as osp
import subprocess
import zlib
import glob
import os
import csv

import torch
from torch_geometric.data import Dataset, Data, download_url

class SparseMatrixDataset(Dataset):
    def __init__(self, root, name):
        self.dataset_name = name
        self.root = root
        self.read_csv(root+"/csv/"+name+".csv")
        super(SparseMatrixDataset, self).__init__(root, None, None)

    def read_csv(self, path):
        if not osp.exists(path):
            raise Exception("Csv file not found at {}".format(path))

        # basename without .csv extension
        self.dataset_name = os.path.basename(path).split('.')[0]

        self.matrix_results = {}
        with open(path, 'r') as f:

            reader = csv.DictReader(f)

            self.matrix_product_results = {}
            for row in reader:

                timestamp = row['timestamp']

                self.matrix_results[timestamp] = {
                    'timestamp': timestamp,
                    'm1_rows': row['matrix 1 rows'],
                    'm1_cols': row['matrix 1 cols'],
                    'm1_nnz': row['matrix 1 nnz'],
                    'm1_nnz_density': row['matrix 1 nnz density'],
                    'm2_rows': row['matrix 2 rows'],
                    'm2_cols': row['matrix 2 cols'],
                    'm2_nnz': row['matrix 2 nnz'],
                    'm2_nnz_density': row['matrix 2 nnz density'],
                    'prod_rows': row['product rows'],
                    'prod_cols': row['product cols'],
                    'prod_nnz': row['product nnz'],
                    'prod_nnz_density': float(row['product nnz density']),
                    'm1_path': row['matrix 1 path'],
                    'm2_path': row['matrix 2 path'],
                    'prod_path': row['product path']
                }

            print("Read matrix results from {}".format(path))          

        self.matrix_names = list(self.matrix_results.keys())
        self.matrix_names.sort()

    @property
    def raw_file_names(self):
        return [f"{key}_{suffix}.mtx" for key in self.matrix_results.keys() for suffix in ["m1", "m2", "product"]]
    
    @property
    def processed_file_names(self):
        return [ f"{key}.pt" for key in self.matrix_results.keys() ]
    
    @property
    def raw_paths(self):
        return [self.root+f"/{self.dataset_name}/"]
    
    @property
    def processed_paths(self):
        return [self.root+f"/../pydataset/{self.dataset_name}/"]
    
    @property
    def num_classes(self):
        return 1    # regression problem
    
    @property
    def num_node_features(self):
        return 16
    
    def len(self):
        return len(self.matrix_results)
    
    def get(self, idx):
        matrices_data = torch.load(f"{self.processed_paths[0]}/{self.matrix_names[idx]}.pt")
        return matrices_data

    def process(self):
        for matrix_name in self.matrix_names:
            print("Processing matrix: ", matrix_name)
            
            m1_path = self.matrix_results[matrix_name]['m1_path']
            m2_path = self.matrix_results[matrix_name]['m2_path']
            prod_nnz_density = self.matrix_results[matrix_name]['prod_nnz_density']

            matrices_data = { "m1": self.process_matrix(m1_path, prod_nnz_density), 
                                "m2": self.process_matrix(m2_path, prod_nnz_density),
                                 "prod_nnz_density": torch.tensor([prod_nnz_density], dtype=torch.float) }
            
            # If pydataset directory does not exist, create it
            if not osp.exists(self.processed_paths[0]):
                os.makedirs(self.processed_paths[0])

            torch.save(matrices_data, f"{self.processed_paths[0]}/{matrix_name}.pt")

    def process_matrix(self, raw_path, prod_nnz_density):
        # if raw_path contains ./dataset, remove it
        # temporary solution. TODO: generate dataset csv without dataset path prefix
        if raw_path.startswith("./dataset"):
            raw_path = self.root + raw_path[9:]

        with open(raw_path, "rb") as f:
            print("\tRead matrix from ", raw_path)
            lines = f.read().decode('utf-8').strip().split('\n')

             # first line header, second line matrix rows cols nnzs
            rows, cols, nnz = lines[1].split()
            lines = lines[2:]

            row_indices = []
            col_indices = []
            values = []

            for line in lines:
                row, col, value = line.split() 
                # use 0-based indexing
                row_indices.append(int(row) - 1) 
                col_indices.append(int(col) - 1)
                values.append(float(value))

            edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)

            num_nodes = max(int(rows), int(cols))

            # print out length
            print("edge_index length: ", edge_index.shape[1])
            
            # Calculate node feature as encoding of degree of each node
            node_features = torch.zeros(num_nodes, self.num_node_features, dtype=torch.float)
            div_term = 10000.0 ** (torch.arange(0.0, self.num_node_features, 2, dtype=torch.float) / self.num_node_features)
            pos = torch.arange(0.0, num_nodes, 1, dtype=torch.float).unsqueeze(1)
            node_features[:, 0::2] = torch.sin(pos / div_term)
            node_features[:, 1::2] = torch.cos(pos / div_term)
            x = node_features

            y = torch.tensor([prod_nnz_density], dtype=torch.float)

            data = Data(x=x, y=y, edge_index=edge_index, 
                        num_nodes=num_nodes)
            
            print("x shape = {}".format(x.shape))
            print("y shape = {}".format(y.shape))

            
            print("Data object before return:", data)

            return data        
        

# test code
if __name__ == '__main__':
    dataset = SparseMatrixDataset(root="./dataset", name="test")