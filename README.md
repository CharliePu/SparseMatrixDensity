<!-- # Generate matrices
## Build matrix generator
``` bash
cd MatrixGenerator
mkdir build
cd build
cmake ..
cmake --build . --target MatrixGenerator
```

## Run matrix generator -->

# Run model
## Environment setup
Requires pytorch, pytorch geometric, matplotlib and tqdm.

## Train model
``` bash
python ./GCNModel/gcn_model.py DATASET_NAME
```

## Evaluate model
In `./GCNModel/evaluate.ipynb` notebook, change `dataset_name` to the correct name.