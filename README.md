# PyTorchTree
Use PyTorch Tensors to Build Regression Trees

The functions as whole give a solution to fitting a complete regression binary tree only using PyTorch tensors.
1. use init_tree to make two tensors to hold the tree split information and predictions at tree leaves
2. use build_tree to fill the two tensors from init_tree
3. use forward_tree to make predictions
