import ete3
from ete3 import Tree
import numpy as np 
np.random.seed(0)
import random
random.seed(0)
import uuid
import copy
import pandas as pd
import sys 
import os
import PE
from PE import pe
import LE
from LE import le
import NE
from NE import ne
import mutation_assigner
from mutation_assigner import assign

if __name__=="__main__":
	# minimum number of single-cells
	c_lb = 20
	# maximum number of single-cells
	c_ub = 100
	# the number of loci
	n_loci = 3375
	# specify the name of the directory 
	target_dir = "./trees_example_dataset/"
	# specify the number of trees that will be generated for each class
	num_trees = 1000
	counter = 0
	print(f"simulating {num_trees} trees ...")
	try:
	    # Create target Directory
	    os.mkdir(target_dir)
	    print("Directory " , target_dir ,  " Created ") 
	except FileExistsError:
	    print("Directory " , target_dir ,  " already exists")

	# generate trees for PE model
	for i in range(num_trees):

		outname = f"{target_dir}tree{i}_0.nw"
		seq_file = f"{target_dir}seq{i}_0.csv"
		n_cells = random.randint(c_lb, c_ub)
		print("generate trees for PE model...")
		print(f"tree number {i+1}")
		print(f"number of desired nodes for the tree {n_cells}")
		tree, n_muts = pe(N=n_cells,n_loci=n_loci)
		print(f"number of nodes in the generated tree {len(tree)}")
		if n_muts <= n_loci:
			counter += 1
			tree, seq_dict = assign(t=tree,n_loci=n_loci)
			tree.write(format=8, outfile=outname)
			df = pd.DataFrame.from_dict(seq_dict)
			df.to_csv(seq_file)

	# generate trees for NE model
	for i in range(num_trees):
		
		outname = f"{target_dir}tree{i}_1.nw"
		seq_file = f"{target_dir}seq{i}_1.csv"
		n_cells = random.randint(c_lb, c_ub)
		print("generate trees for NE model...")
		print(f"tree number {i+1}")
		print(f"number of desired nodes for the tree {n_cells}")
		tree, n_muts = ne(N=n_cells,n_loci=n_loci)
		print(f"number of nodes in the generated tree {len(tree)}")
		if n_muts <= n_loci:
			counter += 1
			tree, seq_dict = assign(t=tree,n_loci=n_loci)
			tree.write(format=8, outfile=outname)
			df = pd.DataFrame.from_dict(seq_dict)
			df.to_csv(seq_file)


	# generate trees for LE model
	for i in range(num_trees):

		outname = f"{target_dir}tree{i}_2.nw"
		seq_file = f"{target_dir}seq{i}_2.csv"
		n_cells = random.randint(c_lb, c_ub)
		print("generate trees for LE model...")
		print(f"tree number {i+1}")
		print(f"number of desired nodes for the tree {n_cells}")
		tree, n_muts = le(N=n_cells, n_loci=n_loci)
		print(f"number of nodes in the generated tree {len(tree)}")
		if n_muts <= n_loci:
			counter += 1
			tree, seq_dict = assign(t=tree,n_loci=n_loci)
			tree.write(format=8, outfile=outname)
			df = pd.DataFrame.from_dict(seq_dict)
			df.to_csv(seq_file)

	# generate trees for BE model
	for i in range(num_trees):

		outname = f"{target_dir}tree{i}_3.nw"
		seq_file = f"{target_dir}seq{i}_3.csv"
		n_cells = random.randint(30, c_ub)
		print("generate trees for BE model...")
		print(f"tree number {i+1}")
		print(f"number of desired nodes for the tree {n_cells}")
		tree, n_muts = le(N=n_cells, n_loci=n_loci)
		print(f"number of nodes in the generated tree {len(tree)}")
		if n_muts <= n_loci:
			counter += 1
			tree, seq_dict = assign(t=tree,n_loci=n_loci)
			tree.write(format=8, outfile=outname)
			df = pd.DataFrame.from_dict(seq_dict)
			df.to_csv(seq_file)

	print("done!")
	print(f"total number of generated trees: {counter}")




