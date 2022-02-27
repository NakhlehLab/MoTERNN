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

def ne(N=20, n_loci=3375):

	# number of cells
	n = N
	if n == 0:
		print("ERROR: the number of nodes is zero!")
		return None
	# alpha
	a = 1e4
	# beta
	b = 1e4

	# sequence of uniform iid values from [0,1]
	us_ = np.random.uniform(0, 1, n-1)
	# sequence of iid values with beta distribution
	bs_ = np.random.beta(float(a+1),float(b+1),n-1)

	# create a root 
	t = Tree(name="root")
	# step 1: split the root into left and right child with respective labels 
	j = 0
	t.add_features(lb=0, ub=1)
	l = t.add_child()
	l.add_features(lb=0, ub=bs_[j])
	r = t.add_child()
	r.add_features(lb=bs_[j], ub=1)
	j += 1
	# step 2: search for the leaf node whose interval covers us_[j], then split it into two child nodes 
	# one with [0,b1b2] interval and the other with [b1b2,1]as its interval
	for leaf in t:
		if leaf.lb < us_[j] and leaf.ub > us_[j]:
			# split the leaf into two child nodes
			l = leaf.add_child() 
			l.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb)*bs_[j])
			r = leaf.add_child()
			r.add_features(lb=leaf.lb + (leaf.ub - leaf.lb)*bs_[j], ub=leaf.ub)
			break
	j += 1
	# step 3: continue for n-1 steps
	while j<n-1:
		for leaf in t:
			if leaf.lb < us_[j] and leaf.ub > us_[j]:
				# split the leaf into two child nodes
				l = leaf.add_child() 
				l.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb)*bs_[j])
				r = leaf.add_child()
				r.add_features(lb=leaf.lb + (leaf.ub - leaf.lb)*bs_[j], ub=leaf.ub)
				j += 1
				break
	# create alphabetical names for the leaves and internal nodes
	k = 1
	for node in t.traverse(strategy="levelorder"):
		node.name = "Taxon"+str(k)
		k+=1
	# assign branch lengths to all branches
	n_loci = n_loci
	n_mutations = 0
	for node in t.get_descendants():
		node.dist = np.random.poisson(5,1)[0]
		n_mutations += node.dist

	return t, n_mutations

if __name__=="__main__":
	tree, n_muts = ne(N=200, n_loci=3375)
	# tree.write(format=3, outfile="neutral_example.nw")
	print(tree)
	print(n_muts)








