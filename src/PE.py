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

def ne(N, node):

	# number of cells
	n = N
	if n == 1 or n == 0:
		return node
	# alpha
	a = 1e4
	# beta
	b = 1e4

	# sequence of uniform iid values from [0,1]
	us_ = np.random.uniform(0, 1, n-1)
	# sequence of iid values with beta distribution

	bs_ = np.random.beta(float(a+1),float(b+1),n-1)

	# step 1: split the root into left and right child with respective labels 
	j = 0
	# it is assumed that the given node has these two features already created
	node.lb = 0
	node.ub = 1
	l = node.add_child()
	l.add_features(lb=0, ub=bs_[j])
	r = node.add_child()
	r.add_features(lb=bs_[j], ub=1)
	j += 1
	if j == n-1:
		return node
	# step 2: search for the leaf node whose interval covers us_[j], then split it into two child nodes 
	# one with [0,b1b2] interval and the other with [b1b2,1]as its interval
	for leaf in node:
		if leaf.lb < us_[j] and leaf.ub > us_[j]:
			# split the leaf into two child nodes
			l = leaf.add_child() 
			l.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb)*bs_[j])
			r = leaf.add_child()
			r.add_features(lb=leaf.lb + (leaf.ub - leaf.lb)*bs_[j], ub=leaf.ub)
			break
	j += 1
	if j == n-1:
		return node
	# step 3: continue for n-1 steps
	while j<n-1:
		for leaf in node:
			if leaf.lb < us_[j] and leaf.ub > us_[j]:
				# split the leaf into two child nodes
				l = leaf.add_child() 
				l.add_features(lb=leaf.lb, ub=leaf.lb + (leaf.ub - leaf.lb)*bs_[j])
				r = leaf.add_child()
				r.add_features(lb=leaf.lb + (leaf.ub - leaf.lb)*bs_[j], ub=leaf.ub)
				j += 1
				break
	return node

def pe(N=10, n_loci=3375):

	# number of cells
	n = N
	if n == 0:
		print("ERROR: the number of nodes is zero!")
		return None

	# choose the number of dominant clones
	K = random.sample([2,3], 1)[0]
	# choose the number of cells for each clone
	# each clone must have at least two cells, so we sample the remaining 
	c_ = np.random.multinomial(n-1-2*K, [1/(K)]*(K), size=1)[0]
	# # create the backbone tree for PE model 
	t = Tree(name="root")
	normal = t.add_child()
	normal.add_features(lb=0, ub=1)
	tumor = t.add_child()
	tumor.add_features(lb=0, ub=1)

	ne(N=K, node=tumor)
	for i,leaf in enumerate(tumor.get_leaves()):
		ne(N=c_[i]+2, node=leaf)

	# create alphabetical names for the leaves and internal nodes
	k = 0
	for node in t.traverse(strategy="levelorder"):
		node.name = "Taxon"+str(k)
		k+=1
	# assign branch lengths to all branches
	n_loci = n_loci
	n_mutations = 0
	# the normal cell has zero distance to the root
	normal.dist = 0
	n_mutations += normal.dist
	# sample the number of mutations on the long trunk of the tree from
	# a Poisson distribution with lambda = 1000
	tumor.dist = np.random.poisson(100,1)[0]
	# tumor.dist = random.randint(100, 1000)
	n_mutations += tumor.dist
	# sample the number of mutations for clonal branches from 
	# a Poisson distribution with lamda = 5
	for node in tumor.get_descendants():
		node.dist = np.random.poisson(5,1)[0]
		n_mutations += node.dist
	# # assign branch lengths to all branches
	# n_loci = n_loci
	# n_loci = random.randint(int(0.5*n_loci), n_loci)
	# n_mutations = 0
	# # the normal cell has zero distance to the root
	# normal.dist = 0
	# n_mutations += normal.dist
	# n_edges = 1+len(tumor.get_descendants())
	# weights = [1/3] + [(2/(3*(n_edges-1)))]*(n_edges-1)
	# m_ = np.random.multinomial(n_loci, weights, size=1)[0]
	# # # sample the number of mutations on the long trunk of the tree from
	# # # a Poisson distribution with lambda = 1000
	# # tumor.dist = np.random.poisson(100,1)[0]
	# counter = 0
	# tumor.dist = m_[counter]
	# counter += 1
	# # tumor.dist = random.randint(100, 1000)
	# n_mutations += tumor.dist
	# # # sample the number of mutations for clonal branches from 
	# # # a Poisson distribution with lamda = 5
	# for node in tumor.get_descendants():
	# 	node.dist = m_[counter]
	# 	# node.dist = np.random.poisson(5,1)[0]
	# 	n_mutations += node.dist
	# 	counter += 1

	return t, n_mutations

if __name__=="__main__":
	tree, n_muts = pe(N=20, n_loci=3375)
	# tree.write(format=3, outfile="punctuated_example.nw")
	print(tree)
	print(n_muts)
	print(len(tree))







