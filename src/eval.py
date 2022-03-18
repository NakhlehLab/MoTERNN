import sys
import os
import random
import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import ete3
from ete3 import Tree
import pandas as pd
import time
import argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# identify the available device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this code is adopted from https://github.com/aykutfirat/pyTorchTree

class RNN(nn.Module):
	def __init__(self, sequenceSize, embedSize=100, numClasses=2):
		super(RNN, self).__init__()
		# dictionary for each genomic sequence 
		# self.embedding = nn.Embedding(int(vocabSize), embedSize)
		self.embedding = nn.Linear(sequenceSize, embedSize)
		# combining the embedding of the children nodes into one embedding 
		self.W = nn.Linear(2*embedSize, embedSize, bias=True)
		# prediction of the node class given the embedding of a node 
		self.projection = nn.Linear(embedSize, numClasses, bias=True)
		# activation function 
		self.activation = F.relu
		# list of class predictions for all the nodes 
		self.nodeProbList = []
		# list of node labels
		self.labelList = []

	# this is a recursive function that traverses the tree from the root 
	# applies the function recursively on all the nodes 
	def traverse(self, node):
		# if the node is a leaf, then only the leaf sequence is considered
		if node.is_leaf():
			currentNode = self.activation(self.embedding(Var(torch.FloatTensor(node.seq)[None,:])))
		# if node is internal, then the embedding of the node will be a function of the children nodes' embeddings
		else:
			currentNode = self.activation(self.W(torch.cat((self.traverse(node.get_children()[0]),self.traverse(node.get_children()[1])),1)))
		if node.is_root():
			# add the class probabilities for the current node into a list
			self.nodeProbList.append(self.projection(currentNode))
			# add the label of the current node into a list
			self.labelList.append(torch.LongTensor([node.label]))
		return currentNode

	def forward(self, x):
		self.nodeProbList = []
		self.labelList = []
		# update the above lists by traversing the given tree
		self.traverse(x)
		self.labelList = Var(torch.cat(self.labelList))
		return torch.cat(self.nodeProbList)

	def getLoss(self, tree):
		# get the probabilities 
		nodes = self.forward(tree)
		predictions = nodes.max(dim=1)[1]
		loss = F.cross_entropy(input=nodes, target=self.labelList)
		return predictions, loss, nodes

	def evaluate(self, trees):
		incorrects = []
		incorrect_labels = []
		# calculate the accuracy of the model
		n = nAll = correctRoot = correctAll = 0.0
		for j, tree in enumerate(trees):
			predictions, loss, logits = self.getLoss(tree)
			correct = (predictions.data == self.labelList.data)
			if predictions.data != self.labelList.data:
				incorrects.append(tree)
				incorrect_labels.append(predictions.data)
			correctRoot += correct.squeeze()
			n += 1
		return correctRoot/n, predictions, logits, incorrects, incorrect_labels

# this function parses an individual tree along with its genotype matrix 
def parse_tree(newick_f, sequence_f, label, universal_set):
	# read the tree from newick file
	tree = Tree(newick_f, format=8)
	# read the sequences from the csv file
	seq_df = pd.read_csv(sequence_f, index_col=0)

	for node in tree.traverse(strategy="levelorder"):
		node.add_features(label=label)
		# if not node.is_root():
		if node.is_leaf():
			node.add_features(seq=list(seq_df[node.name]))
			universal_set.add(tuple(list(seq_df[node.name])))
	return tree

# this function passes the input variable to the current device
def Var(v):
	return Variable(v.to(device))

#################################################
#################### Main #######################
#################################################
if __name__=="__main__":

	parser = argparse.ArgumentParser(description='main script of MoTERNN')
	parser.add_argument('-newick','--newick', help="path to the real data phylogeny in newick format", default='./phylovar.nw')
	parser.add_argument('-seq', '--seq', help="path to the csv file containing the genotypes of the real data", default="./phylovar_seq.csv")
	parser.add_argument('-dim', '--dim', help="embedding dimension for the encoder network", default=256, type=int)
	parser.add_argument('-seed','--seed', help='random seed', default=0, type=int)
	parser.add_argument('-nloci','--nloci', help='number of loci in the genotype profiles (it must match the same arguemnt used in the generator)', default=3375, type=int)
	parser.add_argument('-model', '--model', help='path to the trained model')
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	print(f"input arguments {args}")
	s = time.time()
	print("parsing the input data including real and simulated ...")
	# read phylovar's tree
	print("parsing the real data...")
	real_tree = parse_tree(args.newick, args.seq, label=0, universal_set=set())
	print(f"parsing all data took {time.time()-s} seconds")
	s = time.time()
	model = RNN(sequenceSize=args.nloci, embedSize=args.dim, numClasses=4).to(device)
	## evaluation of the trained model at the end of training
	dict_ = {0:"Punctuated mode", 1:"Neutral mode", 2:"Linear mode", 3:"Branching mode"}
	model.load_state_dict(torch.load(args.model))
	model.eval()
	with torch.no_grad():
		correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate([real_tree])
		print(f"prediction on real tree: {dict_[preds[-1].item()]}")








