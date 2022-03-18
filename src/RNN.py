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
	def __init__(self, vocabSize, sequenceSize, embedSize=100, numClasses=2):
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

# this function reads the simulated trees 
def get_trees(root_folder, nsample):
	treeList = []
	universal_set = set()
	for i in range(nsample):
		treeList.append(parse_tree(f"{root_folder}tree{i}_0.nw", f"{root_folder}seq{i}_0.csv", label=0, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_1.nw", f"{root_folder}seq{i}_1.csv", label=1, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_2.nw", f"{root_folder}seq{i}_2.csv", label=2, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_3.nw", f"{root_folder}seq{i}_3.csv", label=3, universal_set=universal_set))
	vocab = list(universal_set)
	vocab_dict = {}
	for i, word in enumerate(vocab):
		vocab_dict[word] = i
	for j, tree in enumerate(treeList):
		for node in tree.traverse():
			if node.is_leaf():
				node.add_features(word=vocab_dict[tuple(node.seq)])
	return treeList, len(universal_set)

# this function passes the input variable to the current device
def Var(v):
	return Variable(v.to(device))

def range_limited_float_type(arg):
	# specifying a range of values for the percentage of test data 
	# maximum percentage of test data is 95%
	max_value = 0.95
	# minimum percentage of test data is 5%
	min_value = 0.05
	try:
		arg_value = float(arg)
	except ValueError:    
		raise argparse.ArgumentTypeError("Must be a floating point number")
	if arg_value < min_value or arg_value > max_value:
		raise argparse.ArgumentTypeError("The argument must be between < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
	return arg_value

#################################################
#################### Main #######################
#################################################
if __name__=="__main__":

	parser = argparse.ArgumentParser(description='main script of MoTERNN')
	parser.add_argument('-dir','--dir', help='path to the directory of the simulated data', default="./trees_dir")
	parser.add_argument('-test','--test', help='fraction of data (in percent) to be selected as test data', default=0.25, type=range_limited_float_type)
	parser.add_argument('-val', '--val', help="number of datapoints in validation set", default=100, type=int)
	parser.add_argument('-newick','--newick', help="path to the real data phylogeny in newick format", default='./phylovar.nw')
	parser.add_argument('-seq', '--seq', help="path to the csv file containing the genotypes of the real data", default="./phylovar_seq.csv")
	parser.add_argument('-dim', '--dim', help="embedding dimension for the encoder network", default=256, type=int)
	parser.add_argument('-nsample', '--nsample', help="number of datapoints generated for each mode of evolution (it must match the same argument used in the generator)", default=2000, type=int)
	parser.add_argument('-seed','--seed', help='random seed', default=0, type=int)
	parser.add_argument('-nloci','--nloci', help='number of loci in the genotype profiles (it must match the same arguemnt used in the generator)', default=3375, type=int)
	args = parser.parse_args()

	if args.dir.endswith("/"):
		print("not changing the directory name")
	else:
		args.dir += "/"


	torch.manual_seed(args.seed)
	random.seed(args.seed)
	print(f"input arguments {args}")
	s = time.time()
	print("parsing the input data including real and simulated ...")
	# read phylovar's tree
	print("parsing the real data...")
	real_tree = parse_tree(args.newick, args.seq, label=0, universal_set=set())
	# read the input tree(s)
	print("parsing the trees from the root folder...")
	trees_, num_unique_seqs = get_trees(root_folder=args.dir, nsample=args.nsample)
	print(f"total number of trees loaded {len(trees_)}")

	print(f"parsing all data took {time.time()-s} seconds")
	s = time.time()
	print("start training the model ...")
	random.shuffle(trees_)
	num_datapoints = len(trees_)
	num_test = int(args.test * num_datapoints)
	num_val = args.val
	
	test_, val_, train_ = trees_[0:num_test], trees_[num_test:num_test+num_val], trees_[num_test+num_val:] 
	print("done!")
	model = RNN(vocabSize=num_unique_seqs, sequenceSize=args.nloci, embedSize=args.dim, numClasses=4).to(device)
	max_epochs = 1
	learning_rate = 1e-4
	wd = 0.0
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
	bestAll=bestRoot=0.0

	iter_counter = 0
	loss_list = []
	for epoch in range(max_epochs):
		random.shuffle(train_)
		print(f"epoch {epoch}")
		for step, tree in enumerate(train_):

			# train the model
			model.train()
			predictions, loss, logits = model.getLoss(tree)
			optimizer.zero_grad()
			loss.backward()
			clip_grad_norm_(model.parameters(), 5, norm_type=2.0)
			optimizer.step()
			loss_list.append(loss.item())

			# print the iteration index and loss
			print("iteration: {}, loss: {}".format(iter_counter, loss_list[-1]))
			# increment the iteration index
			iter_counter+=1

	## evaluation of the trained model at the end of training
	model.eval()
	with torch.no_grad():
		# evaluation on the entire training set
		correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(train_)
		print("final accuracy of the model on the training set: {}".format(correctRoot))
		# for a in range(len(incorrects)):
		# 	print(incorrects[a])
		# 	print(incorrects[a].get_tree_root().label)
		# 	print(incorrect_labels[a])
		correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(test_)
		print("final accuracy of the model on the test set: {}".format(correctRoot))
		correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate(val_)
		print("final accuracy of the model on the validation set: {}".format(correctRoot))
		correctRoot, preds, logits, incorrects, incorrect_labels = model.evaluate([real_tree])
		print("real tree root accuracy at the end of training: {}".format(correctRoot))
		print("prediction on real tree:")
		print(preds[-1])
		# print("associtation score of the real tree to all classes:")
		# print(logits)




	print(f"training was done in {time.time()-s} seconds")

	## save the trained model
	target_dir = './moternn.pt'
	torch.save(model.state_dict(), target_dir)
	print(f"the trained model was saved at {os.path.abspath(target_dir)}")








