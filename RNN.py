import sys
import os
import random
random.seed(0)
import torch 
torch.manual_seed(0)
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import ete3
from ete3 import Tree
import pandas as pd
import time
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
			# currentNode = self.activation(self.embedding(Var(torch.LongTensor([node.word]))))
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
		# calculate the accuracy of the model
		n = nAll = correctRoot = correctAll = 0.0
		for j, tree in enumerate(trees):
			predictions, loss, logits = self.getLoss(tree)
			correct = (predictions.data == self.labelList.data)
			# correctAll += correct.sum()
			# nAll += correct.squeeze().size()[0]
			# correctRoot += correct.squeeze()[-1]
			correctRoot += correct.squeeze()
			n += 1
		# return correctRoot/n, correctAll/nAll, predictions
		return correctRoot/n, predictions, logits

# this function parses an individual tree along with its genotype matrix 
def parse_tree(newick_f, sequence_f, label, universal_set):
	# read the tree from newick file
	tree = Tree(newick_f, format=8)
	# read the sequences from the csv file
	seq_df = pd.read_csv(sequence_f, index_col=0)

	# add the sequences and labels to the nodes 
	# root = tree.get_tree_root()
	# root.add_features(seq=list(seq_df["Taxon1"]))
	# root.add_features(label=label)
	# universal_set.add(tuple(list(seq_df["Taxon1"])))
	for node in tree.traverse(strategy="levelorder"):
		node.add_features(label=label)
		# if not node.is_root():
		if node.is_leaf():
			node.add_features(seq=list(seq_df[node.name]))
			universal_set.add(tuple(list(seq_df[node.name])))
	return tree

# this function reads the simulated trees 
def get_trees(root_folder):
	treeList = []
	universal_set = set()
	for i in range(2000):
		treeList.append(parse_tree(f"{root_folder}tree{i}_0.nw", f"{root_folder}seq{i}_0.csv", label=0, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_1.nw", f"{root_folder}seq{i}_1.csv", label=1, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_2.nw", f"{root_folder}seq{i}_2.csv", label=2, universal_set=universal_set))
		treeList.append(parse_tree(f"{root_folder}tree{i}_3.nw", f"{root_folder}seq{i}_3.csv", label=2, universal_set=universal_set))
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

#################################################
#################### Main #######################
#################################################
if __name__=="__main__":

	s = time.time()
	print("parsing the input data including real and simulated ....")
	# read phylovar's tree
	print("parsing the real data...")
	real_tree = parse_tree(f"./phylovar.nw", f"./phylovar_seq.csv", label=0, universal_set=set())
	real_tree_original = parse_tree(f"./phylovar_original.nw", f"./phylovar_seq.csv", label=0, universal_set=set())
	# read the input tree(s)
	print("parsing the trees from the root folder...")
	trees_, num_unique_seqs = get_trees(root_folder="./trees/")
	print(f"total number of trees loaded {len(trees_)}")

	print(f"parsing all data took {time.time()-s} seconds")
	s = time.time()
	print("start training the model ...")
	random.shuffle(trees_)
	test_, val_, train_ = trees_[0:2000], trees_[2000:2100], trees_[2100:] 
	print("done!")
	model = RNN(vocabSize=num_unique_seqs, sequenceSize=3375, embedSize=512, numClasses=4).to(device)
	max_epochs = 1
	learning_rate = 1e-5
	wd = 0.0
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
	bestAll=bestRoot=0.0

	iter_counter = 0
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

			# print the iteration index and loss
			print("iteration: {}, loss: {}".format(iter_counter, loss_list[-1]))
			# increment the iteration index
			iter_counter+=1

	## evaluation of the trained model at the end of training
	model.eval()
	with torch.no_grad():
		# evaluation on the entire training set
		correctRoot, preds, logits = model.evaluate(train_)
		print("final accuracy of the model on the training set: {}".format(correctRoot))
		correctRoot, preds, logits = model.evaluate(test_)
		print("final accuracy of the model on the test set: {}".format(correctRoot))
		correctRoot, preds, logits = model.evaluate(val_)
		print("final accuracy of the model on the validation set: {}".format(correctRoot))
		correctRoot, preds, logits = model.evaluate([real_tree])
		print("phylovar's tree root accuracy at the end of training: {}".format(correctRoot))
		print("prediction on phylovar's tree:")
		print(preds[-1])
		print("prediction on all classes:")
		print(logits)




	print(f"training was done in {time.time()-s} seconds")

	## save the trained model
	torch.save(model.state_dict(), 'moternn.pt')








