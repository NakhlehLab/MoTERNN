# MoTERNN 
## Classifying the Mode of Tumor Evolution using Recursive Neural Networks
Tumor evolutionary models have different implications in cancer patients' clinical diagnosis, prognosis, and therapeutic treatment. Four major evolutionary models have been proposed; namely, Linear, Branching, Neutral, and Punctuated models, which are identified from tumor phylogenetic trees. We automated this identification process by defining it as an instance of graph classification problem on phylogenetic trees. Specifically, we employed Recursive Neural Networks to capture the tree structure of tumor phylogenies while predicting the mode of evolution. We trained our model, MoTERNN, using simulated data in a supervised fashion and applied it to a real phylogenetic tree obtained from single-cell DNA-seq data.
## Description of the directories
This repository contains the PyTorch implementation of MoTERNN and the scripts used for generating simulated data. Also, we have provided the trained model and the real data that we applied our method on in the study. Specifically we have:
- `src`: contains all the scripts for training the RNN model and simulating the training data.
- `data`: contains the real dataset including the phylogenetic tree (in Newick format) and the csv file of the genotypes at the leaves. 

The implementation of Recursive Neural Network in this project was adopted from https://github.com/mae6/pyTorchTree. To reporduce the results presented in the paper, please follow the instructions below in [Reproducibility](https://github.com/NakhlehLab/MoTERNN#reproducibility).
## How to install required packages
### Python installation
To better manage Python packages we used Conda. The latest vwesion of Conda can be installed by following instructions in the website https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

### PyTorch installation
Install PyTorch according to the OS and CUDA version. You can find the Conda command for PyTorch and Cuda Toolkit installation from PyTorch's official website https://pytorch.org/get-started/locally/

### ETE Toolkit installation
We used [ETE Toolkit](http://etetoolkit.org) for handling tree objects. It can be installed using the following command for Conda:
```
conda install -c etetoolkit ete3
```
## Reproducibility
  1. ### Simulation of the training data
     Download this package, unzip it, and navigate to the main directory named MoTERNN. To make sure `generator.py` works and see the arguments, run:
     ```
     python generator.py --help
     ```
     which shows the following message:
     ```
     This script generates simulated trees for training MoTERNN
     
     optional arguments:
     -h, --help            show this help message and exit
     -lb LB, --lb LB       minimum number of cells for each phylogeny
     -ub UB, --ub UB       maximum number of cells for each phylogeny
     -nloci NLOCI, --nloci NLOCI
                        number of loci in the genotype profiles
     -nsample NSAMPLE, --nsample NSAMPLE
                        number of datapoints generated for each mode of evolution
     -dir DIR, --dir DIR   destination directory to save the simulated data
     -seed SEED, --seed SEED
                        random seed
     ```
     To generate the same simulated data we used in the paper, run `generator.py` with the default parameters which is simply done by the following command:
     ```
     python generator.py
     ```
     This will run the generator with the default parameters (that we used for simulation study in the paper) and will create a directory named `trees_dir` containing 8000 .nw and .csv files for the four modes of evoltion (2000 datapoints for each mode).
## Contact
If you have any questions, please contact edrisi@rice.edu or edrisi.rice@gmail.com
