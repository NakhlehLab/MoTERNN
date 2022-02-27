# MoTERNN: Classifying the Mode of Tumor Evolution using Recursive Neural Networks
Tumor evolutionary models have different implications in cancer patients' clinical diagnosis, prognosis, and therapeutic treatment. Four major evolutionary models have been proposed; namely, Linear, Branching, Neutral, and Punctuated models, which are identified from tumor phylogenetic trees. We automated this identification process by defining it as an instance of graph classification problem on phylogenetic trees. Specifically, we employed Recursive Neural Networks to capture the tree structure of tumor phylogenies while predicting the mode of evolution. We trained our model, MoTERNN, using simulated data in a supervised fashion and applied it to a real phylogenetic tree obtained from single-cell DNA-seq data.

This repository contains the PyTorch implementation of MoTERNN and the scripts used for generating simulated data. The implementation of Recursive Neural Network in this project was adopted from https://github.com/mae6/pyTorchTree. To reporduce the results presented in the paper, please follow the instructions below.
## Installing the required packages
Install Python. To better manage Python packages we used Conda. The latest vwesion of Conda can be installed by following instructions in the website https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
Install PyTorch according to the OS and CUDA version. You can find the Conda command for PyTorch and Cuda Toolkit installation from PyTorch's official website https://pytorch.org/get-started/locally/
We used ETE Toolkit for handling tree objects (http://etetoolkit.org). It can be installed using the following command for Conda:
```
conda install -c etetoolkit ete3
```
## Reproducibility
## Contact
If you have any questions, please contact edrisi@rice.edu
