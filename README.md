## Leveraging Gene Regulatory Networks for Single-Cell RNA-Seq Imputation
#### Aarthi Venkat, Mamie Wang, Sabrina Su, Jeremy Gygi  

See [Project Outline](https://docs.google.com/document/d/10h6cb9x6TQ07ICjDTP_XGZWstaUCvXFzzkDWunZ6Oig/edit?usp=sharing).  
See [Lab Notebook](https://docs.google.com/document/d/1d5JhdnMPuMJjW2eQ5G-IpQSW2K_oHbssvWabjxENqfo/edit?usp=sharing).

Each cell represents a graph of genes. The task we want to perform is node prediction for each cell  - predicting the gene expression value of a gene that does not have a value due to sc-RNASeq dropout. Features are expressions of other genes for each cell, where the set of other genes can be defined as:
1. Genes considered first-degree neighbors in the gene network graph  
2. All genes except target gene  
3. A set of random genes of size N, where N is the number of first-degree neighbors the gene has in the gene network graph.  

If the gene network graph is a "good" graph, the first-degree neighbors should provide as much or more information than all the genes, and more information than the random set.  

Initial code from Dutil et al and Bertin et al [Github](https://github.com/mila-iqia/gene-graph-conv), altered for the mouse genome, single-cell RNASeq data, and regression task. Primarily using `data/`, `models/`, and `notebooks/`.

Most up-to-date notebook is `notebooks/1.2_MLP_Week8_LN_Regression.ipynb`.
