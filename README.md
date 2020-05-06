## Refine gene interaction network using graph attention networks
#### Aarthi Venkat, Mamie Wang, Sabrina Su, Jeremy Gygi  

Each cell represents a graph of genes. The task we want to perform is node prediction for each cell  - predicting the gene expression value of a gene that does not have a value due to sc-RNASeq dropout. Features are expressions of other genes for each cell, where the set of other genes can be defined as the first-degree neighbors in the gene network graph  

As an attempt to refine networks for data-specific analysis, we aim to leverage Graph Attention Network architecture to perform gene expression imputation on single-cell RNA-sequencing data from a single cell type Cd8, which will in turn elucidate refined functional networks within Cd8 T cells in our datasets.
