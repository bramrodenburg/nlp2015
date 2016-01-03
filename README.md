# Natural Language Processing 2015
For our Natural Language Processing course we implemented a LDA and MG-LDA algorithm using Gibbs Sampling. 
## Retrieving the dataset
The preprocessing part for the LDA and MG-LDA models was written to read and process product reviews from the Multi-Domain Sentiment dataset (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/). 

Known issue: due to a 'bug' in the preprocessing part not the entire 'eletronics' dataset from the Multi-Domain Sentiment dataset can be loaded. All other datasets are, as far as we know, processed correctly.

## Running the LDA program
The LDA program can be run with the following commands:
```
python LDA.py preprocessing-boolean path-to-dataset-directory
```
For example:
```
python LDA.py True data/eletronics/
```
## Running the MG-LDA program
