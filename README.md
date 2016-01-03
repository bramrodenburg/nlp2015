# Natural Language Processing 2015
For our Natural Language Processing course we implemented a LDA and MG-LDA algorithm using Gibbs Sampling. The programs also come with a preprocessing part that can read and process product reviews from the Multi-Domain Sentiment dataset [1].
## Retrieving the dataset
The preprocessing part for the LDA and MG-LDA models was written to read and process product reviews from the Multi-Domain Sentiment dataset [1].

Known issue: due to a 'bug' in the preprocessing part not the entire 'eletronics' dataset from the Multi-Domain Sentiment dataset can be loaded. All other datasets are, as far as we know, processed correctly.

## Running the LDA/MG-LDA program
The LDA/MG-LDA program can be run with the following commands:
```
python LDA.py <preprocessing-boolean> <path-to-dataset-directory>
python MGLDA.py <preprocessing-boolean> <path-to-dataset-directory>
```
For example:
```
python LDA.py True data/electronics/
python MGLDA.py True data/electronics/
```
In the LDA/MG-LDA program the parameters such as the # topics, # gibbs iterations and model parameters can be tuned by configuration global variables in the Python script itself.

## References
[1] Multi-Domain Sentiment dataset, https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
