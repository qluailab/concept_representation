# Concept Embedding
This software is the implementation of the paper "Incorporation Coupling Relationships for 
Concept Embedding Representation".


## Installation
Concept Embedding is designed for Python 2.7 recommends Anaconda for managing your environment. 
We'd recommend creating a custom environment as follows:
```bash
conda create --name concept_embeding python=2.7
conda activate concept_embeding
``` 
and Python packages we use can be installed via pip:
```bash
pip install -r requirements.txt
```

## Data
Download dataset from [wikidump](https://dumps.wikimedia.org/enwiki/) and put them in the ./Dataset/Corpus.  
Select the files in the following two formats:  
enwiki-xxxxxxxx-pages-articles-multistream.xml.bz2  
enwiki-xxxxxxxx-pages-articles-multistream-index.txt.bz2

## Preprocessed data
We modified [WikiExtractor.py](https://github.com/attardi/wikiextractor) in order to extract concept more efficiently.  
You need to change the name of enwiki-xxxxxxxx-pages-articles-multistream.xml.bz2 in preprocess.sh
```bash
bash preprocess.sh
```

