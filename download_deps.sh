mkdir -p data
cd data

# GloVe
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd -

# Counterfitted vectors
wget https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/counter-fitted-vectors.txt.zip
unzip counter-fitted-vectors.txt.zip
rm counter-fitted-vectors.txt.zip

# IMDB
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvzf aclImdb_v1.tar.gz
rm -f aclImdb_v1.tar.gz

# YELP
# We use the YELP with the version from https://github.com/shentianxiao/language-style-transfer. Please download the repository above and copy language-style-transfer/data/yelp to data_set.
git clone https://github.com/shentianxiao/language-style-transfer
cp -r language-style-transfer/data/yelp ./

# SST-2
mkdir -p sst2
cd sst2
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
rm SST-2.zip
cd -
