import os
import numpy as np
from doc2vec import read_fasta_to_kmers, train_doc2vec

# load data
surfaceome_data = read_fasta_to_kmers('./data/positive_2638.fasta')
# pretrain doc2vec model
pretrain_d2v = './Doc2Vec_model' 
if not os.path.exists(pretrain_d2v):
    os.makedirs(pretrain_d2v)
train_doc2vec(surfaceome_data,'./Doc2Vec_model/surfaceome_doc2vec.model')
