# Introduction 
Source code of BindVAE paper on Variational Auto Encoders for learning binding signatures of transcription factors

# Installation
1.	Installation process for the machine learning model
Please create a conda environment as shown below OR using the yaml file: tfp.yaml

conda env create --name tfp python=3.7

conda env create --file=tfp.yaml

2.	Software dependencies
Setting up the conda environment. In some versions of tensorflow / tensorflow-probability, you might get a "KL divergence is negative" error.

conda install tensorflow-gpu

conda install tensorflow-probability

3. Dependencies for the feature generation

install.packages("BiocManager")

BiocManager::install("GenomicRanges")

BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")

install.packages("remotes")

remotes::install_github("ManuSetty/ChIPKernels")



## TRAINING

python bind_vae.py --model_dir test_dir --train_path data/gm12878_all8merfeats_listformat.txt --eval_path data/empty.txt --test_path data/SELEX_probes_features.txt --num_topics 25 --prior_initial_value 10 --mode train --vocab_path data/all8mers_with_wc_vocab.npy

Parameters that are most sensitive and best ones to tweak:
batch_size  (currently set at 32)
num_topics
prior_initial_value
prior_burn_in_steps

## TEST (or getting TOPIC POSTERIORS)

If you want to use a previously saved model to do inference on new data, use the code in "test" mode as follows:

python bind_vae.py --model_dir test_dir --train_path data/gm12878_all8merfeats_listformat.txt --eval_path data/empty.txt --test_path data/SELEX_probes_features.txt --num_topics 25 --prior_initial_value 10 --mode test --vocab_path data/all8mers_with_wc_vocab.npy

Output: a matrix of size N x K, where N = number of examples in the input file, K = number of topics / latent dimensions.

## K-MER DISTRIBUTIONS (DECODER PARAMETERS that encode the TOPIC distributions over words)
python train.py --model_dir ../../test_dir/ --num_topics 25 --prior_initial_value 10 --mode beta --vocab_path ../../data/ATACseq/all8mers_with_wc_vocab.npy

# FILE FORMATS

## Data file format:
A list of feature-ids separated by spaces. The training / test files are formatted as lists of features where if a feature has count k, then it appears k times in the list. Each line of the file is one example.
If you want to change this input format, please look at sparse_matrix_dataset (or let me know and i can help with it). See below for a file with two input examples (documents). The feature ids should be in an increasing order. Also see attached sample file (gm12878_all8merfeats_listformat.txt).

112 113 113 113 122 134 144 144 144 144 159 178
115 115 189 194 194 202 202 202

## Vocabulary format:
Please see the sample vocabulary file (.npy file) for how to format the <feature-id>  <feature-name>  mapping.  It is in a dictionary format. For example, below are the top few lines of the vocabulary for the k-mer model, which was converted into the all8mers_with_wc_vocab.npy file. So, if you load the dictionary, d['EMPTY']=0  and d['AAAAAAAA']=1 and so on. Please keep the first dictionary entry a dummy feature like 'EMPTY' and assign it to the index 0. Obviously, none of the examples will contain this feature :-) This is due to how the indexing is done after loading the vocabulary (i.e. the useful features should have indices >=1).

EMPTY
AAAAAAAA
AAAAAAAC
AAAAAAAG
AAAAAAAT
AAAAAACA
AAAAAACC
AAAAAACG
AAAAAACT
AAAAAAGA
AAAAAAGC
AAAAAAGG

# OUTPUTS

To monitor what the model is learning, you can look at the periodic outputs. The frequency of outputs is controlled by the parameter viz_steps in the code. It is currently set to 20000, but feel free to set it to 1000 or so in the initial runs till you understand what's going on.

Here's what it looks like for k-mers and ATAC-seq peaks. Only the top few are printed. Again this can be controlled by looking at the method get_topics_strings.

elbo
-2646.7239

kl
32.239582

loss
2646.5957

perplexity
79969.914

reconstruction
-2614.485

topics
b'index=92 alpha=4.94 CCGCCNNC NNGGGCGG NNCCGCCC NNGGCGGG CCGCNNCC NNCCCGCC CNNCGCCC CCCGCNNC GCNNCGCC CNNCCGCC'
b'index=14 alpha=1.80 NNCAGAGA NNTCTCTG NNTCTGTG NNCACAGA NNCTCTGT NNACAGAG CACAGNNA CAGAGNNA ANNCACAG NNTCACAG'
b'index=17 alpha=1.74 CCCNNCCC CCNNCCCC CCCCNNCC AGGGGNNG NNGGGGAG NNCTCCCC CNNCCCCC CNNCCCCA CCCCANNC CCCCTNNC'
....

global_step
160000


# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 
