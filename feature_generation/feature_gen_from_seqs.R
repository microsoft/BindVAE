#library(SeqGL)
suppressMessages(library("GenomicRanges"))
#suppressMessages(library(BSgenome.Hsapiens.UCSC.hg19))
#setwd("~/projects/code/featgen_seqgl")

source("build_train_test_data.R")

no.cores = 10

args <- commandArgs(trailingOnly = TRUE)
bed.file = args[1]
res.dir = args[2]
prefix = args[4]

bed <- read.table(bed.file,header = F, sep="\n",stringsAsFactors=FALSE)
pos.seqs = bed$V1
pos.lens = unlist(lapply(pos.seqs, nchar))
print(max(pos.lens))
print(min(pos.lens))

if (!dir.exists(res.dir))
  dir.create(res.dir, recursive = TRUE)

dictionary.file = "positional_mismatch_dict_kmer8_mismatches2.Rdata"

##############################

train.test.data <- build.train.test.data.seqs(pos.seqs, dictionary.file)

print(dim(train.test.data$features))

    saveRDS(train.test.data, file = sprintf("%s/%s_features.Rds",res.dir,prefix))

