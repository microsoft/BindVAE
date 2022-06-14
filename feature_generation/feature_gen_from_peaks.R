#library(SeqGL)
suppressMessages(library("GenomicRanges"))
suppressMessages(library(BSgenome.Hsapiens.UCSC.hg19))
#setwd("~/projects/code/featgen_seqgl")

source("build_train_test_data.R")

#span=150
span=200
motifs = FALSE
no.cores = 10
org="hg19"
max.examples=100000

args <- commandArgs(trailingOnly = TRUE)
bed.file = args[1]
res.dir = args[2]
feats.file = args[3]
prefix = args[4]

bed <- read.table(bed.file,header = F, sep="\t",stringsAsFactors=FALSE)
peaks <- GRanges(seqnames = bed$V1, ranges = IRanges(bed$V2,bed$V3), summit.pos=bed$V10, score=bed$V5)


if (!dir.exists(res.dir))
  dir.create(res.dir, recursive = TRUE)

dictionary.file = "positional_mismatch_dict_kmer8_mismatches2.Rdata"
#dictionary.file = "wildcard_dict_kmer10_mismatches2_alpha5_consecutive_mis.Rdata" 

##############################


    # ranking the examples by score
    peaks <- peaks[sort(peaks$score, index.return = TRUE,decreasing = TRUE)$ix]
    
    #for each peak, take a region of length "span" centered at summit as positives.
    start(peaks) <- end(peaks) <- start(peaks) + peaks$summit.pos - 1
    peaks <- resize(peaks, fix = "center", span)
    
    #take regions of "2*span" bp upstream of positives as negatives. #neg depend on width of peaks? summary(width(gm.peaks.filter[[1]]))
   # neg.regions <- shift(peaks, span * 2)

    peaks.length <- length (peaks)
    org <- getBSgenome (org)
    exclude = which (end (peaks) > seqlengths (org)[as.character (seqnames (peaks))])
    include <- setdiff ((1:peaks.length), exclude)
    peaks = peaks[include]
    
    #take the sequences.
    pos.seqs <- getSeq(Hsapiens,seqnames(peaks),start(peaks),end(peaks))
    
    #remove any region pair contains N's
    seqs.with.n <- grep("N", pos.seqs)
    #seqs.with.n <- union(grep("N", pos.seqs), grep("N", neg.seqs))
    if (length(seqs.with.n) > 0) {
      show("Some peaks and/or their flanks have Ns in sequences and will not be assigned to any groups")
      peaks <- peaks[-seqs.with.n]
    }
    
    pos.regions <- peaks[1:min(max.examples, length(peaks))]
    #neg.regions <- shift(pos.regions, span * 2)

    #use.inds <- which(countOverlaps(neg.regions, pos.regions) == 0)
    #pos.regions <- pos.regions[use.inds]
    #neg.regions <- neg.regions[use.inds]
    train.test.data <- build.train.test.data(pos.regions, NULL, dictionary.file, org = org, select.top.features = FALSE)

print(dim(train.test.data$features))

    saveRDS(train.test.data, file = sprintf("%s/%s_all8mer_features.Rds",res.dir,prefix))

