suppressMessages(library("GenomicRanges"))
suppressMessages(library("ChIPKernels"))
suppressMessages(library(BSgenome.Hsapiens.UCSC.hg19))
#setwd("~/projects/code/featgen_seqgl")


#' Function to extract sequences and build feature matrices
#'
#' @param peaks Genomic Ranges object representing positive class regions
#' @param neg.regions Genomic Ranges object representing negative class regions
#' @param dictionary.file Dictionary file for the kernel build using \code{ChIPKernels}
#' @param org Name of the organism. Default is hg19. This assumes that the corresponding BSGenome library is installed
#' @param select.top.features Logical indicating whether feature selection should be performed
#' @param ... Additional arugments to \code{select.top.features}
#' @return List containging feature matrix, labels and indexes for both train and test.
#' @return Matrix of positions indicating the position in the sequence with maximum kmer score.
#' @seealso \code{\link{select.top.features}}
#' @export

build.train.test.data <- function (peaks, neg.regions, 
	dictionary.file, org="hg19",
	select.top.features=TRUE, ...) {

   start.time = Sys.time()

	# Sample examples for train and test
	pos.length <- length (peaks); 

	# Extract sequences
    show ('Extract sequences for all training and test examples...')
    org <- getBSgenome (org)
    pos.seqs <- getSeq(Hsapiens,seqnames(peaks),start(peaks),end(peaks))
	gc ()

    # Check and reomove Ns in sequences
    #exclude <- grep ('N', pos.seqs)
    # Remove peaks which cross the chromosome limits
    #exclude <- c(exclude, 
    #	which (end (peaks) > seqlengths (org)[as.character (seqnames (peaks))]))
    #pos.inds <- setdiff ((1:pos.length), exclude); 
    #pos.seqs = pos.seqs[pos.inds]
    #peaks = peaks[pos.inds]

	# Build train and test features
	show ('Building features from dictionary...')
	features <- build.features.kernels  (dictionary.file, pos.seqs, NULL, FALSE)$features

	# Package and return
	show ('Package and return...')
	results <- list (features=features, peaks=peaks, pos.seqs=pos.seqs,
		dictionary.file = dictionary.file)

     time.end = Sys.time()

    show (sprintf ("Total time for constructing data: %.2f minutes", (time.end - start.time)/60 ))

	return (results)

}


build.train.test.data.new <- function (peaks, pos.seqs, 
	dictionary.file) {

   start.time = Sys.time()

	# Sample examples for train and test
	pos.length <- length (peaks); 

	# Build train and test features
	show ('Building features from dictionary...')
	features <- build.features.kernels  (dictionary.file, pos.seqs, NULL, FALSE)$features

	# Package and return
	show ('Package and return...')
	results <- list (features=features, peaks=peaks, pos.seqs=pos.seqs,
		dictionary.file = dictionary.file)

     time.end = Sys.time()

    show (sprintf ("Total time for constructing data: %.2f minutes", (time.end - start.time)/60 ))

	return (results)

}




build.train.test.data.seqs <- function (pos.seqs, 
	dictionary.file) {

   start.time = Sys.time()

	# Build train and test features
	show ('Building features from dictionary...')
	features <- build.features.kernels  (dictionary.file, pos.seqs, NULL, FALSE)$features

	# Package and return
	show ('Package and return...')
	results <- list (features=features, pos.seqs=pos.seqs,
		dictionary.file = dictionary.file)

     time.end = Sys.time()

    show (sprintf ("Total time for constructing data: %.2f minutes", (time.end - start.time)/60 ))

	return (results)

}




#' Function to select top features
#'
#' @param features Feature matrix 
#' @param labels Feature labels (Should be +1/-1)
#' @param feature.count Number of features to select. Default is 10000.
#' @param min.example.fraction Minimal fraction of examples that a feature should defined in. Default is 1\%.
#' @description The differences are determined by weighted means. The number of examples in which a particular
#' feature is defined is used to estimate the means in both classes.
#' @return Vector of selected feature indexes.
#' @export

select.top.features <- function (features, labels, 
	feature.count = 10000, min.example.fraction=0.01) {

	# Based on non zero means 
	# Determine summary of sparse matrices for faster computation
	pos.summary <- as.matrix (summary (features[labels == 1,]))
	neg.summary <- as.matrix (summary (features[labels == -1,]))

	# Differences of means
	pos.counts <- neg.counts <- pos.means <- neg.means <- rep (0, ncol (features))
	means <- tapply (pos.summary[,3], pos.summary[,2], mean)
	pos.means[as.numeric (names (means))] <- means

	means <- tapply (neg.summary[,3], neg.summary[,2], mean)
	neg.means[as.numeric (names (means))] <- means
	diffs <- abs (pos.means - neg.means)

	# Eliminate examples without significant presence
	counts <- table (pos.summary[,2])
	pos.counts[as.numeric (names (counts))] <- as.vector (counts)
	counts <- table (neg.summary[,2])
	neg.counts[as.numeric (names (counts))] <- as.vector (counts)
	inds <- which (pos.counts < min.example.fraction * length (which (labels == 1)) &
		neg.counts < min.example.fraction * length (which (labels == -1)))
	diffs[inds] <- 0

	# Sort and pick features
	feature.count <- min (feature.count, length (which (diffs > 0)))
	return (sort (diffs, index.return=TRUE, decreasing=TRUE)$ix[1:feature.count])
}
