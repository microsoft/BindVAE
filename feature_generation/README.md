Commands to generate features

The feature generation scripts can take either peaks as inputs or DNA-sequences (one sequence per line)

# Input: Peaks in a bed file

Command: Rscript feature_gen_from_peaks.R  <bed-file>  <output-dir>  <out-prefix>

Example:
Rscript feature_gen_from_peaks.R ../datasets/ATACseq/A549_ENCFF548PSN.bed ../features A549


# Input: DNA sequences

Command: Rscript feature_gen_from_seqs.R  <sequences.txt>  <output-dir>  <out-prefix>
