# Convert the noun wordbags of the audited articles into a document-term count table.
#
# The wordbags (one cleaned noun sequence per article, produced by pdf_extraction.py)
# keep the word order of the full texts and are not distributed. This script records how the distributed count
# table was made: the same tokenization as the clustering script (03_clustering/code/cluster.R, function readbag),
# applied to each wordbag, written as one row per article and token.
#
# Usage: Rscript wordbags_to_counts.R <wordbag_dir> <ids.csv> <out.csv>
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
ids <- read.csv(args[2], colClasses = "character")$id
paths <- file.path(args[1], paste0(ids, ".txt"))
stopifnot(all(file.exists(paths)), !anyDuplicated(ids))
readbag <- function(p) {
  s <- paste(scan(p, what = "character", sep = "\n", quiet = TRUE), collapse = " ")
  s <- gsub("- ", "", s); s <- gsub("\f", " ", s); s <- gsub("[^A-Za-z[:space:]]", " ", s)
  s <- gsub(" +", " ", s); s <- gsub("\n", " ", s); s <- tolower(trimws(s))
  table(strsplit(s, " "))
}
rows <- lapply(seq_along(ids), function(j) {
  t <- readbag(paths[j])
  data.frame(id = ids[j], token = names(t), count = as.integer(t), stringsAsFactors = FALSE)
})
out <- do.call(rbind, rows)
write.csv(out, args[3], row.names = FALSE)
cat(length(ids), "articles,", nrow(out), "article-token rows\n")
