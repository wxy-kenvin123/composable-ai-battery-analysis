# Ward clustering of the audited articles (K = 6-10), IndVal keywords and DCA.
#
# Usage: Rscript code/cluster.R [output_dir]        (default: rerun/ inside 03_clustering)
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
sec <- dirname(script_dir())
args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args)) args[1] else file.path(sec, "rerun")
dir.create(file.path(run, "data"), recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages({library(cluster); library(vegan)})
ids <- read.csv(file.path(sec, "input", "ids_772.csv"), colClasses = "character")$id
cnt <- read.csv(file.path(sec, "input", "dtm_counts_772.csv"), colClasses = c("character", "character", "integer"))
stopifnot(!anyDuplicated(ids), setequal(unique(cnt$id), ids))
sp <- split(cnt, factor(cnt$id, levels = ids))
a <- lapply(sp, function(r) { v <- as.numeric(r$count); names(v) <- r$token; v }); names(a) <- ids
lens <- vapply(a, length, integer(1)); hinges <- fivenum(lens); keep <- !(lens %in% boxplot.stats(lens)$out) & lens > 0
write.csv(data.frame(id = ids, unique_words = lens, iqr_keep = keep), file.path(run, "data/iqr_membership.csv"), row.names = FALSE)
write.csv(data.frame(q1 = hinges[2], q3 = hinges[4], lower = hinges[2] - 1.5 * (hinges[4] - hinges[2]), upper = hinges[4] + 1.5 * (hinges[4] - hinges[2])), file.path(run, "data/iqr_thresholds.csv"), row.names = FALSE)
a <- a[keep]; lens <- lens[keep]
f <- as.factor(sort(unique(unlist(lapply(a, names))))); f <- f[nchar(as.character(f)) < 20]; vocab <- levels(as.factor(f))
w <- matrix(0, length(vocab), length(a), dimnames = list(vocab, names(a)))
for (j in seq_along(a)) { ii <- match(names(a[[j]]), vocab); ok <- !is.na(ii); w[ii[ok], j] <- as.numeric(a[[j]])[ok] }
w <- sweep(w, 2, lens, "/") * 1000
freq <- 1; nr <- nrow(w); top <- round(nr * .2); occ <- rowSums(w > 0)
while (nr >= top) { nr <- sum(occ > freq); freq <- freq + 1 }; freq <- freq - 1
x <- log(t(w[occ > freq, , drop = FALSE]) + 1); nonzero <- rowSums(x) > 0
write.csv(data.frame(id = rownames(x), nonzero = nonzero), file.path(run, "data/feature_membership.csv"), row.names = FALSE); x <- x[nonzero, , drop = FALSE]
stopifnot(!anyDuplicated(rownames(x)), all(is.finite(x)))
h <- agnes(x, method = "ward"); hc <- as.hclust(h); K <- 6:10
for (k in K) {
  g <- cutree(hc, k)
  write.csv(data.frame(id = rownames(x), Cluster = as.integer(g)), file.path(run, "data", paste0("assignments_k", k, ".csv")), row.names = FALSE)
  avg <- sapply(1:k, function(j) colMeans(x[g == j, , drop = FALSE]))
  fr <- sapply(1:k, function(j) colMeans(x[g == j, , drop = FALSE] > 0))
  iv <- avg / rowSums(avg) * fr
  kw <- do.call(rbind, lapply(1:k, function(j) { ii <- head(order(iv[, j], decreasing = TRUE), 15); data.frame(Cluster = j, word = rownames(iv)[ii], IndVal = iv[ii, j]) }))
  write.csv(kw, file.path(run, "data", paste0("keywords_k", k, ".csv")), row.names = FALSE)
}
z <- decorana(x, iweigh = 0, ira = 0)
saveRDS(list(x = x, h = h, dca = z), file.path(run, "data/cluster_model.rds"))
write.csv(data.frame(id = rownames(x), scores(z, display = "sites", choices = 1:2)), file.path(run, "data/dca_sites.csv"), row.names = FALSE)
cat("Wrote clustering outputs (K = 6..10) to", file.path(run, "data"), "\n")
