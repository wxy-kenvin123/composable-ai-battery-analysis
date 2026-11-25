# IndVal of the K = 8 partition (labdsv::indval, 1,000 permutations) and DCA species scores.
#
# Usage: Rscript code/adapt_model.R [output_dir]     (default: rerun/)
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
code <- script_dir(); sec <- dirname(code)
args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args)) args[1] else file.path(sec, "rerun")
dir.create(file.path(run, "data", "clustering"), recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages({library(cluster); library(vegan); library(labdsv)})
z <- readRDS(file.path(run, "data/cluster_model.rds")); x <- z$x; h <- z$h; model <- z$dca; g <- cutree(as.hclust(h), 8)
stopifnot(identical(names(g), rownames(x)))
set.seed(20260920); iv <- labdsv::indval(x, g, numitr = 1000)
coords <- as.data.frame(vegan::scores(model, display = "species", choices = 1:2)); names(coords) <- c("DCA1", "DCA2"); stopifnot(identical(rownames(coords), colnames(x)))
fc <- data.frame(round(iv$indval, 2), pval = iv$pval, word = names(iv$pval), coords, check.names = FALSE)
fc <- fc[fc$pval < .05 & rowMeans(fc[, 1:8]) > .01, , drop = FALSE]
subs <- lapply(1:8, function(i) { q <- fc[head(order(fc[, i], decreasing = TRUE), 5), , drop = FALSE]; q$cluster <- i; q })
write.csv(data.frame(article = names(g), Cluster = as.integer(g)), file.path(run, "data/clustering/article_clusters_analysis.csv"), row.names = FALSE)
write.csv(data.frame(word = colnames(x), iv$indval, pval = iv$pval, coords), file.path(run, "data/indval_k8_all.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(1:8, function(i) data.frame(Cluster = i, word = rownames(subs[[i]]), IndVal = subs[[i]][, i], DCA1 = subs[[i]]$DCA1, DCA2 = subs[[i]]$DCA2))), file.path(run, "data/wordcloud_selected_terms.csv"), row.names = FALSE)
write.csv(data.frame(word = rownames(coords), coords), file.path(run, "data/dca_species.csv"), row.names = FALSE)
ss <- vegan::scores(model, display = "sites", choices = 1:2)
stopifnot(max(abs(ss - as.matrix(read.csv(file.path(run, "data/dca_sites.csv"), row.names = 1)))) < 1e-10)
print(table(g)); print(do.call(rbind, lapply(1:8, function(i) data.frame(Cluster = i, word = rownames(subs[[i]])))))
