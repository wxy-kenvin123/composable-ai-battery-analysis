# Eigenvalues and gradient lengths of the first four DCA axes (Supplementary Note 1).
#
# Reads the DCA model of the clustering run (03_clustering/output/data/cluster_model.rds, made by
# vegan::decorana on the clustering matrix) and writes rerun/data/dca_axes.csv. Stops if a value differs from the
# manuscript (eigenvalues 0.083, 0.072, 0.056, 0.045; gradient lengths 1.40, 1.56, 1.25, 1.15 SD units).
#
# Usage: Rscript scripts/dca_axes.R [output_dir]
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
sec <- dirname(script_dir())
args <- commandArgs(trailingOnly = TRUE)
out_root <- if (length(args)) args[1] else file.path(sec, "rerun")
suppressPackageStartupMessages(library(vegan))
m <- readRDS(file.path(dirname(sec), "03_clustering", "output", "data", "cluster_model.rds"))$dca
print(m)
axes <- data.frame(axis = colnames(m$rproj), eigenvalue = unname(m$evals),
                   gradient_length = unname(apply(m$rproj, 2, function(z) diff(range(z)))))
out <- file.path(out_root, "data")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
write.csv(axes, file.path(out, "dca_axes.csv"), row.names = FALSE)
stopifnot(all.equal(round(axes$eigenvalue, 3), c(0.083, 0.072, 0.056, 0.045)),
          all.equal(round(axes$gradient_length, 2), c(1.40, 1.56, 1.25, 1.15)))
cat("DCA axes as in Supplementary Note 1; written", file.path(out, "dca_axes.csv"), "\n")
