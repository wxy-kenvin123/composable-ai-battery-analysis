###############################################################################
# Scale-gap check, part 1: where the eight clusters sit along
# DCA1, whether the axis has an empty stretch between neighbouring clusters,
# and how much clusters mix in the word space the Ward clustering used.
#
# Inputs (read only): 03_clustering/output/data/
#   dca_sites.csv, assignments_k8.csv, cluster_model.rds (x = log word matrix, h = Ward tree)
# Outputs: data/cluster_positions.csv, data/boundary_gaps.csv,
#   data/silhouette_mixing.csv, data/pair_mixing.csv, data/cluster_distances.csv,
#   svg/dca1_by_cluster.svg, qa/dca1_by_cluster.png
#
# Run: "Rscript" scripts/position.R
###############################################################################
suppressPackageStartupMessages({
  library(cluster)
  library(ggplot2)
  library(patchwork)
})

# Package version: reads 03_clustering/output/data and writes to rerun/ (or the folder given).
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
SEC  <- dirname(script_dir())
RUN  <- file.path(dirname(SEC), "03_clustering", "output", "data")
args <- commandArgs(trailingOnly = TRUE)
OUT  <- if (length(args)) args[1] else file.path(SEC, "rerun")
dir.create(file.path(OUT, "data"), recursive = TRUE, showWarnings = FALSE)

SCALE_ORDER <- c(1, 8, 2, 3, 5, 7, 4, 6)          # atoms -> cells, fixed in panel g
LEFT  <- c(1, 8, 2, 3)                            # top split of the Ward tree (panel b)
RIGHT <- c(4, 5, 6, 7)
K_NN  <- 10
NAMES <- c("1" = "C1 Atomistic dynamics & transport", "2" = "C2 Molecular & polymer properties",
           "3" = "C3 Solid-material properties", "4" = "C4 Microstructure & manufacturing",
           "5" = "C5 Materials design & optimization", "6" = "C6 Battery state & lifetime",
           "7" = "C7 Electrode performance prediction", "8" = "C8 Structural stability & mechanisms")
COLS  <- c("1" = "#8DB2D7", "2" = "#F9DA8E", "3" = "#79BED5", "4" = "#F8B62D",
           "5" = "#274C9F", "6" = "#E9504F", "7" = "#B7DFD9", "8" = "#FBDED7")

# ---------------------------------------------------------------- inputs
sites <- read.csv(file.path(RUN, "dca_sites.csv"), colClasses = c(id = "character"))
asg   <- read.csv(file.path(RUN, "assignments_k8.csv"), colClasses = c(id = "character"))
model <- readRDS(file.path(RUN, "cluster_model.rds"))
x <- model$x
d <- merge(sites, asg, by = "id")
stopifnot(nrow(d) == 755, nrow(x) == 755, setequal(rownames(x), d$id))
d <- d[match(rownames(x), d$id), ]

# the Ward tree cut in two must give exactly LEFT vs RIGHT
k2 <- cutree(as.hclust(model$h), 2)
split_tab <- table(cluster = d$Cluster, top = k2)
stopifnot(all(rowSums(split_tab > 0) == 1))       # each cluster falls wholly on one side
side <- apply(split_tab, 1, function(r) which(r > 0))
stopifnot(length(unique(side[as.character(LEFT)])) == 1,
          length(unique(side[as.character(RIGHT)])) == 1,
          side[as.character(LEFT)][1] != side[as.character(RIGHT)][1])

# ---------------------------------------------------------------- positions
pos <- do.call(rbind, lapply(1:8, function(k) {
  v <- d$DCA1[d$Cluster == k]
  data.frame(cluster = k, n = length(v), mean_dca1 = mean(v), median_dca1 = median(v),
             q25 = unname(quantile(v, 0.25)), q75 = unname(quantile(v, 0.75)),
             mean_dca2 = mean(d$DCA2[d$Cluster == k]),
             scale_rank = match(k, SCALE_ORDER))
}))
pos$dca1_rank <- rank(pos$median_dca1)
rho <- cor(pos$scale_rank, pos$median_dca1, method = "spearman")
write.csv(pos, file.path(OUT, "data/cluster_positions.csv"), row.names = FALSE)

# ---------------------------------------------------------------- empty stretches along DCA1
dens <- density(d$DCA1, n = 4096)
inside <- dens$x >= min(d$DCA1) & dens$x <= max(d$DCA1)
mean_dens <- mean(dens$y[inside])
ord <- pos$cluster[order(pos$median_dca1)]
med <- setNames(pos$median_dca1, pos$cluster)
gaps <- do.call(rbind, lapply(1:7, function(i) {
  a <- ord[i]; b <- ord[i + 1]
  w <- dens$x >= med[as.character(a)] & dens$x <= med[as.character(b)]
  j <- which(w)[which.min(dens$y[w])]
  mid <- (med[as.character(a)] + med[as.character(b)]) / 2
  k <- which.min(abs(dens$x - mid))
  data.frame(boundary = sprintf("C%d|C%d", a, b), left = a, right = b,
             trough_at = dens$x[j], trough_density = dens$y[j], ratio_to_mean = dens$y[j] / mean_dens,
             interior_trough = j != min(which(w)) && j != max(which(w)),   # FALSE: lowest point is a window edge, i.e. no valley
             midpoint = unname(mid), ratio_at_midpoint = dens$y[k] / mean_dens)
}))
gaps$rank_lowest <- rank(gaps$ratio_to_mean)
gaps$is_A <- gaps$left == 6 | gaps$right == 6
left_meds <- med[as.character(LEFT)]; right_meds <- med[as.character(RIGHT)]
b_contiguous <- max(left_meds) < min(right_meds) || max(right_meds) < min(left_meds)
write.csv(data.frame(spearman_scale_order_vs_median_dca1 = rho, b_split_contiguous_on_dca1 = b_contiguous),
          file.path(OUT, "data/axis_order.csv"), row.names = FALSE)
gaps$is_B <- b_contiguous & ((gaps$left %in% LEFT & gaps$right %in% RIGHT) | (gaps$left %in% RIGHT & gaps$right %in% LEFT))

# ---------------------------------------------------------------- mixing in the full word space
D <- as.matrix(dist(x))
sil <- silhouette(d$Cluster, as.dist(D))
nn <- t(apply(D, 1, function(r) order(r)[2:(K_NN + 1)]))   # first column is the paper itself
nn_cl <- matrix(d$Cluster[nn], nrow = nrow(nn))
mixing <- do.call(rbind, lapply(1:8, function(k) {
  rows <- d$Cluster == k
  data.frame(cluster = k, n = sum(rows), mean_silhouette = mean(sil[rows, "sil_width"]),
             share_neighbours_other = mean(nn_cl[rows, ] != k),
             share_neighbours_other_side = mean(nn_cl[rows, ] %in% (if (k %in% LEFT) RIGHT else LEFT)))
}))
write.csv(mixing, file.path(OUT, "data/silhouette_mixing.csv"), row.names = FALSE)

pairs <- t(combn(1:8, 2))
pair_mix <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(p) {
  a <- pairs[p, 1]; b <- pairs[p, 2]
  ra <- d$Cluster == a; rb <- d$Cluster == b
  links <- sum(nn_cl[ra, ] == b) + sum(nn_cl[rb, ] == a)
  data.frame(pair = sprintf("C%d-C%d", a, b), a = a, b = b,
             knn_links = links, knn_share = links / (K_NN * (sum(ra) + sum(rb))),
             mean_distance = mean(D[ra, rb]),
             adjacent_on_dca1 = abs(match(a, ord) - match(b, ord)) == 1,
             crosses_A = (a == 6) != (b == 6),
             crosses_B = (a %in% LEFT) != (b %in% LEFT))
}))
write.csv(pair_mix, file.path(OUT, "data/pair_mixing.csv"), row.names = FALSE)
gaps <- merge(gaps, pair_mix[, c("a", "b", "knn_share")], by.x = c("left", "right"), by.y = c("a", "b"), all.x = TRUE)
miss <- is.na(gaps$knn_share)
if (any(miss)) gaps$knn_share[miss] <- mapply(function(l, r) pair_mix$knn_share[pair_mix$a == min(l, r) & pair_mix$b == max(l, r)],
                                              gaps$left[miss], gaps$right[miss])
gaps$knn_share_vs_median <- gaps$knn_share / median(gaps$knn_share)
gaps <- gaps[order(gaps$trough_at), ]
write.csv(gaps, file.path(OUT, "data/boundary_gaps.csv"), row.names = FALSE)

within <- sapply(1:8, function(k) mean(D[d$Cluster == k, d$Cluster == k][upper.tri(D[d$Cluster == k, d$Cluster == k])]))
dist_mat <- outer(1:8, 1:8, Vectorize(function(a, b) if (a == b) within[a] else mean(D[d$Cluster == a, d$Cluster == b])))
dimnames(dist_mat) <- list(paste0("C", 1:8), paste0("C", 1:8))
write.csv(dist_mat, file.path(OUT, "data/cluster_distances.csv"))
c6_nearest <- names(sort(dist_mat["C6", -6]))[1]

# ---------------------------------------------------------------- diagnostic figure
d$lab <- factor(NAMES[as.character(d$Cluster)], levels = rev(NAMES[as.character(SCALE_ORDER)]))
xr <- range(d$DCA1)
mark <- rbind(
  data.frame(what = paste("A", gaps$boundary[gaps$is_A]), at = gaps$midpoint[gaps$is_A]),
  if (any(gaps$is_B)) data.frame(what = paste("B", gaps$boundary[gaps$is_B]), at = gaps$midpoint[gaps$is_B]) else NULL)
dens_df <- data.frame(x = dens$x[inside], y = dens$y[inside])
p_top <- ggplot(dens_df, aes(x, y)) +
  geom_area(fill = "grey85") + geom_line(colour = "grey30", linewidth = 0.5) +
  geom_hline(yintercept = mean_dens / 2, linetype = "dotted", colour = "grey40") +
  geom_vline(data = mark, aes(xintercept = at), linetype = "dashed", colour = "#B2182B") +
  geom_label(data = mark, aes(x = at, y = max(dens_df$y) * 1.12, label = what), colour = "#B2182B",
             family = "Arial", size = 3.2, linewidth = 0, fill = "white", label.padding = unit(0.1, "lines")) +
  annotate("text", x = xr[2], y = mean_dens / 2, label = "half of\nmean", hjust = -0.1, vjust = 0.5,
           family = "Arial", size = 2.8, colour = "grey40") +
  scale_x_continuous(limits = xr) +
  scale_y_continuous(limits = c(0, max(dens_df$y) * 1.2), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Density of\narticles") +
  theme_classic(base_family = "Arial", base_size = 10) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), plot.margin = margin(4, 36, 2, 4))
p_bot <- ggplot(d, aes(DCA1, lab)) +
  geom_jitter(aes(colour = factor(Cluster)), height = 0.18, width = 0, size = 0.9, alpha = 0.7) +
  geom_boxplot(fill = NA, outlier.shape = NA, width = 0.55, linewidth = 0.4) +
  geom_vline(data = mark, aes(xintercept = at), linetype = "dashed", colour = "#B2182B") +
  scale_colour_manual(values = COLS, guide = "none") +
  scale_x_continuous(limits = xr) +
  labs(x = "DCA1 (atoms  \u2192  cells)", y = NULL) +
  theme_classic(base_family = "Arial", base_size = 10)
fig <- p_top / p_bot + plot_layout(heights = c(1, 2.6))
# plot suppressed
# ggsave(file.path(OUT, "svg/dca1_by_cluster.svg"), fig, width = 7.2, height = 4.8, device = svglite::svglite)
# rsvg::rsvg_png(file.path(OUT, "svg/dca1_by_cluster.svg"), file.path(OUT, "qa/dca1_by_cluster.png"), width = 2000)

# ---------------------------------------------------------------- console summary
cat(sprintf("
cluster DCA1 means: %s\n", paste(sprintf("C%d %+.2f", pos$cluster, pos$mean_dca1), collapse = ", ")))
cat(sprintf("Spearman(scale order, median DCA1) = %.2f\n", rho))
cat(sprintf("B is a contiguous split along DCA1: %s\n", b_contiguous))
print(gaps[, c("boundary", "ratio_to_mean", "interior_trough", "ratio_at_midpoint", "rank_lowest", "knn_share_vs_median", "is_A", "is_B")], row.names = FALSE)
print(mixing, row.names = FALSE)
cat(sprintf("C6 is nearest (mean distance) to %s\n", c6_nearest))
