###############################################################################
# Two-ends disconnect test: are C1 x C6 paper pairs less alike in
# wording than other cross-cluster pairs the same distance apart on the scale
# axis?
#
# Inputs (read only): 03_clustering/output/data/
#   cluster_model.rds (x = log word matrix), dca_sites.csv, assignments_k8.csv
# Outputs: data/bins_<test>.csv, data/summary.csv
#
# Run: "Rscript" scripts/control_test.R
###############################################################################
# Package version: reads 03_clustering/output/data and writes to rerun/control_test/ (or the folder given).
script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
SEC  <- dirname(script_dir())
RUN  <- file.path(dirname(SEC), "03_clustering", "output", "data")
args <- commandArgs(trailingOnly = TRUE)
OUT  <- if (length(args)) args[1] else file.path(SEC, "rerun", "control_test")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
INPUTS    <- file.path(RUN, c("cluster_model.rds", "dca_sites.csv", "assignments_k8.csv"))

BIN       <- 0.05      # bin width on the distance axis
MIN_PAIRS <- 30        # a bin is kept only with >= 30 target and >= 30 control pairs
B         <- 10000     # bootstrap resamples
SEED      <- 20260927
CHUNK     <- 100       # resamples handled per matrix pass
ZERO      <- 1e-12     # a bin difference this small counts as neither lower nor higher
TARGET_COL  <- "#B2182B"
CONTROL_COL <- "grey45"

dir.create(file.path(OUT, "data"), showWarnings = FALSE)

# ---------------------------------------------------------------- inputs
x     <- readRDS(INPUTS[1])$x
sites <- read.csv(INPUTS[2], colClasses = c(id = "character"))
asg   <- read.csv(INPUTS[3], colClasses = c(id = "character"))
ids <- rownames(x)
stopifnot(length(ids) == 755, setequal(ids, sites$id), setequal(ids, asg$id))
d1 <- sites$DCA1[match(ids, sites$id)]
d2 <- sites$DCA2[match(ids, sites$id)]
cl <- asg$Cluster[match(ids, asg$id)]
n  <- length(ids)

# ---------------------------------------------------------------- cross-cluster pairs
xn <- x / sqrt(rowSums(x^2))
S  <- tcrossprod(xn)                                   # cosine similarity, papers x papers
ij <- which(upper.tri(S), arr.ind = TRUE)
ij <- ij[cl[ij[, 1]] != cl[ij[, 2]], ]                 # different clusters only
pi_ <- ij[, 1]; pj <- ij[, 2]
pair_sim   <- S[ij]
is_target  <- (cl[pi_] == 1 & cl[pj] == 6) | (cl[pi_] == 6 & cl[pj] == 1)
one_end    <- xor(cl[pi_] %in% c(1, 6), cl[pj] %in% c(1, 6))   # exactly one paper in C1 or C6
dist_dca1  <- abs(d1[pi_] - d1[pj])
dist_2d    <- sqrt((d1[pi_] - d1[pj])^2 + (d2[pi_] - d2[pj])^2)
stopifnot(sum(is_target) == 153 * 59)

TESTS <- list(
  primary        = list(dist = dist_dca1, control = !is_target,
                        label = "Primary: same distance on DCA1"),
  sens1_2d       = list(dist = dist_2d,   control = !is_target,
                        label = "Check 1: same distance on DCA1 and DCA2"),
  sens2_partners = list(dist = dist_dca1, control = one_end,
                        label = "Check 2: C1 or C6 paired with the other six"))

# one test = the pairs it uses, their kept distance bin and whether they are targets
setup <- function(t) {
  bin <- floor(t$dist / BIN) + 1L
  nb  <- max(bin[is_target | t$control])
  nT  <- tabulate(bin[is_target], nb)
  nC  <- tabulate(bin[t$control], nb)
  kept <- which(nT >= MIN_PAIRS & nC >= MIN_PAIRS)
  use <- (is_target | t$control) & bin %in% kept
  pos <- match(bin[use], kept)
  list(i = pi_[use], j = pj[use], sim = pair_sim[use], target = is_target[use], pos = pos,
       key = 2L * (pos - 1L) + ifelse(is_target[use], 1L, 2L), K = length(kept), kept = kept,
       counts = data.frame(bin = seq_len(nb), n_target = nT, n_control = nC, kept = seq_len(nb) %in% kept))
}

# per-bin weighted sums for R sets of pair weights (W: pairs x R)
bin_means <- function(A, W, sim) {
  sw  <- rowsum(W, A$key, reorder = TRUE)
  sws <- rowsum(W * sim, A$key, reorder = TRUE)
  stopifnot(nrow(sw) == 2 * A$K)
  t_ <- seq(1, 2 * A$K, 2); c_ <- t_ + 1
  list(wT = sw[t_, , drop = FALSE], wC = sw[c_, , drop = FALSE],
       mT = sws[t_, , drop = FALSE] / sw[t_, , drop = FALSE],
       mC = sws[c_, , drop = FALSE] / sw[c_, , drop = FALSE])
}

# totals per column: bin differences weighted by target weight, skipping bins with a zero weight
totals <- function(m) {
  ok <- m$wT > 0 & m$wC > 0
  w  <- ifelse(ok, m$wT, 0)
  dif <- ifelse(ok, m$mT - m$mC, NA_real_)
  tot  <- colSums(w * ifelse(ok, dif, 0)) / colSums(w)
  ctrl <- colSums(w * ifelse(ok, m$mC, 0)) / colSums(w)
  list(diff_bins = dif, total = tot, rel = tot / ctrl, ctrl = ctrl, skipped = sum(!ok))
}

point <- function(A, sim) {
  m <- bin_means(A, matrix(1, length(A$i), 1), sim)
  c(totals(m), list(mT = m$mT[, 1], mC = m$mC[, 1]))
}

# paper multiplicities for every resample: drawn with replacement within each cluster
set.seed(SEED)
M <- matrix(0L, n, B)
groups <- split(seq_len(n), cl)
for (b in seq_len(B)) for (g in groups) M[g, b] <- tabulate(sample.int(length(g), length(g), replace = TRUE), length(g))
stopifnot(all(colSums(M) == n))

boot <- function(A, sim) {
  tot <- rel <- numeric(B); dif <- matrix(NA_real_, A$K, B); skipped <- 0
  for (s in seq(1, B, by = CHUNK)) {
    cols <- s:min(B, s + CHUNK - 1)
    W <- M[A$i, cols, drop = FALSE] * M[A$j, cols, drop = FALSE]   # pair weight = product of multiplicities
    r <- totals(bin_means(A, W, sim))
    tot[cols] <- r$total; rel[cols] <- r$rel; dif[, cols] <- r$diff_bins
    skipped <- skipped + r$skipped
  }
  list(total = tot, rel = rel, diff_bins = dif, skipped = skipped)
}

verdict <- function(lo, hi, share_lower, share_higher) {
  if (hi < 0 && share_lower >= 0.8) "support" else if (lo > 0 && share_higher >= 0.8) "against" else "inconclusive"
}

# ---------------------------------------------------------------- machinery checks, before the result
stopifnot(verdict(-0.2, -0.1, 0.9, 0.1) == "support", verdict(-0.2, -0.1, 0.7, 0.3) == "inconclusive",
          verdict(0.1, 0.2, 0.1, 0.85) == "against", verdict(-0.1, 0.1, 1, 0) == "inconclusive")
A0 <- setup(TESTS$primary)
base  <- point(A0, A0$sim)
shift <- point(A0, A0$sim - 0.05 * A0$target)
stopifnot(abs((shift$total - base$total) + 0.05) < 1e-12)
sim0 <- A0$sim
sim0[A0$target] <- base$mC[A0$pos[A0$target]]          # targets set to their bin's control mean
zero <- point(A0, sim0)
zb   <- boot(A0, sim0)
zv   <- verdict(quantile(zb$total, 0.025), quantile(zb$total, 0.975),
                mean(zero$diff_bins < -ZERO), mean(zero$diff_bins > ZERO))
stopifnot(abs(zero$total) < 1e-12, zv == "inconclusive")
cat(sprintf("machinery: shift -0.05 -> %+.12f; zero effect -> %.1e, verdict %s\n",
            shift$total - base$total, zero$total, zv))
rm(base, shift, sim0, zero, zb)

# ---------------------------------------------------------------- the three tests
summ <- list(); draws <- data.frame(rep = seq_len(B)); bins_all <- list()
for (nm in names(TESTS)) {
  A  <- setup(TESTS[[nm]])
  p  <- point(A, A$sim)
  bt <- boot(A, A$sim)
  lower  <- mean(p$diff_bins < -ZERO); higher <- mean(p$diff_bins > ZERO)
  ci  <- unname(quantile(bt$total, c(0.025, 0.975)))
  rci <- unname(quantile(bt$rel, c(0.025, 0.975)))
  cnt <- A$counts
  cnt$bin_lo <- (cnt$bin - 1) * BIN; cnt$bin_hi <- cnt$bin * BIN
  cnt$mean_target <- cnt$mean_control <- cnt$diff <- cnt$diff_lo <- cnt$diff_hi <- NA_real_
  k <- A$kept
  cnt$mean_target[k] <- p$mT; cnt$mean_control[k] <- p$mC; cnt$diff[k] <- p$diff_bins[, 1]
  cnt$diff_lo[k] <- apply(bt$diff_bins, 1, quantile, 0.025, na.rm = TRUE)
  cnt$diff_hi[k] <- apply(bt$diff_bins, 1, quantile, 0.975, na.rm = TRUE)
  cnt <- cnt[cnt$n_target > 0 | cnt$n_control > 0,
             c("bin", "bin_lo", "bin_hi", "n_target", "n_control", "kept", "mean_target", "mean_control", "diff", "diff_lo", "diff_hi")]
  write.csv(cnt, file.path(OUT, "data", sprintf("bins_%s.csv", nm)), row.names = FALSE)
  bins_all[[nm]] <- cnt
  summ[[nm]] <- data.frame(
    test = nm, label = TESTS[[nm]]$label, bins_kept = A$K,
    bins_with_targets = sum(cnt$n_target > 0), target_pairs_kept = sum(A$target),
    target_pairs_dropped_share = 1 - sum(A$target) / sum(is_target), control_pairs_kept = sum(!A$target),
    mean_target_weighted = p$ctrl + p$total, mean_control_weighted = p$ctrl,
    diff = p$total, diff_lo = ci[1], diff_hi = ci[2], rel_diff = p$rel, rel_lo = rci[1], rel_hi = rci[2],
    share_bins_lower = lower, share_bins_higher = higher, bin_resamples_skipped = bt$skipped,
    verdict = verdict(ci[1], ci[2], lower, higher))
  draws[[nm]] <- bt$total
}
summ <- do.call(rbind, summ)
write.csv(summ, file.path(OUT, "data/summary.csv"), row.names = FALSE)

print(summ[, c("test", "bins_kept", "target_pairs_dropped_share", "mean_target_weighted", "mean_control_weighted",
               "diff", "diff_lo", "diff_hi", "rel_diff", "share_bins_lower", "bin_resamples_skipped", "verdict")], row.names = FALSE)
