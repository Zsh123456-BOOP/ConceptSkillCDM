#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_a_support_corruption.R <input_dir> <fig_dir>")
}

input_dir <- args[[1]]
fig_dir <- args[[2]]
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

safe_num <- function(x) suppressWarnings(as.numeric(x))

csv <- file.path(input_dir, "a_support_corruption_aggregate.csv")
if (!file.exists(csv)) {
  stop(paste("missing", csv))
}

rows <- read.csv(csv, stringsAsFactors = FALSE, check.names = FALSE)
rows$fraction <- safe_num(rows$fraction)
rows$auc_drop_mean <- safe_num(rows$auc_drop_mean)
rows$bce_increase_mean <- safe_num(rows$bce_increase_mean)

keep_groups <- c("all", "graph_hits_history", "high_support_mass", "query_seq_top5_q4_high")
sub <- rows[rows$group %in% keep_groups, ]
sub <- sub[order(sub$group, sub$fraction), ]

png(file.path(fig_dir, "a_support_corruption_counterfactual.png"), width = 1600, height = 1000, res = 160)
op <- par(mfrow = c(2, 1), mar = c(4, 5, 3, 2))

ylim_auc <- c(min(0, min(sub$auc_drop_mean, na.rm = TRUE)), max(sub$auc_drop_mean, na.rm = TRUE) * 1.15)
plot(NA, xlim = range(sub$fraction, na.rm = TRUE), ylim = ylim_auc, xlab = "Corrupted support fraction", ylab = "AUC drop", main = "A Support Corruption Counterfactual")
cols <- c(all = "#4C78A8", graph_hits_history = "#54A24B", high_support_mass = "#E45756", query_seq_top5_q4_high = "#F58518")
for (grp in keep_groups) {
  g <- sub[sub$group == grp, ]
  if (nrow(g) <= 0) next
  lines(g$fraction, g$auc_drop_mean, type = "b", pch = 16, col = cols[[grp]])
}
legend("topleft", legend = keep_groups, col = unname(cols[keep_groups]), lty = 1, pch = 16, bty = "n", cex = 0.8)

ylim_bce <- c(min(0, min(sub$bce_increase_mean, na.rm = TRUE)), max(sub$bce_increase_mean, na.rm = TRUE) * 1.15)
plot(NA, xlim = range(sub$fraction, na.rm = TRUE), ylim = ylim_bce, xlab = "Corrupted support fraction", ylab = "BCE increase", main = "Prediction Loss Under Degree-Matched Support Corruption")
for (grp in keep_groups) {
  g <- sub[sub$group == grp, ]
  if (nrow(g) <= 0) next
  lines(g$fraction, g$bce_increase_mean, type = "b", pch = 16, col = cols[[grp]])
}
legend("topleft", legend = keep_groups, col = unname(cols[keep_groups]), lty = 1, pch = 16, bty = "n", cex = 0.8)

par(op)
dev.off()
