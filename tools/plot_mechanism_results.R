#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_mechanism_results.R <mechanism_results.csv> <out_dir>")
}

result_csv <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

rows <- read.csv(result_csv, stringsAsFactors = FALSE, check.names = FALSE)
rows <- rows[rows$status %in% c("ok", "metrics_ok") & !is.na(rows$test_auc), ]
if (nrow(rows) == 0) {
  stop("No successful rows with test_auc found.")
}

rows$test_auc <- as.numeric(rows$test_auc)
rows$phase_dataset <- paste(rows$phase, rows$dataset, sep = " / ")

variant_order <- c(
  "full", "no_A", "no_E", "A_item_only", "A_seq_only",
  "A_uniform", "A_self_only",
  "full_fair", "no_A_fair", "no_E_fair",
  "A_fused_neutralE", "A_item_neutralE", "A_seq_neutralE",
  "A_uniform_neutralE", "A_self_neutralE",
  "E_prior_only", "E_frozen_alpha", "E_full_fair",
  "E_global_posterior", "E_posterior_only", "E_query_only"
)
rows$variant <- factor(rows$variant, levels = variant_order)

png(file.path(out_dir, "auc_by_variant.png"), width = 1800, height = 1100, res = 160)
op <- par(mar = c(12, 5, 4, 2), las = 2)
dataset_levels <- unique(rows$phase_dataset)
plot(
  NA,
  xlim = c(0.5, length(dataset_levels) + 0.5),
  ylim = range(rows$test_auc, na.rm = TRUE),
  xaxt = "n",
  xlab = "",
  ylab = "Test AUC",
  main = "Mechanism Experiments: AUC by Variant"
)
axis(1, at = seq_along(dataset_levels), labels = dataset_levels, cex.axis = 0.65)
cols <- grDevices::rainbow(length(variant_order), s = 0.7, v = 0.75)
names(cols) <- variant_order
for (i in seq_along(dataset_levels)) {
  sub <- rows[rows$phase_dataset == dataset_levels[[i]], ]
  offsets <- seq(-0.32, 0.32, length.out = length(variant_order))
  for (j in seq_along(variant_order)) {
    r <- sub[as.character(sub$variant) == variant_order[[j]], ]
    if (nrow(r) > 0) {
      points(i + offsets[[j]], r$test_auc[[which.max(r$test_auc)]], pch = 16, col = cols[[j]], cex = 0.9)
    }
  }
}
legend("bottomleft", legend = variant_order, col = cols, pch = 16, cex = 0.65, ncol = 3, bty = "n")
par(op)
dev.off()

summary_csv <- file.path(dirname(result_csv), "mechanism_effectiveness_summary.csv")
if (file.exists(summary_csv)) {
  s <- read.csv(summary_csv, stringsAsFactors = FALSE, check.names = FALSE)
  metric_cols <- c("drop_no_A", "drop_no_E", "evidence_gain_vs_uniform", "seq_minus_item")
  for (m in metric_cols) {
    if (m %in% names(s)) {
      s[[m]] <- as.numeric(s[[m]])
    }
  }
  s$phase_dataset <- paste(s$phase, s$dataset, sep = " / ")

  png(file.path(out_dir, "mechanism_drops.png"), width = 1800, height = 1100, res = 160)
  op <- par(mar = c(12, 5, 4, 2), las = 2)
  mat <- t(as.matrix(s[, c("drop_no_A", "drop_no_E", "evidence_gain_vs_uniform")]))
  colnames(mat) <- s$phase_dataset
  barplot(
    mat,
    beside = TRUE,
    col = c("#4C78A8", "#F58518", "#54A24B"),
    ylab = "AUC Difference",
    main = "Mechanism Effect Sizes",
    cex.names = 0.65
  )
  abline(h = 0, lty = 2, col = "gray40")
  legend(
    "topright",
    legend = c("full - no_A", "full - no_E", "full - A_uniform"),
    fill = c("#4C78A8", "#F58518", "#54A24B"),
    cex = 0.75,
    bty = "n"
  )
  par(op)
  dev.off()

  if ("query_row_posterior_kl" %in% names(s) && "drop_no_E" %in% names(s)) {
    s$query_row_posterior_kl <- as.numeric(s$query_row_posterior_kl)
    png(file.path(out_dir, "e_drop_vs_posterior_kl.png"), width = 1400, height = 1000, res = 160)
    op <- par(mar = c(5, 5, 4, 2))
    plot(
      s$query_row_posterior_kl,
      s$drop_no_E,
      pch = 16,
      col = "#B279A2",
      xlab = "E posterior KL",
      ylab = "full - no_E AUC",
      main = "E Effect vs Posterior Movement"
    )
    text(s$query_row_posterior_kl, s$drop_no_E, labels = s$dataset, pos = 3, cex = 0.7)
    abline(h = 0, lty = 2, col = "gray40")
    par(op)
    dev.off()
  }
}

cat("R plots written to", out_dir, "\n")
