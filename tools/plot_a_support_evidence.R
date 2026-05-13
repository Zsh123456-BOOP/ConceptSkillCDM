#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_a_support_evidence.R <a_support_output_dir> <fig_dir>")
}

input_dir <- args[[1]]
fig_dir <- args[[2]]
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

transition_csv <- file.path(input_dir, "a_transition_retrieval.csv")
if (file.exists(transition_csv)) {
  rows <- read.csv(transition_csv, stringsAsFactors = FALSE, check.names = FALSE)
  metric <- if ("hit@10" %in% names(rows)) "hit@10" else tail(grep("^hit@", names(rows), value = TRUE), 1)
  rows[[metric]] <- as.numeric(rows[[metric]])
  rows <- rows[order(rows[[metric]], decreasing = TRUE), ]
  png(file.path(fig_dir, "a_transition_retrieval.png"), width = 1500, height = 900, res = 160)
  op <- par(mar = c(9, 5, 4, 1), las = 2)
  barplot(
    rows[[metric]],
    names.arg = rows$variant,
    col = "#4C78A8",
    ylab = metric,
    main = "Held-out Concept Transition Retrieval",
    ylim = c(0, max(rows[[metric]], na.rm = TRUE) * 1.18)
  )
  text(
    x = seq_along(rows[[metric]]) * 1.2 - 0.5,
    y = rows[[metric]],
    labels = sprintf("%.3f", rows[[metric]]),
    pos = 3,
    cex = 0.65
  )
  par(op)
  dev.off()
}

subgroup_csv <- file.path(input_dir, "a_subgroup_auc.csv")
if (file.exists(subgroup_csv)) {
  rows <- read.csv(subgroup_csv, stringsAsFactors = FALSE, check.names = FALSE)
  gain_cols <- grep("_minus_no_A_auc$", names(rows), value = TRUE)
  if (length(gain_cols) > 0 && nrow(rows) > 0) {
    mat <- as.matrix(rows[, gain_cols, drop = FALSE])
    mat <- apply(mat, 2, as.numeric)
    rownames(mat) <- rows$group
    png(file.path(fig_dir, "a_relevant_subgroup_gain.png"), width = 1700, height = 950, res = 160)
    op <- par(mar = c(9, 5, 4, 2), las = 2)
    barplot(
      t(mat),
      beside = TRUE,
      col = grDevices::rainbow(ncol(mat), s = 0.7, v = 0.75),
      ylab = "AUC Gain vs no_A",
      main = "A-Relevant Subgroup Gains",
      legend.text = gsub("_minus_no_A_auc$", "", gain_cols),
      args.legend = list(x = "topright", cex = 0.7, bty = "n")
    )
    par(op)
    dev.off()
  }
}
