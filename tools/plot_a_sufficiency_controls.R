#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_a_sufficiency_controls.R <input_dir> <fig_dir>")
}

input_dir <- args[[1]]
fig_dir <- args[[2]]
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

safe_num <- function(x) suppressWarnings(as.numeric(x))

subgroup_csv <- file.path(input_dir, "a_relevant_subgroup_monotonicity.csv")
if (file.exists(subgroup_csv)) {
  rows <- read.csv(subgroup_csv, stringsAsFactors = FALSE, check.names = FALSE)
  for (bin_type in unique(rows$group_type[grepl("_bin$", rows$group_type)])) {
    bins <- rows[rows$group_type == bin_type, ]
    if (nrow(bins) <= 0) {
      next
    }
    bins$support_mass_mean <- safe_num(bins$support_mass_mean)
    bins$no_A_minus_A_fused_bce <- safe_num(bins$no_A_minus_A_fused_bce)
    bins$A_fused_minus_no_A_auc <- safe_num(bins$A_fused_minus_no_A_auc)
    bins <- bins[order(bins$support_mass_mean), ]
    out_name <- paste0("a_relevant_monotonicity_", gsub("[^A-Za-z0-9_]+", "_", bin_type), ".png")
    png(file.path(fig_dir, out_name), width = 1500, height = 900, res = 160)
    op <- par(mar = c(8, 5, 4, 5), las = 2)
    x <- seq_len(nrow(bins))
    bp <- barplot(
      bins$no_A_minus_A_fused_bce,
      names.arg = bins$group,
      col = "#4C78A8",
      ylab = "BCE gain of A_fused over no_A",
      main = paste("A-Relevant Subgroup Monotonicity:", bin_type)
    )
    par(new = TRUE)
    plot(
      bp,
      bins$A_fused_minus_no_A_auc,
      type = "b",
      pch = 16,
      axes = FALSE,
      xlab = "",
      ylab = "",
      col = "#E45756"
    )
    axis(4, las = 1)
    mtext("AUC gain", side = 4, line = 3)
    legend("topleft", legend = c("BCE gain", "AUC gain"), fill = c("#4C78A8", NA), border = c("#4C78A8", NA), lty = c(NA, 1), pch = c(NA, 16), col = c("#4C78A8", "#E45756"), bty = "n")
    par(op)
    dev.off()
    if (bin_type == "support_mass_bin") {
      file.copy(file.path(fig_dir, out_name), file.path(fig_dir, "a_relevant_monotonicity.png"), overwrite = TRUE)
    }
  }
}

edge_csv <- file.path(input_dir, "a_edge_deletion_aggregate.csv")
if (file.exists(edge_csv)) {
  rows <- read.csv(edge_csv, stringsAsFactors = FALSE, check.names = FALSE)
  rows$fraction <- safe_num(rows$fraction)
  rows$bce_increase_mean <- safe_num(rows$bce_increase_mean)
  keep_groups <- c("all", "graph_hits_history", "high_support_mass")
  sub <- rows[rows$group %in% keep_groups, ]
  if (nrow(sub) > 0) {
    png(file.path(fig_dir, "a_edge_deletion_necessity.png"), width = 1500, height = 950, res = 160)
    op <- par(mfrow = c(length(unique(sub$group)), 1), mar = c(4, 5, 3, 2))
    for (grp in unique(sub$group)) {
      g <- sub[sub$group == grp, ]
      ylim <- c(min(0, min(g$bce_increase_mean, na.rm = TRUE)), max(g$bce_increase_mean, na.rm = TRUE) * 1.15)
      plot(
        NA,
        xlim = range(g$fraction, na.rm = TRUE),
        ylim = ylim,
        xlab = "Deleted fraction per row",
        ylab = "BCE increase",
        main = paste("Edge deletion:", grp)
      )
      for (mode in unique(g$delete_mode)) {
        m <- g[g$delete_mode == mode, ]
        m <- m[order(m$fraction), ]
        col <- if (mode == "top") "#E45756" else "#4C78A8"
        lines(m$fraction, m$bce_increase_mean, type = "b", pch = 16, col = col)
      }
      legend("topleft", legend = c("top evidence edges", "random support edges"), col = c("#E45756", "#4C78A8"), lty = 1, pch = 16, bty = "n")
    }
    par(op)
    dev.off()
  }
}

retrieval_csv <- file.path(input_dir, "a_transition_retrieval.csv")
ci_csv <- file.path(input_dir, "a_transition_retrieval_bootstrap_ci.csv")
if (file.exists(retrieval_csv)) {
  rows <- read.csv(retrieval_csv, stringsAsFactors = FALSE, check.names = FALSE)
  metric <- if ("hit@10" %in% names(rows)) "hit@10" else tail(grep("^hit@", names(rows), value = TRUE), 1)
  rows[[metric]] <- safe_num(rows[[metric]])
  rows <- rows[order(rows[[metric]], decreasing = TRUE), ]
  ci <- NULL
  if (file.exists(ci_csv)) {
    ci <- read.csv(ci_csv, stringsAsFactors = FALSE, check.names = FALSE)
    ci <- ci[ci$metric == metric, ]
  }
  png(file.path(fig_dir, "a_transition_retrieval_ci.png"), width = 1600, height = 900, res = 160)
  op <- par(mar = c(9, 5, 4, 1), las = 2)
  bp <- barplot(
    rows[[metric]],
    names.arg = rows$variant,
    col = "#54A24B",
    ylab = metric,
    main = "Held-out Concept Transition Retrieval with Bootstrap CI",
    ylim = c(0, max(rows[[metric]], na.rm = TRUE) * 1.20)
  )
  if (!is.null(ci) && nrow(ci) > 0) {
    for (i in seq_len(nrow(rows))) {
      item <- ci[ci$variant == rows$variant[[i]], ]
      if (nrow(item) > 0) {
        arrows(bp[[i]], safe_num(item$ci_low[[1]]), bp[[i]], safe_num(item$ci_high[[1]]), angle = 90, code = 3, length = 0.04, col = "black")
      }
    }
  }
  text(bp, rows[[metric]], labels = sprintf("%.3f", rows[[metric]]), pos = 3, cex = 0.65)
  par(op)
  dev.off()
}
