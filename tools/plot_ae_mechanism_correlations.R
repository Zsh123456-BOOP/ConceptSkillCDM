#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_ae_mechanism_correlations.R <correlation_dir> <out_dir>")
}

cor_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_table <- function(name) {
  path <- file.path(cor_dir, name)
  if (!file.exists(path)) return(data.frame())
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

safe_num <- function(x) {
  y <- suppressWarnings(as.numeric(x))
  y[!is.finite(y)] <- NA
  y
}

corr <- read_table("mechanism_correlations.csv")
bins <- read_table("mechanism_metric_bins.csv")

if (nrow(corr) > 0) {
  corr$spearman <- safe_num(corr$spearman)
  corr <- corr[is.finite(corr$spearman), ]
  if (nrow(corr) > 0) {
    corr$label <- paste(corr$module, corr$metric, corr$target, sep = " / ")
    corr <- corr[order(abs(corr$spearman), decreasing = TRUE), ]
    corr <- corr[seq_len(min(18, nrow(corr))), ]
    png(file.path(out_dir, "mechanism_spearman_top.png"), width = 1900, height = 1200, res = 165)
    op <- par(mar = c(7, 15, 4, 2), las = 1)
    cols <- ifelse(corr$spearman >= 0, "#4C78A8", "#E45756")
    barplot(
      rev(corr$spearman),
      horiz = TRUE,
      names.arg = rev(corr$label),
      col = rev(cols),
      xlim = c(-1, 1),
      xlab = "Spearman correlation",
      main = "CRG/LCRF mechanism metrics vs gain/rescue"
    )
    abline(v = 0, lty = 2, col = "gray40")
    par(op)
    dev.off()
  }
}

plot_metric_bins <- function(module_name, metric_name, file_name, title) {
  sub <- bins[bins$module == module_name & bins$metric == metric_name, ]
  if (nrow(sub) == 0) return(FALSE)
  sub$bin <- as.integer(sub$bin)
  sub$gain_mean <- safe_num(sub$gain_mean)
  sub$rescue_rate <- safe_num(sub$rescue_rate)
  sub <- sub[order(sub$bin), ]
  png(file.path(out_dir, file_name), width = 1600, height = 1050, res = 165)
  op <- par(mar = c(5, 5, 4, 5))
  plot(
    sub$bin,
    sub$gain_mean,
    type = "b",
    pch = 16,
    col = "#4C78A8",
    lwd = 2,
    xlab = paste(metric_name, "quantile bin"),
    ylab = "Mean gain",
    main = title
  )
  abline(h = 0, lty = 2, col = "gray50")
  par(new = TRUE)
  plot(
    sub$bin,
    sub$rescue_rate,
    type = "b",
    pch = 17,
    col = "#F58518",
    lwd = 2,
    axes = FALSE,
    xlab = "",
    ylab = "",
    ylim = c(0, 1)
  )
  axis(4)
  mtext("Rescue rate", side = 4, line = 3)
  legend("topleft", legend = c("mean gain", "rescue rate"), col = c("#4C78A8", "#F58518"), pch = c(16, 17), lwd = 2, bty = "n")
  par(op)
  dev.off()
  TRUE
}

plot_metric_bins("CRG", "a_edge_evidence_mass", "CRG_gain_by_edge_evidence_mass.png", "CRG evidence edge mass vs no_CRG rescue gain")
plot_metric_bins("CRG", "a_top_edge_entropy", "CRG_gain_by_edge_entropy.png", "CRG edge entropy vs no_CRG rescue gain")
plot_metric_bins("LCRF", "query_row_posterior_kl", "LCRF_gain_by_posterior_kl.png", "LCRF posterior movement vs no_LCRF rescue gain")
plot_metric_bins("LCRF", "e_observed_shift_abs", "LCRF_gain_by_observed_shift.png", "LCRF observed student-state shift vs no_LCRF rescue gain")

cat("Mechanism correlation plots written to", out_dir, "\n")
