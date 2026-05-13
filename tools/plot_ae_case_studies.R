#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript tools/plot_ae_case_studies.R <case_dir> <out_dir>")
}

case_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_table <- function(name) {
  path <- file.path(case_dir, name)
  if (!file.exists(path)) {
    return(data.frame())
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

safe_num <- function(x) {
  y <- suppressWarnings(as.numeric(x))
  y[!is.finite(y)] <- 0
  y
}

clip_scale <- function(x, lo = NULL, hi = NULL) {
  x <- safe_num(x)
  if (is.null(lo)) lo <- min(x, na.rm = TRUE)
  if (is.null(hi)) hi <- max(x, na.rm = TRUE)
  if (!is.finite(lo) || !is.finite(hi) || abs(hi - lo) < 1e-12) {
    return(rep(0.5, length(x)))
  }
  pmin(1, pmax(0, (x - lo) / (hi - lo)))
}

draw_matrix_heatmap <- function(mat, row_labels, col_labels, main, file, palette, zlim = NULL, label_mat = NULL, label_digits = 2) {
  if (length(mat) == 0 || nrow(mat) == 0 || ncol(mat) == 0) return(FALSE)
  png(file, width = 1800, height = 1400, res = 170)
  op <- par(mar = c(9, 10, 5, 8), xpd = NA)
  if (is.null(zlim)) {
    zlim <- range(mat, finite = TRUE)
  }
  if (!all(is.finite(zlim)) || abs(diff(zlim)) < 1e-12) {
    zlim <- c(0, 1)
  }
  image(
    x = seq_len(ncol(mat)),
    y = seq_len(nrow(mat)),
    z = t(mat[nrow(mat):1, , drop = FALSE]),
    col = palette,
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = main,
    zlim = zlim
  )
  axis(1, at = seq_len(ncol(mat)), labels = col_labels, las = 2, cex.axis = 0.75)
  axis(2, at = seq_len(nrow(mat)), labels = rev(row_labels), las = 2, cex.axis = 0.75)
  box()
  if (nrow(mat) <= 12 && ncol(mat) <= 12) {
    if (is.null(label_mat)) label_mat <- mat
    for (i in seq_len(nrow(mat))) {
      for (j in seq_len(ncol(mat))) {
        text(j, nrow(mat) - i + 1, sprintf(paste0("%.", label_digits, "f"), label_mat[i, j]), cex = 0.55, col = "black")
      }
    }
  }
  par(op)
  dev.off()
  TRUE
}

a_matrix <- read_table("a_case_matrix.csv")
a_cases <- read_table("a_cases.csv")
if (nrow(a_matrix) > 0) {
  blue_pal <- colorRampPalette(c("#f7fbff", "#9ecae1", "#08519c"))(100)
  for (case_id in unique(a_matrix$case_id)) {
    sub <- a_matrix[a_matrix$case_id == case_id, ]
    sub <- sub[order(sub$row_order, sub$col_order), ]
    rows <- unique(sub[order(sub$row_order), c("row_order", "row_concept")]$row_concept)
    cols <- unique(sub[order(sub$col_order), c("col_order", "col_concept")]$col_concept)
    mat <- matrix(0, nrow = length(rows), ncol = length(cols), dimnames = list(rows, cols))
    for (i in seq_len(nrow(sub))) {
      mat[sub$row_concept[[i]], sub$col_concept[[i]]] <- safe_num(sub$a_weight[[i]])
    }
    meta <- a_cases[a_cases$case_id == case_id, ]
    title <- if (nrow(meta) > 0) {
      sprintf(
        "A Global Roadmap: %s | y=%s full=%.3f no_A=%.3f gain=%.3f",
        case_id,
        meta$label[[1]],
        safe_num(meta$full_prob[[1]]),
        safe_num(meta$no_A_prob[[1]]),
        safe_num(meta$a_gain[[1]])
      )
    } else {
      paste("A Global Roadmap:", case_id)
    }
    draw_matrix_heatmap(
      mat,
      rownames(mat),
      colnames(mat),
      title,
      file.path(out_dir, paste0("A_roadmap_heatmap_", case_id, ".png")),
      blue_pal,
      zlim = c(0, max(mat, 1e-6))
    )
  }
}

a_edges <- read_table("a_case_edges.csv")
if (nrow(a_edges) > 0) {
  png(file.path(out_dir, "A_top_edges_by_case.png"), width = 1800, height = 1200, res = 170)
  op <- par(mar = c(8, 6, 4, 2), mfrow = c(max(1, ceiling(length(unique(a_edges$case_id)) / 2)), 2))
  for (case_id in unique(a_edges$case_id)) {
    sub <- a_edges[a_edges$case_id == case_id, ]
    sub <- sub[order(-safe_num(sub$a_weight)), ][seq_len(min(8, nrow(sub))), ]
    labels <- paste(sub$query_concept, "->", sub$support_concept, "\n", sub$source)
    barplot(
      safe_num(sub$a_weight),
      names.arg = labels,
      las = 2,
      col = "#4C78A8",
      ylab = "A edge weight",
      main = case_id,
      cex.names = 0.65
    )
  }
  par(op)
  dev.off()
}

e_edges <- read_table("e_case_edges.csv")
e_cases <- read_table("e_cases.csv")
if (nrow(e_edges) > 0) {
  if (nrow(e_cases) > 0 && all(c("E_shuffle_prob", "E_mean_prob") %in% names(e_cases))) {
    png(file.path(out_dir, "E_counterfactual_probability_comparison.png"), width = 1900, height = 1100, res = 170)
    op <- par(mar = c(10, 5, 4, 2))
    labels <- paste(e_cases$case_id, "y=", e_cases$label)
    values <- rbind(
      safe_num(e_cases$full_prob),
      safe_num(e_cases$no_E_prob),
      safe_num(e_cases$E_shuffle_prob),
      safe_num(e_cases$E_mean_prob)
    )
    barplot(
      values,
      beside = TRUE,
      names.arg = labels,
      las = 2,
      col = c("#4C78A8", "#54A24B", "#E45756", "#B279A2"),
      ylim = c(0, 1),
      ylab = "Predicted probability",
      main = "E Counterfactual: real student state vs no_E / shuffled / mean"
    )
    abline(h = 0.5, lty = 2, col = "gray40")
    legend("topright", legend = c("full actual E", "no_E", "E shuffled", "E mean"), fill = c("#4C78A8", "#54A24B", "#E45756", "#B279A2"), bty = "n")
    par(op)
    dev.off()
  }

  for (case_id in unique(e_edges$case_id)) {
    sub <- e_edges[e_edges$case_id == case_id, ]
    sub <- sub[order(-abs(safe_num(sub$delta))), ][seq_len(min(10, nrow(sub))), ]
    metric_names <- c("A prior", "E posterior", "E - A", "mastery", "recent")
    mat <- cbind(
      safe_num(sub$a_prior),
      safe_num(sub$e_posterior),
      safe_num(sub$delta),
      safe_num(sub$student_support_mastery_logit),
      safe_num(sub$student_support_recent_logit)
    )
    colnames(mat) <- metric_names
    rownames(mat) <- paste(sub$query_concept, "->", sub$support_concept)
    # Keep prior/posterior in probability scale and use diverging range for
    # the signed columns by clipping all columns to a shared visual range.
    visual <- mat
    visual[, 1] <- clip_scale(mat[, 1], 0, max(mat[, 1:2], 1e-6))
    visual[, 2] <- clip_scale(mat[, 2], 0, max(mat[, 1:2], 1e-6))
    visual[, 3] <- clip_scale(mat[, 3], -max(abs(mat[, 3]), 1e-6), max(abs(mat[, 3]), 1e-6))
    visual[, 4] <- clip_scale(mat[, 4], -2, 2)
    visual[, 5] <- clip_scale(mat[, 5], -2, 2)
    meta <- e_cases[e_cases$case_id == case_id, ]
    max_delta <- max(abs(safe_num(sub$delta)), na.rm = TRUE)
    title <- if (nrow(meta) > 0) {
      sprintf(
        "E Personalized Tutor Map: %s | y=%s full=%.3f no_E=%.3f max|E-A|=%.3f",
        case_id,
        meta$label[[1]],
        safe_num(meta$full_prob[[1]]),
        safe_num(meta$no_E_prob[[1]]),
        max_delta
      )
    } else {
      paste("E Personalized Tutor Map:", case_id)
    }
    draw_matrix_heatmap(
      visual,
      rownames(visual),
      colnames(visual),
      title,
      file.path(out_dir, paste0("E_tutor_heatmap_", case_id, ".png")),
      colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(100),
      zlim = c(0, 1),
      label_mat = mat,
      label_digits = 2
    )
  }

  case_ids <- unique(e_edges$case_id)
  edge_keys <- unique(paste(e_edges$query_concept, "->", e_edges$support_concept))
  if (length(case_ids) > 1 && length(edge_keys) > 0) {
    delta_mat <- matrix(0, nrow = length(edge_keys), ncol = length(case_ids), dimnames = list(edge_keys, case_ids))
    for (i in seq_len(nrow(e_edges))) {
      key <- paste(e_edges$query_concept[[i]], "->", e_edges$support_concept[[i]])
      delta_mat[key, e_edges$case_id[[i]]] <- safe_num(e_edges$delta[[i]])
    }
    keep <- order(rowSums(abs(delta_mat)), decreasing = TRUE)[seq_len(min(10, nrow(delta_mat)))]
    delta_mat <- delta_mat[keep, , drop = FALSE]
    z <- max(abs(delta_mat), na.rm = TRUE)
    draw_matrix_heatmap(
      delta_mat,
      rownames(delta_mat),
      colnames(delta_mat),
      "E Personalized Delta Across Students: posterior minus A prior",
      file.path(out_dir, "E_posterior_delta_by_student.png"),
      colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(100),
      zlim = c(-max(z, 1e-6), max(z, 1e-6)),
      label_mat = delta_mat,
      label_digits = 3
    )
  }

  support_keys <- unique(paste(e_edges$query_concept, "->", e_edges$support_concept))
  if (length(case_ids) > 1 && length(support_keys) > 0) {
    mastery_mat <- matrix(0, nrow = length(support_keys), ncol = length(case_ids), dimnames = list(support_keys, case_ids))
    recent_mat <- matrix(0, nrow = length(support_keys), ncol = length(case_ids), dimnames = list(support_keys, case_ids))
    for (i in seq_len(nrow(e_edges))) {
      key <- paste(e_edges$query_concept[[i]], "->", e_edges$support_concept[[i]])
      mastery_mat[key, e_edges$case_id[[i]]] <- safe_num(e_edges$student_support_mastery_logit[[i]])
      recent_mat[key, e_edges$case_id[[i]]] <- safe_num(e_edges$student_support_recent_logit[[i]])
    }
    row_score <- rowSums(abs(mastery_mat), na.rm = TRUE) + rowSums(abs(recent_mat), na.rm = TRUE)
    keep <- order(row_score, decreasing = TRUE)[seq_len(min(10, nrow(mastery_mat)))]
    mastery_mat <- mastery_mat[keep, , drop = FALSE]
    recent_mat <- recent_mat[keep, , drop = FALSE]
    z <- max(abs(c(mastery_mat, recent_mat)), na.rm = TRUE)
    z <- max(z, 1e-6)
    png(file.path(out_dir, "E_student_state_by_support.png"), width = 2200, height = 1200, res = 170)
    op <- par(mar = c(8, 9, 4, 3), mfrow = c(1, 2))
    draw_one <- function(mat, main) {
      image(
        x = seq_len(ncol(mat)),
        y = seq_len(nrow(mat)),
        z = t(mat[nrow(mat):1, , drop = FALSE]),
        col = colorRampPalette(c("#2166ac", "#f7f7f7", "#b2182b"))(100),
        axes = FALSE,
        xlab = "",
        ylab = "",
        main = main,
        zlim = c(-z, z)
      )
      axis(1, at = seq_len(ncol(mat)), labels = colnames(mat), las = 2, cex.axis = 0.7)
      axis(2, at = seq_len(nrow(mat)), labels = rev(rownames(mat)), las = 2, cex.axis = 0.7)
      box()
      for (i in seq_len(nrow(mat))) {
        for (j in seq_len(ncol(mat))) {
          text(j, nrow(mat) - i + 1, sprintf("%.2f", mat[i, j]), cex = 0.52, col = "black")
        }
      }
    }
    draw_one(mastery_mat, "Student support mastery")
    draw_one(recent_mat, "Student recent state")
    par(op)
    dev.off()
  }

  png(file.path(out_dir, "E_prior_to_posterior_shift.png"), width = 1800, height = 1200, res = 170)
  op <- par(mar = c(7, 5, 4, 2), mfrow = c(max(1, ceiling(length(unique(e_edges$case_id)) / 2)), 2))
  for (case_id in unique(e_edges$case_id)) {
    sub <- e_edges[e_edges$case_id == case_id, ]
    sub <- sub[order(-abs(safe_num(sub$delta))), ][seq_len(min(8, nrow(sub))), ]
    ymax <- max(c(sub$a_prior, sub$e_posterior), na.rm = TRUE)
    plot(c(1, 2), c(0, max(ymax, 1e-4)), type = "n", xaxt = "n", xlab = "", ylab = "Weight", main = case_id)
    axis(1, at = c(1, 2), labels = c("A prior", "E posterior"))
    for (i in seq_len(nrow(sub))) {
      col <- ifelse(safe_num(sub$delta[[i]]) >= 0, "#b2182b", "#2166ac")
      lines(c(1, 2), c(safe_num(sub$a_prior[[i]]), safe_num(sub$e_posterior[[i]])), col = col, lwd = 2)
      points(c(1, 2), c(safe_num(sub$a_prior[[i]]), safe_num(sub$e_posterior[[i]])), col = col, pch = 16)
    }
  }
  par(op)
  dev.off()
}

hist <- read_table("e_case_history.csv")
if (nrow(hist) > 0) {
  case_ids <- unique(hist$case_id)
  png(file.path(out_dir, "E_recent_history_strip.png"), width = 1800, height = 900, res = 170)
  op <- par(mar = c(5, 8, 4, 8), xpd = NA)
  x_rng <- range(hist$history_pos, na.rm = TRUE)
  plot(
    NA,
    xlim = x_rng + c(-0.5, 2.0),
    ylim = c(0.5, length(case_ids) + 0.5),
    xaxt = "n",
    yaxt = "n",
    xlab = "Recent train interactions touching case-related concepts",
    ylab = "",
    main = "E Case: Student Recent Concept History"
  )
  axis(1, at = sort(unique(hist$history_pos)))
  axis(2, at = seq_along(case_ids), labels = case_ids, las = 2)
  for (i in seq_len(nrow(hist))) {
    y <- match(hist$case_id[[i]], case_ids)
    col <- ifelse(safe_num(hist$hist_label[[i]]) > 0.5, "#2ca25f", "#de2d26")
    pch <- ifelse(safe_num(hist$hit_related[[i]]) > 0.5, 16, 1)
    points(safe_num(hist$history_pos[[i]]), y, pch = pch, col = col, cex = 1.5)
  }
  legend(max(x_rng) + 0.25, length(case_ids) + 0.45, legend = c("correct", "wrong", "related concept", "other"), col = c("#2ca25f", "#de2d26", "black", "black"), pch = c(16, 16, 16, 1), bty = "n")
  par(op)
  dev.off()
}

if (nrow(a_edges) > 0) {
  mix <- aggregate(safe_num(a_edges$a_weight), by = list(case_id = a_edges$case_id, source = a_edges$source), FUN = sum)
  names(mix)[3] <- "weight"
  case_ids <- unique(mix$case_id)
  sources <- unique(mix$source)
  mat <- matrix(0, nrow = length(sources), ncol = length(case_ids), dimnames = list(sources, case_ids))
  for (i in seq_len(nrow(mix))) {
    mat[mix$source[[i]], mix$case_id[[i]]] <- safe_num(mix$weight[[i]])
  }
  if (ncol(mat) > 0) {
    png(file.path(out_dir, "A_evidence_source_mix.png"), width = 1700, height = 1000, res = 170)
    op <- par(mar = c(7, 5, 4, 2))
    barplot(
      mat,
      beside = FALSE,
      las = 2,
      col = c("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#BAB0AC")[seq_len(nrow(mat))],
      ylab = "Sum of selected A edge weights",
      main = "A Global Roadmap: Evidence Source Mix"
    )
    legend("topright", legend = rownames(mat), fill = c("#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D", "#BAB0AC")[seq_len(nrow(mat))], bty = "n", cex = 0.75)
    par(op)
    dev.off()
  }
}

selected <- read_table("selected_cases.csv")
if (nrow(selected) > 0) {
  png(file.path(out_dir, "case_probability_comparison.png"), width = 1800, height = 1100, res = 170)
  op <- par(mar = c(10, 5, 4, 2))
  labels <- paste(selected$case_type, selected$eval_row_id, "y=", selected$label)
  values <- rbind(
    safe_num(selected$full_prob),
    if ("no_A_prob" %in% names(selected)) safe_num(selected$no_A_prob) else rep(NA, nrow(selected)),
    if ("no_E_prob" %in% names(selected)) safe_num(selected$no_E_prob) else rep(NA, nrow(selected))
  )
  barplot(
    values,
    beside = TRUE,
    names.arg = labels,
    las = 2,
    col = c("#4C78A8", "#F58518", "#54A24B"),
    ylim = c(0, 1),
    ylab = "Predicted probability",
    main = "Selected Case Prediction Probabilities"
  )
  abline(h = 0.5, lty = 2, col = "gray40")
  legend("topright", legend = c("full", "no_A", "no_E"), fill = c("#4C78A8", "#F58518", "#54A24B"), bty = "n")
  par(op)
  dev.off()
}

cat("Case-study plots written to", out_dir, "\n")
