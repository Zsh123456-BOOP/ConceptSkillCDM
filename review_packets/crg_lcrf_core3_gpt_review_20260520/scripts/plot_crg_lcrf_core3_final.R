#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

if (!requireNamespace("gridExtra", quietly = TRUE)) {
  stop("gridExtra is required for compact multi-panel figures")
}

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "results/crg_lcrf_core3_final_20260520"
fig_dir <- file.path(root, "paper_figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

dataset_levels <- c("assist_09", "assist_17", "junyi")

read_csv <- function(path) {
  if (!file.exists(path)) {
    return(data.frame())
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

save_both <- function(plot, name, width = 7.2, height = 4.2) {
  ggsave(file.path(fig_dir, paste0(name, ".pdf")), plot, width = width, height = height, units = "in", device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, paste0(name, ".png")), plot, width = width, height = height, units = "in", dpi = 300, bg = "white")
}

theme_core <- function(base_size = 8) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(linewidth = 0.2, color = "grey88"),
      panel.grid.major.y = element_blank(),
      strip.text = element_text(face = "bold", size = base_size),
      legend.title = element_blank(),
      legend.position = "bottom",
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 1),
      plot.caption = element_text(size = base_size - 1, hjust = 0)
    )
}

palette <- c(
  "Best CRG" = "#2878B5",
  "Self" = "#7F7F7F",
  "Rand" = "#D59A2D",
  "Deg-rand" = "#B65F5F",
  "Evidence" = "#2878B5",
  "Seq-shuf" = "#D59A2D",
  "Self-only" = "#7F7F7F",
  "no_filter" = "#7F7F7F",
  "mean_state" = "#D59A2D",
  "shuffle_state" = "#B65F5F",
  "assist_09" = "#2878B5",
  "assist_17" = "#4C9A6A",
  "junyi" = "#9A9A9A"
)

fmt_pct <- function(x) sprintf("%.1f%%", 100 * as.numeric(x))
fmt_num <- function(x, digits = 3) {
  ifelse(is.na(x), "n/a", sprintf(paste0("%.", digits, "f"), as.numeric(x)))
}

label_corruption <- function(x) {
  out <- as.character(x)
  out[out == "evidence_support_corruption"] <- "Evidence"
  out[out == "degree_matched_random_support"] <- "Deg-rand"
  out[out == "sequence_shuffled_support"] <- "Seq-shuf"
  out[out == "self_only_fallback"] <- "Self-only"
  factor(out, levels = c("Evidence", "Deg-rand", "Seq-shuf", "Self-only"))
}

make_table <- function(df, rows = NULL, base_size = 8) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  nr <- nrow(df)
  nc <- ncol(df)
  cell <- expand.grid(row = seq_len(nr + 1), col = seq_len(nc))
  cell$label <- ""
  for (j in seq_len(nc)) {
    cell$label[cell$row == 1 & cell$col == j] <- names(df)[j]
    for (i in seq_len(nr)) {
      cell$label[cell$row == i + 1 & cell$col == j] <- as.character(df[i, j])
    }
  }
  cell$fill <- ifelse(cell$row == 1, "#EEF2F6", "white")
  cell$fontface <- ifelse(cell$row == 1, "bold", "plain")
  ggplot(cell, aes(col, -row)) +
    geom_tile(aes(fill = fill), color = "grey86", linewidth = 0.25) +
    geom_text(aes(label = label, fontface = fontface), size = base_size / 2.8, lineheight = 0.92) +
    scale_fill_identity() +
    scale_x_continuous(expand = c(0, 0), limits = c(0.5, nc + 0.5)) +
    scale_y_continuous(expand = c(0, 0), limits = c(-(nr + 1.5), -0.5)) +
    coord_fixed(ratio = 0.35, clip = "off") +
    theme_void()
}

# ---------------------------------------------------------------------------
# Figure 2 final: dataset cards + retrieval lift.
# ---------------------------------------------------------------------------
cards <- read_csv(file.path(root, "data_story", "dataset_story_cards_core3.csv"))
retr <- read_csv(file.path(fig_dir, "fig2_core3_retrieval_summary.csv"))
if (nrow(cards) > 0 && nrow(retr) > 0) {
  cards <- cards[cards$dataset %in% dataset_levels, ]
  cards$dataset <- factor(cards$dataset, levels = dataset_levels)
  cards <- cards[order(cards$dataset), ]
  card_table <- data.frame(
    Dataset = as.character(cards$dataset),
    Single = fmt_pct(cards$single_concept_rate),
    `Item edge` = fmt_pct(cards$item_edge_density),
    `Seq edge` = fmt_pct(cards$seq_edge_density),
    Unseen = fmt_pct(cards$direct_unseen_rate),
    Bridge = fmt_pct(cards$bridge_only_rate),
    `Hist med` = sprintf("%.0f", cards$history_len_median),
    check.names = FALSE
  )
  card_grob <- make_table(card_table, rows = NULL, base_size = 8)

  retr <- retr[retr$dataset %in% dataset_levels, ]
  retr$method <- ifelse(retr$role == "best_CRG", "Best CRG",
                        ifelse(retr$variant == "CRG_self_only", "Self",
                               ifelse(retr$variant == "CRG_degree_random", "Deg-rand", "Rand")))
  retr$method <- factor(retr$method, levels = c("Self", "Rand", "Deg-rand", "Best CRG"))
  retr$dataset <- factor(retr$dataset, levels = dataset_levels)
  p_retr <- ggplot(retr, aes(`hit@10`, method, color = method)) +
    geom_line(aes(group = dataset), color = "grey82", linewidth = 0.35) +
    geom_point(size = 2.2) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "Hit@10", y = NULL) +
    theme_core()

  g2 <- gridExtra::arrangeGrob(card_grob, p_retr, ncol = 1, heights = c(0.85, 1.25))
  ggsave(file.path(fig_dir, "fig2_core3_data_and_crg_retrieval_final.pdf"), g2, width = 7.3, height = 4.6, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "fig2_core3_data_and_crg_retrieval_final.png"), g2, width = 7.3, height = 4.6, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Figure 3 final: corruption curves + evidence-minus-degree gap.
# ---------------------------------------------------------------------------
support <- read_csv(file.path(root, "crg_support_audit", "crg_support_gap_audit_core3.csv"))
if (nrow(support) > 0) {
  support <- support[support$dataset %in% dataset_levels, ]
  all_rows <- subset(support, subgroup == "all" & corruption_ratio > 0)
  all_mean <- aggregate(
    cbind(auc_drop_from_clean, bce_increase_from_clean) ~ dataset + corruption_type + corruption_ratio,
    all_rows, mean, na.rm = TRUE
  )
  all_mean$dataset <- factor(all_mean$dataset, levels = dataset_levels)
  all_mean$type_label <- label_corruption(all_mean$corruption_type)

  p_auc <- ggplot(all_mean, aes(corruption_ratio, auc_drop_from_clean, color = type_label)) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.4) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "corruption ratio", y = "AUC drop") +
    theme_core()

  p_bce <- ggplot(all_mean, aes(corruption_ratio, bce_increase_from_clean, color = type_label)) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.4) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "corruption ratio", y = "BCE increase") +
    theme_core()

  gap <- subset(all_rows, corruption_type == "evidence_support_corruption" & corruption_ratio == 1)
  gap_mean <- aggregate(
    cbind(evidence_minus_degree_random_auc_drop, evidence_minus_degree_random_bce_increase) ~ dataset,
    gap, mean, na.rm = TRUE
  )
  gap_sd <- aggregate(evidence_minus_degree_random_auc_drop ~ dataset, gap, sd, na.rm = TRUE)
  names(gap_sd)[2] <- "gap_sd"
  gap_mean <- merge(gap_mean, gap_sd, by = "dataset", all.x = TRUE)
  gap_mean$gap_sd[is.na(gap_mean$gap_sd)] <- 0
  gap_mean$status <- ifelse(gap_mean$dataset == "assist_17", "evidence gap",
                            ifelse(gap_mean$dataset == "assist_09", "support-only", "weak"))
  gap_mean$dataset <- factor(gap_mean$dataset, levels = dataset_levels)
  p_gap <- ggplot(gap_mean, aes(dataset, evidence_minus_degree_random_auc_drop, fill = dataset)) +
    geom_col(width = 0.55) +
    geom_errorbar(aes(
      ymin = evidence_minus_degree_random_auc_drop - gap_sd,
      ymax = evidence_minus_degree_random_auc_drop + gap_sd
    ), width = 0.12, linewidth = 0.25) +
    geom_hline(yintercept = 0, linewidth = 0.25) +
    geom_text(aes(label = status), vjust = ifelse(gap_mean$evidence_minus_degree_random_auc_drop >= 0, -0.45, 1.25), size = 2.4) +
    scale_fill_manual(values = palette) +
    labs(x = NULL, y = "Evidence - Deg-rand AUC drop") +
    theme_core() +
    theme(legend.position = "none")

  caption <- "Evidence-specific necessity is dataset dependent: assist_17 is the clean evidence-gap case, assist_09 supports support dependence, and Junyi is weak at prediction-level corruption."
  cap <- textGrob(caption, x = 0, hjust = 0, gp = gpar(fontsize = 7, col = "grey30"))
  g3 <- gridExtra::arrangeGrob(p_auc, p_bce, p_gap, cap, ncol = 1, heights = c(1, 1, 0.85, 0.16))
  ggsave(file.path(fig_dir, "fig3_core3_support_corruption_final.pdf"), g3, width = 7.3, height = 7.2, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "fig3_core3_support_corruption_final.png"), g3, width = 7.3, height = 7.2, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Figure 4 final: shared-y LCRF counterfactual deltas.
# ---------------------------------------------------------------------------
cf <- read_csv(file.path(root, "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv"))
if (nrow(cf) > 0) {
  cf <- cf[cf$dataset %in% dataset_levels, ]
  cf$dataset <- factor(cf$dataset, levels = dataset_levels)
  cf$variant <- factor(cf$variant, levels = c("no_filter", "mean_state", "shuffle_state"))
  cf$variant_label <- factor(ifelse(cf$variant == "no_filter", "No filter",
                                    ifelse(cf$variant == "mean_state", "Mean state", "Shuffle state")),
                             levels = c("No filter", "Mean state", "Shuffle state"))
  p4_auc <- ggplot(cf, aes(variant_label, auc_drop_from_full, fill = variant)) +
    geom_col(width = 0.62) +
    facet_wrap(~ dataset, nrow = 1) +
    geom_text(
      data = subset(cf, dataset == "junyi" & variant == "shuffle_state"),
      aes(label = "weak", y = auc_drop_from_full + 0.012),
      color = "grey35", size = 2.5
    ) +
    scale_fill_manual(values = palette) +
    labs(x = NULL, y = expression(Delta * "AUC from full")) +
    theme_core() +
    theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "none")
  ggsave(file.path(fig_dir, "fig4_core3_lcrf_counterfactual_final.pdf"), p4_auc, width = 7.3, height = 3.1, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "fig4_core3_lcrf_counterfactual_final.png"), p4_auc, width = 7.3, height = 3.1, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Figure 5 final: assist_17 same-query posterior with learner annotation.
# ---------------------------------------------------------------------------
same <- read_csv(file.path(root, "lcrf_same_query", "lcrf_same_query_annotated_core3.csv"))
twostu <- read_csv(file.path(root, "lcrf_same_query", "lcrf_two_student_path_case_core3.csv"))
if (nrow(same) > 0) {
  chosen_case_id <- if (nrow(twostu) > 0 && any(twostu$dataset == "assist_17")) {
    twostu$case_id[twostu$dataset == "assist_17"][1]
  } else {
    assist17 <- subset(same, dataset == "assist_17")
    if (nrow(assist17) > 0) assist17$case_id[1] else same$case_id[1]
  }
  one <- same[same$case_id == chosen_case_id, ]
  one <- one[one$dataset == one$dataset[1], ]
  top_support <- aggregate(abs(posterior_minus_global) ~ support_concept_name, one, mean, na.rm = TRUE)
  top_support <- top_support[order(-top_support$`abs(posterior_minus_global)`), ]
  support_keep <- head(top_support$support_concept_name, 12)
  one <- one[one$support_concept_name %in% support_keep, ]
  learner_order <- unique(one$learner_id_anonymized)
  one$learner_id_anonymized <- factor(one$learner_id_anonymized, levels = learner_order)
  one$support_concept_name <- factor(one$support_concept_name, levels = rev(support_keep))

  support_bar <- aggregate(global_support_prob ~ support_concept_name, one, mean, na.rm = TRUE)
  p5a <- ggplot(support_bar, aes(support_concept_name, global_support_prob)) +
    geom_col(fill = "#2878B5", width = 0.65) +
    coord_flip() +
    labs(x = NULL, y = "CRG prob") +
    theme_core()

  p5b <- ggplot(one, aes(support_concept_name, learner_id_anonymized, fill = posterior_minus_global)) +
    geom_tile(color = "white", linewidth = 0.25) +
    scale_fill_gradient2(low = "#B65F5F", mid = "white", high = "#2878B5") +
    labs(x = "support concept", y = "learner") +
    theme_core()

  anno <- unique(one[, c("learner_id_anonymized", "query_mastery", "query_recent_mastery", "true_label", "pred_full")])
  anno <- anno[order(anno$learner_id_anonymized), ]
  anno_table <- data.frame(
    Learner = as.character(anno$learner_id_anonymized),
    Mastery = sprintf("%.2f", anno$query_mastery),
    Recent = sprintf("%.2f", anno$query_recent_mastery),
    Label = sprintf("%.0f", anno$true_label),
    `Full pred` = sprintf("%.2f", anno$pred_full),
    check.names = FALSE
  )
  anno_grob <- make_table(anno_table, rows = NULL, base_size = 7)

  shift <- unique(one[, c("learner_id_anonymized", "pred_global", "pred_no_filter", "pred_full", "true_label")])
  p5c <- ggplot(shift, aes(y = learner_id_anonymized)) +
    geom_segment(aes(x = pred_global, xend = pred_full, yend = learner_id_anonymized), color = "grey65") +
    geom_point(aes(x = pred_global), color = "#7F7F7F", size = 1.6) +
    geom_point(aes(x = pred_no_filter), color = "#D59A2D", size = 1.6) +
    geom_point(aes(x = pred_full), color = "#2878B5", size = 1.8) +
    labs(x = "prediction: global / no-filter / full", y = NULL) +
    theme_core()

  two <- twostu[twostu$dataset == one$dataset[1] & twostu$case_id == chosen_case_id, ]
  if (nrow(two) > 0) {
    two$support_concept_name <- factor(two$support_concept_name, levels = unique(two$support_concept_name))
    p5d <- ggplot(two, aes(support_concept_name, posterior_prob, fill = learner_id_anonymized)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.62) +
      geom_text(aes(label = sprintf("%.2f", posterior_prob)), position = position_dodge(width = 0.72), vjust = -0.25, size = 2.1) +
      scale_y_continuous(expand = expansion(mult = c(0, 0.14))) +
      labs(x = "top posterior route", y = "posterior") +
      theme_core()
    s1 <- unique(two[, c("learner_id_anonymized", "query_mastery", "query_recent_mastery")])
    state_txt <- paste(
      paste0(s1$learner_id_anonymized, ": mastery ", sprintf("%.2f", s1$query_mastery),
             ", recent ", sprintf("%.2f", s1$query_recent_mastery)),
      collapse = " | "
    )
    state_grob <- textGrob(state_txt, x = 0, hjust = 0, gp = gpar(fontsize = 7, col = "grey30"))
  } else {
    p5d <- p5c
    state_grob <- nullGrob()
  }

  left <- gridExtra::arrangeGrob(anno_grob, p5a, ncol = 1, heights = c(1.05, 1.1))
  right <- gridExtra::arrangeGrob(p5b, p5c, p5d, state_grob, ncol = 1, heights = c(1.35, 0.9, 0.95, 0.12))
  g5 <- gridExtra::arrangeGrob(left, right, ncol = 2, widths = c(1.0, 1.75))
  ggsave(file.path(fig_dir, "fig5_core3_lcrf_same_query_posterior_final.pdf"), g5, width = 7.6, height = 6.2, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "fig5_core3_lcrf_same_query_posterior_final.png"), g5, width = 7.6, height = 6.2, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Appendix: readable CRG local route cases. Missing predictions are explicit.
# ---------------------------------------------------------------------------
route_summary <- read_csv(file.path(root, "crg_local_route_cases", "crg_local_route_case_summary_core3.csv"))
route_edges <- read_csv(file.path(root, "crg_local_route_cases", "crg_local_route_case_edges_core3.csv"))
if (nrow(route_summary) > 0 && nrow(route_edges) > 0) {
  selected_ids <- c(
    route_summary$case_id[route_summary$dataset == "assist_17" & route_summary$selected_case_type == "CRG-positive"][1],
    route_summary$case_id[route_summary$dataset == "assist_09" & route_summary$selected_case_type != "CRG-positive"][1]
  )
  selected_ids <- selected_ids[!is.na(selected_ids)]
  sum_sel <- route_summary[route_summary$case_id %in% selected_ids, ]
  pred_table <- data.frame(
    Dataset = sum_sel$dataset,
    Case = sum_sel$case_id,
    Type = sum_sel$selected_case_type,
    Query = sum_sel$query_concept_name,
    Clean = sprintf("%.2f", sum_sel$clean_pred),
    Evidence = fmt_num(sum_sel$evidence_corrupt_pred, 2),
    `Deg-rand` = fmt_num(sum_sel$degree_random_pred_mean, 2),
    `Self-only` = sprintf("%.2f", sum_sel$self_only_pred),
    check.names = FALSE
  )
  pred_grob <- make_table(pred_table, rows = NULL, base_size = 7)
  edge_sel <- route_edges[route_edges$case_id %in% selected_ids, ]
  edge_sel <- edge_sel[order(edge_sel$case_id, -edge_sel$fused_crg_prob), ]
  edge_sel <- do.call(rbind, by(edge_sel, edge_sel$case_id, head, 6))
  edge_sel$case_id <- factor(edge_sel$case_id, levels = selected_ids)
  p_route <- ggplot(edge_sel, aes(reorder(target_concept_name, fused_crg_prob), fused_crg_prob, fill = sequence_evidence_score)) +
    geom_col(width = 0.62) +
    facet_wrap(~ case_id, scales = "free_y") +
    coord_flip() +
    scale_fill_gradient(low = "#E6EEF7", high = "#2878B5") +
    labs(x = "CRG support route", y = "CRG prob") +
    theme_core() +
    theme(legend.position = "bottom")
  note <- textGrob("History-concept fields and degree-random predictions were not exported in the existing CSV; unavailable values are shown as n/a.", x = 0, hjust = 0, gp = gpar(fontsize = 7, col = "grey35"))
  g_route <- gridExtra::arrangeGrob(pred_grob, p_route, note, ncol = 1, heights = c(0.8, 1.55, 0.16))
  ggsave(file.path(fig_dir, "figS_crg_local_route_cases_core3.pdf"), g_route, width = 7.6, height = 5.3, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "figS_crg_local_route_cases_core3.png"), g_route, width = 7.6, height = 5.3, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Appendix: specific student evidence as a case table, not a timeline.
# ---------------------------------------------------------------------------
timeline <- read_csv(file.path(root, "lcrf_student_timeline", "lcrf_specific_student_timeline_core3.csv"))
if (nrow(timeline) > 0) {
  tl <- timeline[, c(
    "dataset", "student_id_anonymized", "query_concept_name", "true_label",
    "pred_global", "pred_no_filter", "pred_full", "pred_shuffle_state", "pred_mean_state",
    "bce_global", "bce_no_filter", "bce_full", "top1_support_concept", "case_comment"
  )]
  names(tl) <- c("Dataset", "Learner", "Query", "Label", "Global", "No filter", "Full",
                 "Shuffle", "Mean", "BCE global", "BCE no-filter", "BCE full", "Top support", "Comment")
  for (nm in c("Global", "No filter", "Full", "Shuffle", "Mean", "BCE global", "BCE no-filter", "BCE full")) {
    tl[[nm]] <- sprintf("%.2f", as.numeric(tl[[nm]]))
  }
  tl <- head(tl, 6)
  g_tl <- make_table(tl, rows = NULL, base_size = 6)
  ggsave(file.path(fig_dir, "figS_lcrf_specific_student_timeline_core3.pdf"), g_tl, width = 7.8, height = 2.6, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "figS_lcrf_specific_student_timeline_core3.png"), g_tl, width = 7.8, height = 2.6, dpi = 300, bg = "white")
}

# ---------------------------------------------------------------------------
# Appendix: state-source audit as a limitation figure.
# ---------------------------------------------------------------------------
state <- read_csv(file.path(root, "lcrf_state_source_audit", "lcrf_state_source_audit_core3.csv"))
if (nrow(state) > 0) {
  state <- state[state$dataset %in% c("assist_09", "assist_17"), ]
  state$variant <- factor(state$variant, levels = c(
    "global_only", "mean_state_keep_id", "shuffle_state_keep_id",
    "shuffle_id_keep_state", "zero_id_keep_state", "full"
  ))
  p_state <- ggplot(state, aes(variant, auc_drop_from_full, fill = variant)) +
    geom_col(width = 0.62, na.rm = TRUE) +
    facet_wrap(~ dataset, nrow = 1) +
    coord_flip() +
    scale_fill_brewer(palette = "Set2", na.value = "grey85") +
    labs(x = NULL, y = "AUC drop from full") +
    theme_core() +
    theme(legend.position = "none")
  note <- textGrob(
    "Limitation: shuffle_id_keep_state and zero_id_keep_state hooks were not available, so this audit cannot fully rule out a student-ID shortcut.",
    x = 0, hjust = 0, gp = gpar(fontsize = 7, col = "grey35")
  )
  g_state <- gridExtra::arrangeGrob(p_state, note, ncol = 1, heights = c(1, 0.13))
  ggsave(file.path(fig_dir, "figS_lcrf_state_source_audit_core3.pdf"), g_state, width = 7.4, height = 3.8, device = cairo_pdf, bg = "white")
  ggsave(file.path(fig_dir, "figS_lcrf_state_source_audit_core3.png"), g_state, width = 7.4, height = 3.8, dpi = 300, bg = "white")
}
