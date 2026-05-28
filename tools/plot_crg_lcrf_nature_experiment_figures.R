#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(ragg)
})

root <- getwd()
png_dir <- file.path(root, "docs", "paper_review_2025_2026", "figures_preview_png")
pdf_dir <- file.path(root, "docs", "paper_review_2025_2026", "figures_main_pdf")
tex_fig_dir <- file.path(
  root, "docs", "paper_review_2025_2026", "icdm2026_template",
  "IEEEtran_CTAN", "IEEEtran", "figures"
)
dir.create(png_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(pdf_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tex_fig_dir, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  blue = "#244C8F",
  blue_mid = "#5E84BF",
  teal = "#3A9D9A",
  teal_light = "#9ED5CF",
  grey = "#B8B8B8",
  grey_dark = "#555555",
  orange = "#D98C32",
  red = "#C8524A"
)
cv <- function(name) unname(pal[name])

theme_set(
  theme_classic(base_size = 7, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.30, colour = "black"),
      axis.title = element_text(size = 7),
      axis.text = element_text(size = 6.4, colour = "black"),
      legend.title = element_blank(),
      legend.text = element_text(size = 6.1),
      legend.key.size = unit(3.5, "mm"),
      legend.position = "bottom",
      strip.background = element_blank(),
      strip.text = element_text(size = 6.6, face = "bold"),
      plot.title = element_text(size = 7.2, face = "bold", margin = margin(b = 3)),
      plot.subtitle = element_text(size = 6.3, colour = pal["grey_dark"]),
      plot.margin = margin(3, 3, 3, 3),
      panel.grid = element_blank()
    )
)

save_pub <- function(plot, stem, width_mm = 183, height_mm = 110, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  grDevices::cairo_pdf(
    file.path(pdf_dir, paste0(stem, ".pdf")),
    width = w, height = h, family = "Arial"
  )
  print(plot)
  dev.off()
  ragg::agg_png(
    file.path(png_dir, paste0(stem, ".png")),
    width = w, height = h, units = "in", res = dpi, background = "white"
  )
  print(plot)
  dev.off()
  file.copy(
    file.path(png_dir, paste0(stem, ".png")),
    file.path(tex_fig_dir, paste0(stem, ".png")),
    overwrite = TRUE
  )
  file.copy(
    file.path(pdf_dir, paste0(stem, ".pdf")),
    file.path(tex_fig_dir, paste0(stem, ".pdf")),
    overwrite = TRUE
  )
}

dataset_levels <- c("assist_09", "junyi", "assist_17")
dataset_labels <- c(assist_09 = "ASSIST09", junyi = "Junyi", assist_17 = "ASSIST17")

method_labels <- c(
  "random" = "Rand",
  "degree-random" = "Deg-rand",
  "self-only" = "Self",
  "seq-only" = "Seq",
  "item-only" = "Item",
  "fused CRG" = "Fused"
)

short_dataset <- function(x) factor(x, levels = dataset_levels, labels = dataset_labels[dataset_levels])

read_history_retrieval <- function() {
  files <- list.files(
    file.path(root, "results", "main_problem_experiments_20260523"),
    pattern = "main_problem_exp1_history_to_query_route_summary\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  bind_rows(lapply(files, read_csv, show_col_types = FALSE)) %>%
    mutate(dataset = factor(dataset, levels = dataset_levels))
}

read_route_cases <- function() {
  files <- list.files(
    file.path(root, "results", "main_problem_experiments_20260523"),
    pattern = "main_problem_exp1_route_case\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  bind_rows(lapply(files, read_csv, show_col_types = FALSE)) %>%
    mutate(dataset = factor(dataset, levels = dataset_levels))
}

read_global_retrieval <- function() {
  path <- file.path(
    root, "results", "crg_lcrf_core3_final_20260520",
    "paper_figures", "fig2_core3_retrieval_summary.csv"
  )
  read_csv(path, show_col_types = FALSE) %>%
    mutate(
      method_group = case_when(
        grepl("self", variant, ignore.case = TRUE) ~ "Self",
        grepl("degree", variant, ignore.case = TRUE) ~ "Deg-rand",
        grepl("uniform|random", variant, ignore.case = TRUE) ~ "Rand",
        grepl("seq|fused", variant, ignore.case = TRUE) ~ "Best CRG",
        TRUE ~ "Other"
      )
    ) %>%
    filter(method_group != "Other") %>%
    group_by(dataset, method_group) %>%
    slice_max(order_by = `hit@10`, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(dataset = factor(dataset, levels = dataset_levels))
}

fig_route_retrieval <- function() {
  hist <- read_history_retrieval() %>%
    filter(group == "direct_unseen_bridgeable",
           method %in% c("random", "seq-only", "fused CRG")) %>%
    mutate(
      method = factor(unname(method_labels[method]), levels = c("Rand", "Seq", "Fused")),
      dataset_lab = short_dataset(as.character(dataset))
    )

  ggplot(hist, aes(dataset_lab, hit10, fill = method)) +
    geom_col(position = position_dodge(width = 0.64), width = 0.52, linewidth = 0.15, colour = "grey25") +
    geom_text(
      aes(label = sprintf("%.2f", hit10)),
      position = position_dodge(width = 0.68),
      vjust = -0.35, size = 1.7
    ) +
    scale_fill_manual(values = c(Rand = cv("grey"), Seq = cv("teal"), Fused = cv("blue"))) +
    guides(fill = guide_legend(nrow = 1, keywidth = unit(3.2, "mm"), keyheight = unit(2.8, "mm"))) +
    scale_y_continuous(limits = c(0, max(hist$hit10, na.rm = TRUE) * 1.22), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = "Hit@10") +
    theme(legend.position = "top", legend.text = element_text(size = 5.4))
}

fig_heldout_retrieval <- function() {
  global <- read_global_retrieval() %>%
    filter(method_group %in% c("Self", "Rand", "Deg-rand", "Best CRG")) %>%
    mutate(
      method_group = recode(method_group, "Best CRG" = "Best"),
      method_group = factor(method_group, levels = c("Self", "Rand", "Deg-rand", "Best")),
      dataset_lab = short_dataset(as.character(dataset))
    )

  p_global <- ggplot(global, aes(dataset_lab, `hit@10`, fill = method_group)) +
    geom_col(position = position_dodge(width = 0.70), width = 0.50, linewidth = 0.15, colour = "grey25") +
    scale_fill_manual(values = c("Self" = "#D2D2D2", "Rand" = "#A8A8A8", "Deg-rand" = cv("blue_mid"), "Best" = cv("blue"))) +
    guides(fill = guide_legend(nrow = 1, keywidth = unit(3.2, "mm"), keyheight = unit(2.8, "mm"))) +
    scale_y_continuous(limits = c(0, max(global$`hit@10`, na.rm = TRUE) * 1.18), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = "Hit@10") +
    theme(legend.position = "top", legend.text = element_text(size = 5.4))
}

fig_coverage <- function() {
  path <- file.path(root, "results", "main_problem_experiments_20260523", "coverage_conditioned_prediction_table.csv")
  cov <- read_csv(path, show_col_types = FALSE) %>%
    filter(
      comparison %in% c("full_vs_no_CRG", "full_vs_no_LCRF"),
      subgroup %in% c("direct_unseen_bridgeable", "weak_direct_evidence", "high_route_mass")
    ) %>%
    mutate(
      dataset_lab = short_dataset(dataset),
      subgroup = recode(subgroup,
        "direct_unseen_bridgeable" = "unseen",
        "weak_direct_evidence" = "weak",
        "high_route_mass" = "high"
      ),
      subgroup = factor(subgroup, levels = c("high", "weak", "unseen")),
      comparison = recode(comparison, "full_vs_no_CRG" = "vs w/o CRG", "full_vs_no_LCRF" = "vs w/o LCRF")
    )

  ggplot(cov, aes(subgroup, delta_bce, fill = comparison, alpha = dataset != "junyi")) +
    geom_hline(yintercept = 0, linewidth = 0.32, colour = "grey55", linetype = "dashed") +
    geom_col(position = position_dodge(width = 0.62), width = 0.50, colour = "grey25", linewidth = 0.14) +
    geom_errorbar(
      aes(ymin = delta_bce_ci_low, ymax = delta_bce_ci_high),
      position = position_dodge(width = 0.62),
      width = 0.12, linewidth = 0.28
    ) +
    facet_wrap(~ dataset_lab, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = c("vs w/o CRG" = cv("blue"), "vs w/o LCRF" = cv("teal"))) +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.42), guide = "none") +
    labs(x = NULL, y = "Delta BCE") +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 5.8),
      axis.text.x = element_text(size = 5.7),
      strip.text = element_text(size = 6.3, face = "bold"),
      panel.spacing = unit(2.2, "mm")
    )
}

fig_route_coverage_combined <- function() {
  hist <- read_history_retrieval() %>%
    filter(group == "direct_unseen_bridgeable",
           method %in% c("random", "seq-only", "fused CRG")) %>%
    mutate(
      method = factor(unname(method_labels[method]), levels = c("Rand", "Seq", "Fused")),
      dataset_lab = short_dataset(as.character(dataset))
    )

  p_route <- ggplot(hist, aes(dataset_lab, hit10, fill = method)) +
    geom_col(
      position = position_dodge(width = 0.72),
      width = 0.64, linewidth = 0.18, colour = "grey25"
    ) +
    scale_fill_manual(values = c(Rand = cv("grey"), Seq = cv("teal"), Fused = cv("blue"))) +
    scale_y_continuous(
      limits = c(0, max(hist$hit10, na.rm = TRUE) * 1.15),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(x = NULL, y = "Hit@10") +
    guides(fill = guide_legend(nrow = 1, keywidth = unit(4.0, "mm"), keyheight = unit(2.8, "mm"))) +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 5.7),
      axis.text.x = element_text(size = 6.0),
      plot.margin = margin(2, 3, 0, 3)
    )

  path <- file.path(root, "results", "main_problem_experiments_20260523", "coverage_conditioned_prediction_table.csv")
  cov <- read_csv(path, show_col_types = FALSE) %>%
    filter(
      comparison %in% c("full_vs_no_CRG", "full_vs_no_LCRF"),
      subgroup %in% c("direct_unseen_bridgeable", "weak_direct_evidence", "high_route_mass")
    ) %>%
    mutate(
      dataset_lab = recode(dataset, assist_09 = "A09", junyi = "Junyi", assist_17 = "A17"),
      subgroup = recode(subgroup,
        "direct_unseen_bridgeable" = "unseen",
        "weak_direct_evidence" = "weak",
        "high_route_mass" = "high"
      ),
      subgroup = factor(subgroup, levels = c("unseen", "weak", "high")),
      comparison = recode(comparison, "full_vs_no_CRG" = "vs w/o CRG", "full_vs_no_LCRF" = "vs w/o LCRF"),
      row_lab = paste(dataset_lab, subgroup, sep = " / "),
      row_lab = factor(row_lab, levels = rev(c(
        "A09 / unseen", "A09 / weak", "A09 / high",
        "Junyi / unseen", "Junyi / weak", "Junyi / high",
        "A17 / unseen", "A17 / weak", "A17 / high"
      ))),
      alpha_val = if_else(dataset == "junyi", 0.45, 1)
    )

  p_cov <- ggplot(cov, aes(delta_bce, row_lab, fill = comparison, alpha = alpha_val)) +
    geom_vline(xintercept = 0, linewidth = 0.28, colour = "grey60", linetype = "dashed") +
    geom_col(
      position = position_dodge(width = 0.72),
      width = 0.62, colour = "grey25", linewidth = 0.14
    ) +
    geom_errorbar(
      aes(xmin = delta_bce_ci_low, xmax = delta_bce_ci_high),
      position = position_dodge(width = 0.72),
      width = 0.14, linewidth = 0.25, colour = "grey25", orientation = "y"
    ) +
    scale_fill_manual(values = c("vs w/o CRG" = cv("blue"), "vs w/o LCRF" = cv("teal"))) +
    scale_alpha_identity() +
    scale_x_continuous(labels = label_number(accuracy = 0.01)) +
    labs(x = "Delta BCE", y = NULL) +
    guides(fill = guide_legend(nrow = 1, keywidth = unit(4.0, "mm"), keyheight = unit(2.8, "mm"))) +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 5.7),
      axis.text.y = element_text(size = 5.5),
      axis.text.x = element_text(size = 5.6),
      plot.margin = margin(0, 3, 2, 3)
    )

  p_route / p_cov + plot_layout(heights = c(0.95, 1.35))
}

read_support_corruption <- function() {
  path <- file.path(
    root, "results", "crg_lcrf_core3_final_20260520",
    "crg_support_audit", "crg_support_gap_audit_core3.csv"
  )
  support <- read_csv(path, show_col_types = FALSE) %>%
    filter(
      subgroup == "all",
      corruption_type %in% c(
        "evidence_support_corruption",
        "degree_matched_random_support",
        "sequence_shuffled_support",
        "self_only_fallback"
      )
    ) %>%
    mutate(
      dataset_lab = short_dataset(dataset),
      corruption_type = recode(corruption_type,
        "evidence_support_corruption" = "Evidence",
        "degree_matched_random_support" = "Deg-rand",
        "sequence_shuffled_support" = "Seq-shuf",
        "self_only_fallback" = "Self-only"
      ),
      corruption_type = factor(corruption_type, levels = c("Evidence", "Deg-rand", "Seq-shuf", "Self-only"))
    ) %>%
    group_by(dataset_lab, corruption_type, corruption_ratio) %>%
    summarise(
      auc_drop = mean(auc_drop_from_clean, na.rm = TRUE),
      bce_inc = mean(bce_increase_from_clean, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      ratio_pct = corruption_ratio * 100
    )
}

fig_support_corruption_metric <- function(metric_name = c("auc_drop", "bce_inc")) {
  metric_name <- match.arg(metric_name)
  support <- read_support_corruption()

  y_lab <- if (metric_name == "auc_drop") "AUC drop" else "BCE increase"
  curve_df <- support %>% mutate(value = .data[[metric_name]])

  ggplot(curve_df, aes(ratio_pct, value, colour = corruption_type, linetype = corruption_type)) +
    geom_line(linewidth = 0.45) +
    geom_point(size = 1.1) +
    facet_wrap(~ dataset_lab, nrow = 1, scales = "free_y") +
    scale_x_continuous(breaks = c(0, 25, 50, 75, 100)) +
    scale_colour_manual(values = c(Evidence = cv("blue"), "Deg-rand" = cv("blue_mid"), "Seq-shuf" = cv("teal"), "Self-only" = cv("grey_dark"))) +
    scale_linetype_manual(values = c(Evidence = "solid", "Deg-rand" = "dashed", "Seq-shuf" = "dotdash", "Self-only" = "dotted")) +
    labs(x = "Corruption ratio (%)", y = y_lab) +
    theme(
      legend.position = "top",
      strip.placement = "outside"
    )
}

fig_support_corruption_combined <- function() {
  support <- read_support_corruption() %>%
    pivot_longer(c(auc_drop, bce_inc), names_to = "metric", values_to = "value") %>%
    mutate(
      metric = recode(metric, auc_drop = "AUC drop", bce_inc = "BCE increase"),
      metric = factor(metric, levels = c("AUC drop", "BCE increase"))
    )

  ggplot(support, aes(ratio_pct, value, colour = corruption_type, linetype = corruption_type)) +
    geom_line(linewidth = 0.42) +
    geom_point(size = 0.95) +
    facet_grid(metric ~ dataset_lab, scales = "free_y") +
    scale_x_continuous(breaks = c(0, 50, 100)) +
    scale_colour_manual(values = c(Evidence = cv("blue"), "Deg-rand" = cv("blue_mid"), "Seq-shuf" = cv("teal"), "Self-only" = cv("grey_dark"))) +
    scale_linetype_manual(values = c(Evidence = "solid", "Deg-rand" = "dashed", "Seq-shuf" = "dotdash", "Self-only" = "dotted")) +
    labs(x = "Corruption ratio (%)", y = NULL) +
    theme(
      legend.position = "top",
      legend.text = element_text(size = 5.2),
      legend.key.width = unit(4.6, "mm"),
      legend.key.height = unit(2.8, "mm"),
      strip.text = element_text(size = 5.9, face = "bold"),
      panel.spacing.x = unit(1.8, "mm"),
      panel.spacing.y = unit(1.7, "mm"),
      axis.text.x = element_text(size = 5.2)
    )
}

fig_lcrf_counterfactual <- function() {
  path <- file.path(
    root, "results", "crg_lcrf_core3_final_20260520",
    "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv"
  )
  cf <- read_csv(path, show_col_types = FALSE) %>%
    filter(dataset %in% c("assist_09", "junyi", "assist_17")) %>%
    mutate(
      dataset_lab = short_dataset(dataset),
      variant = recode(variant, "no_filter" = "No filter", "mean_state" = "Mean state", "shuffle_state" = "Shuffle state"),
      variant = factor(variant, levels = c("No filter", "Mean state", "Shuffle state")),
      alpha_val = if_else(dataset == "junyi", 0.42, 1)
    )

  ggplot(cf, aes(dataset_lab, auc_drop_from_full, fill = variant, alpha = alpha_val)) +
    geom_col(position = position_dodge(width = 0.64), width = 0.50, colour = "grey25", linewidth = 0.15) +
    geom_text(
      aes(label = sprintf("%.2f", auc_drop_from_full)),
      position = position_dodge(width = 0.64),
      vjust = -0.35, size = 1.85,
      colour = "#2B2B2B"
    ) +
    scale_fill_manual(values = c("No filter" = cv("blue_mid"), "Mean state" = cv("teal"), "Shuffle state" = cv("blue"))) +
    scale_alpha_identity() +
    scale_y_continuous(limits = c(0, max(cf$auc_drop_from_full, na.rm = TRUE) * 1.18), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = "AUC drop") +
    theme(
      legend.position = "top",
      legend.key.width = unit(5.0, "mm"),
      legend.key.height = unit(2.7, "mm")
    )
}

fig_same_query <- function() {
  path <- file.path(
    root, "results", "crg_lcrf_core3_final_20260520",
    "lcrf_same_query", "lcrf_two_student_path_case_core3.csv"
  )
  case_df <- read_csv(path, show_col_types = FALSE) %>%
    filter(dataset == "assist_17") %>%
    mutate(
      learner_id_anonymized = factor(learner_id_anonymized, levels = unique(learner_id_anonymized)),
      support_label = paste0("C", support_concept_id)
    )

  top_support <- case_df %>%
    group_by(support_label) %>%
    summarise(global_support_prob = mean(global_support_prob, na.rm = TRUE), .groups = "drop") %>%
    slice_max(order_by = global_support_prob, n = 6, with_ties = FALSE)

  support_levels <- top_support %>% arrange(global_support_prob) %>% pull(support_label)

  global_plot_df <- top_support %>%
    mutate(support_label = factor(support_label, levels = support_levels))

  posterior_plot_df <- case_df %>%
    filter(support_label %in% support_levels) %>%
    mutate(support_label = factor(support_label, levels = support_levels))

  pred <- case_df %>%
    group_by(learner_id_anonymized) %>%
    summarise(
      pred_global = first(pred_global),
      pred_full = first(pred_full),
      true_label = first(true_label),
      .groups = "drop"
    ) %>%
    mutate(y = as.numeric(learner_id_anonymized))

  state <- case_df %>%
    group_by(learner_id_anonymized) %>%
    summarise(
      Mastery = first(query_mastery),
      Recent = first(query_recent_mastery),
      y = first(true_label),
      .groups = "drop"
    ) %>%
    pivot_longer(-learner_id_anonymized, names_to = "feature", values_to = "value") %>%
    mutate(
      feature = factor(feature, levels = c("Mastery", "Recent", "y")),
      value_label = if_else(feature == "y", paste0("y=", value), sprintf("%.2f", value))
    )

  state_labels <- state %>%
    pivot_wider(
      id_cols = learner_id_anonymized,
      names_from = feature,
      values_from = value_label,
      values_fn = dplyr::first
    ) %>%
    mutate(row_label = paste0(learner_id_anonymized, " (", Mastery, "/", Recent, ", ", y, ")")) %>%
    select(learner_id_anonymized, row_label)

  route_df <- posterior_plot_df %>%
    left_join(state_labels, by = "learner_id_anonymized") %>%
    mutate(
      support_label = factor(support_label, levels = support_levels),
      row_label = factor(row_label, levels = unique(row_label))
    )

  heat_df <- route_df %>%
    mutate(delta = posterior_prob - global_support_prob)

  heat_plot <- ggplot(heat_df, aes(support_label, row_label, fill = delta)) +
    geom_tile(colour = "white", linewidth = 0.45) +
    geom_text(aes(label = sprintf("%.2f", posterior_prob)), size = 1.75, colour = "grey15") +
    scale_fill_gradient2(
      low = "#2F5597", mid = "#F7F7F7", high = "#C8524A",
      midpoint = 0,
      limits = c(-max(abs(heat_df$delta), na.rm = TRUE), max(abs(heat_df$delta), na.rm = TRUE)),
      name = "Post.-prior"
    ) +
    labs(x = "Support concept", y = NULL) +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 5.5),
      legend.text = element_text(size = 5.3),
      legend.key.height = unit(9, "mm"),
      axis.text.x = element_text(size = 5.8, angle = 0, hjust = 0.5),
      axis.text.y = element_text(size = 5.8),
      axis.ticks = element_blank()
    )

  pred_df <- case_df %>%
    left_join(state_labels, by = "learner_id_anonymized") %>%
    distinct(row_label, pred_global, pred_full, true_label) %>%
    pivot_longer(c(pred_global, pred_full), names_to = "variant", values_to = "pred") %>%
    mutate(
      variant = recode(variant, pred_global = "Global", pred_full = "Full"),
      variant = factor(variant, levels = c("Global", "Full")),
      row_label = factor(row_label, levels = levels(route_df$row_label))
    )

  pred_plot <- ggplot(pred_df, aes(pred, row_label, colour = variant, group = row_label)) +
    geom_line(colour = "grey70", linewidth = 0.35) +
    geom_point(size = 1.45) +
    scale_colour_manual(values = c(Global = cv("grey_dark"), Full = cv("red"))) +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
    labs(x = "Prediction", y = NULL) +
    theme(
      legend.position = "bottom",
      legend.text = element_text(size = 5.8),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )

  heat_plot / pred_plot + plot_layout(heights = c(2.55, 1.0))
}

save_pub(fig_route_retrieval(), "fig4_history_to_query_retrieval", width_mm = 88, height_mm = 52)
save_pub(fig_heldout_retrieval(), "fig5_heldout_transition_retrieval", width_mm = 88, height_mm = 52)
save_pub(fig_coverage(), "fig6_coverage_conditioned_prediction", width_mm = 88, height_mm = 72)
save_pub(fig_route_coverage_combined(), "fig3_route_retrieval_coverage_prediction", width_mm = 88, height_mm = 82)
save_pub(fig_support_corruption_metric("auc_drop"), "fig7_crg_support_auc_drop", width_mm = 88, height_mm = 58)
save_pub(fig_support_corruption_metric("bce_inc"), "fig8_crg_support_bce_increase", width_mm = 88, height_mm = 58)
save_pub(fig_support_corruption_combined(), "fig7_crg_support_perturbation", width_mm = 88, height_mm = 62)
save_pub(fig_lcrf_counterfactual(), "fig9_lcrf_counterfactual", width_mm = 88, height_mm = 45)
save_pub(fig_same_query(), "fig10_lcrf_same_query", width_mm = 88, height_mm = 72)

message("Saved CRG/LCRF nature-style experiment figures to:")
message("  ", png_dir)
message("  ", pdf_dir)
