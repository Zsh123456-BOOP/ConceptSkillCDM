#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1) args[[1]] else "."
out_dir <- if (length(args) >= 2) args[[2]] else file.path(repo_root, "docs", "paper_review_2025_2026", "figures_main_pdf")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dataset_levels <- c("assist_09", "assist_17", "junyi")

theme_set(
  theme_classic(base_size = 6.8, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.28, colour = "grey25"),
      axis.ticks = element_line(linewidth = 0.28, colour = "grey25"),
      axis.text = element_text(colour = "grey20"),
      axis.title = element_text(colour = "grey10"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 6.8),
      legend.title = element_blank(),
      legend.text = element_text(size = 6.1),
      legend.key.height = unit(3.8, "mm"),
      legend.key.width = unit(6, "mm"),
      plot.title = element_text(size = 7.4, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 6.1, colour = "grey35", hjust = 0),
      plot.caption = element_text(size = 5.8, colour = "grey35", hjust = 0),
      panel.grid.major.y = element_line(linewidth = 0.16, colour = "grey90"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )
)

pal <- c(
  blue = "#2B6CB0",
  blue2 = "#6BAED6",
  orange = "#D88C3A",
  red = "#C44E52",
  green = "#4C9A6A",
  grey = "#8A8F98",
  dark = "#2F3437",
  verylight = "#F8FAFC"
)

dataset_pal <- c("assist_09" = unname(pal["blue"]), "assist_17" = unname(pal["green"]), "junyi" = unname(pal["grey"]))
method_pal <- c(
  "Self" = unname(pal["grey"]),
  "Rand" = "#B8A06A",
  "Deg-rand" = unname(pal["orange"]),
  "Best CRG" = unname(pal["blue"]),
  "Seq-only" = unname(pal["green"]),
  "Fused" = unname(pal["blue"]),
  "no_CRG" = unname(pal["blue"]),
  "no_LCRF" = unname(pal["red"])
)

read_csv_silent <- function(path) {
  if (!file.exists(path)) {
    warning("missing file: ", path)
    return(tibble())
  }
  as_tibble(utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE))
}

save_pdf <- function(plot, name, width_mm = 180, height_mm = 115) {
  width <- width_mm / 25.4
  height <- height_mm / 25.4
  ggsave(
    file.path(out_dir, paste0(name, ".pdf")),
    plot,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
}

compact_percent <- function(x) sprintf("%.0f", 100 * as.numeric(x))

# Figure 2 contract:
# Conclusion: the three datasets expose different concept-coverage regimes,
# and train-stage CRG routes provide route retrieval beyond random/self controls.
# Panels: dataset cards, history-to-query retrieval, global transition retrieval.
plot_fig2 <- function() {
  cards <- read_csv_silent(file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "data_story", "dataset_story_cards_core3.csv")) %>%
    filter(dataset %in% dataset_levels) %>%
    mutate(dataset = factor(dataset, levels = dataset_levels)) %>%
    arrange(dataset)

  hist <- bind_rows(lapply(dataset_levels, function(ds) {
    path <- file.path(repo_root, "results", "main_problem_experiments_20260523", ds, "main_problem_exp1_history_to_query_route_summary.csv")
    read_csv_silent(path)
  })) %>%
    filter(group == "direct_unseen_bridgeable", method %in% c("random", "seq-only", "fused CRG")) %>%
    mutate(
      dataset = factor(dataset, levels = dataset_levels),
      method = recode(method, "random" = "Rand", "seq-only" = "Seq-only", "fused CRG" = "Fused"),
      method = factor(method, levels = c("Rand", "Seq-only", "Fused"))
    )

  global <- read_csv_silent(file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "crg_retrieval", "crg_retrieval_full_core3.csv")) %>%
    filter(dataset %in% dataset_levels) %>%
    mutate(
      method = case_when(
        variant == "CRG_self_only" ~ "Self",
        variant == "CRG_degree_random" ~ "Deg-rand",
        role == "random_or_uniform" ~ "Rand",
        role == "crg_candidate" ~ "Best CRG",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(method)) %>%
    group_by(dataset, method) %>%
    slice_max(order_by = `hit@10`, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      dataset = factor(dataset, levels = dataset_levels),
      method = factor(method, levels = c("Self", "Rand", "Deg-rand", "Best CRG"))
    )

  card_long <- cards %>%
    transmute(
      dataset,
      Single = compact_percent(single_concept_rate),
      `Item edge` = compact_percent(item_edge_density),
      `Seq edge` = compact_percent(seq_edge_density),
      Unseen = compact_percent(direct_unseen_rate),
      Bridge = compact_percent(bridge_only_rate),
      `Hist med` = sprintf("%.0f", history_len_median)
    ) %>%
    pivot_longer(-dataset, names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric, levels = c("Single", "Item edge", "Seq edge", "Unseen", "Bridge", "Hist med")))

  p_cards <- ggplot(card_long, aes(metric, dataset)) +
    geom_tile(fill = pal["verylight"], color = "white", linewidth = 1.2) +
    geom_text(aes(label = value), size = 2.55, fontface = "bold", colour = pal["dark"]) +
    geom_text(aes(label = metric, y = as.numeric(dataset) - 0.34), size = 1.75, colour = "grey45") +
    scale_y_discrete(limits = rev(dataset_levels)) +
    labs(x = NULL, y = NULL, title = "Dataset coverage cards", subtitle = "Percent values omit the % sign; history is median interactions.") +
    theme(axis.text.x = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank(), plot.margin = margin(3, 3, 3, 3))

  p_hist <- ggplot(hist, aes(dataset, hit10, fill = method)) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62) +
    scale_fill_manual(values = method_pal) +
    scale_y_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0, 0.08))) +
    labs(x = NULL, y = "Hit@10", title = "History-to-query route retrieval", subtitle = "Direct-unseen bridgeable query concepts.") +
    theme(legend.position = "bottom", panel.grid.major.x = element_blank())

  p_global <- ggplot(global, aes(dataset, `hit@10`, fill = method)) +
    geom_col(position = position_dodge(width = 0.74), width = 0.64) +
    scale_fill_manual(values = method_pal) +
    scale_y_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0, 0.08))) +
    labs(x = NULL, y = "Hit@10", title = "Held-out transition retrieval", subtitle = "Best CRG against self/random controls.") +
    theme(legend.position = "bottom", panel.grid.major.x = element_blank())

  fig <- p_cards / (p_hist | p_global) + plot_layout(heights = c(0.82, 1.18))
  save_pdf(fig, "fig2_nature_data_and_crg_retrieval", 180, 125)
}

# Figure 3 contract:
# Conclusion: prediction gains concentrate in selected coverage-conditioned
# subgroups; Junyi has smaller prediction-level gaps.
# Archetype: quantitative forest plot.
plot_fig3 <- function() {
  cov <- read_csv_silent(file.path(repo_root, "results", "main_problem_experiments_20260523", "coverage_conditioned_prediction_table.csv")) %>%
    filter(dataset %in% dataset_levels, subgroup %in% c("direct_unseen_bridgeable", "weak_direct_evidence", "high_route_mass")) %>%
    mutate(
      dataset = factor(dataset, levels = dataset_levels),
      subgroup_label = recode(
        subgroup,
        "direct_unseen_bridgeable" = "unseen+bridge",
        "weak_direct_evidence" = "weak-direct",
        "high_route_mass" = "high-route"
      ),
      comparison_label = recode(comparison, "full_vs_no_CRG" = "no_CRG", "full_vs_no_LCRF" = "no_LCRF"),
      comparison_label = factor(comparison_label, levels = c("no_CRG", "no_LCRF")),
      y_label = paste0(dataset, " / ", subgroup_label),
      y_label = factor(y_label, levels = rev(unique(paste0(dataset, " / ", subgroup_label)))),
      dataset_alpha = if_else(dataset == "junyi", 0.45, 1.0)
    )

  p_bce <- ggplot(cov, aes(delta_bce, y_label, color = comparison_label, alpha = dataset_alpha)) +
    geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey45") +
    geom_errorbar(aes(xmin = delta_bce_ci_low, xmax = delta_bce_ci_high), orientation = "y", width = 0.18, linewidth = 0.5, position = position_dodge(width = 0.48)) +
    geom_point(size = 1.8, position = position_dodge(width = 0.48)) +
    scale_color_manual(values = method_pal) +
    scale_alpha_identity() +
    scale_x_continuous(labels = number_format(accuracy = 0.001), expand = expansion(mult = c(0.08, 0.08))) +
    labs(
      x = expression(Delta*"BCE (variant - Full; positive favors Full)"),
      y = NULL,
      title = "Coverage-conditioned prediction",
      subtitle = "Forest plot of BCE gain with 500-sample bootstrap 95% CI."
    ) +
    theme(legend.position = "bottom", panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.2))

  p_auc <- ggplot(cov, aes(delta_auc, y_label, color = comparison_label, alpha = dataset_alpha)) +
    geom_vline(xintercept = 0, linewidth = 0.35, colour = "grey45") +
    geom_errorbar(aes(xmin = delta_auc_ci_low, xmax = delta_auc_ci_high), orientation = "y", width = 0.18, linewidth = 0.5, position = position_dodge(width = 0.48)) +
    geom_point(size = 1.8, position = position_dodge(width = 0.48)) +
    scale_color_manual(values = method_pal) +
    scale_alpha_identity() +
    scale_x_continuous(labels = number_format(accuracy = 0.001), expand = expansion(mult = c(0.08, 0.08))) +
    labs(
      x = expression(Delta*"AUC (Full - variant)"),
      y = NULL,
      title = "AUC check",
      subtitle = "AUC and BCE do not always move identically across subgroups."
    ) +
    theme(legend.position = "none", panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.2))

  n_lab <- cov %>%
    distinct(dataset, subgroup_label, y_label, n_eval) %>%
    mutate(label = paste0("n=", scales::comma(n_eval)))

  p_n <- ggplot(n_lab, aes(x = 1, y = y_label)) +
    geom_text(aes(label = label), size = 2.2, hjust = 0, colour = "grey35") +
    xlim(1, 2.15) +
    labs(x = NULL, y = NULL, title = "Sample size", subtitle = "Subgroup n.") +
    theme_void(base_size = 6.8) +
    theme(plot.title = element_text(size = 7.4, face = "bold"), plot.subtitle = element_text(size = 6.1, colour = "grey35"))

  fig <- (p_bce | p_auc | p_n) + plot_layout(widths = c(1.35, 1.05, 0.42))
  save_pdf(fig, "fig3_nature_coverage_conditioned_prediction", 183, 108)
}

plot_fig2()
plot_fig3()
