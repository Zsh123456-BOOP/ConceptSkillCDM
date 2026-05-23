#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "results/crg_lcrf_core3_final_20260520"
out_dir <- if (length(args) >= 2) args[[2]] else file.path(root, "paper_figures_nature")
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
  light = "#EEF2F7",
  verylight = "#F8FAFC"
)

method_pal <- c(
  "Self" = unname(pal["grey"]),
  "Rand" = "#B8A06A",
  "Deg-rand" = unname(pal["orange"]),
  "Best CRG" = unname(pal["blue"]),
  "Evidence" = unname(pal["blue"]),
  "Seq-shuf" = "#9B8CCB",
  "Self-only" = unname(pal["grey"])
)

dataset_pal <- c("assist_09" = unname(pal["blue"]), "assist_17" = unname(pal["green"]), "junyi" = unname(pal["grey"]))

read_csv_silent <- function(path) {
  if (!file.exists(path)) {
    warning("missing file: ", path)
    return(tibble())
  }
  suppressMessages(readr::read_csv(path, show_col_types = FALSE))
}

save_pub <- function(plot, name, width_mm = 180, height_mm = 115, dpi = 450) {
  width <- width_mm / 25.4
  height <- height_mm / 25.4
  pdf_path <- file.path(out_dir, paste0(name, ".pdf"))
  png_path <- file.path(out_dir, paste0(name, ".png"))
  svg_path <- file.path(out_dir, paste0(name, ".svg"))
  ggsave(pdf_path, plot, width = width, height = height, units = "in", device = cairo_pdf, bg = "white")
  if (requireNamespace("ragg", quietly = TRUE)) {
    ragg::agg_png(png_path, width = width, height = height, units = "in", res = dpi, background = "white")
    print(plot)
    dev.off()
  } else {
    ggsave(png_path, plot, width = width, height = height, units = "in", dpi = dpi, bg = "white")
  }
  if (requireNamespace("svglite", quietly = TRUE)) {
    svglite::svglite(svg_path, width = width, height = height, bg = "white")
    print(plot)
    dev.off()
  }
}

panel_label <- function(label) {
  ggplot() +
    annotate("text", 0, 0, label = label, fontface = "bold", size = 2.8, hjust = 0) +
    xlim(0, 1) + ylim(-1, 1) +
    theme_void()
}

compact_percent <- function(x) sprintf("%.0f", 100 * as.numeric(x))

# -------------------------------------------------------------------------
# Figure 2: dataset cards + CRG retrieval lift.
# Claim: the core datasets expose concept evidence gaps, and train-stage CRG
# routes retrieve held-out transitions better than self/random controls.
# -------------------------------------------------------------------------
cards <- read_csv_silent(file.path(root, "data_story", "dataset_story_cards_core3.csv")) %>%
  filter(dataset %in% dataset_levels) %>%
  mutate(dataset = factor(dataset, levels = dataset_levels)) %>%
  arrange(dataset)

retr <- read_csv_silent(file.path(root, "crg_retrieval", "crg_retrieval_full_core3.csv")) %>%
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

if (nrow(cards) > 0 && nrow(retr) > 0) {
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
    labs(x = NULL, y = NULL, title = "Dataset evidence-gap cards", subtitle = "Percent values omit the % sign; history is median interactions.") +
    theme(
      axis.text.x = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      plot.margin = margin(3, 3, 3, 3)
    )

  p_retr <- ggplot(retr, aes(`hit@10`, method, color = method)) +
    geom_line(aes(group = dataset), color = "grey84", linewidth = 0.45) +
    geom_point(size = 2.2) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = method_pal) +
    scale_x_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0.02, 0.08))) +
    labs(x = "Held-out transition retrieval, Hit@10", y = NULL, title = "CRG retrieval lift", subtitle = "Best train-stage CRG route vs self/random controls.") +
    theme(legend.position = "none", plot.margin = margin(3, 3, 3, 3))

  fig2 <- (p_cards / p_retr) + plot_layout(heights = c(0.92, 1.08))
  save_pub(fig2, "fig2_nature_data_and_crg_retrieval", 180, 118)
}

# -------------------------------------------------------------------------
# Figure 3: CRG support corruption.
# Claim: prediction-level support dependence is dataset-dependent, strongest
# for assist_17 and weaker for Junyi.
# -------------------------------------------------------------------------
support <- read_csv_silent(file.path(root, "crg_support_audit", "crg_support_gap_audit_core3.csv")) %>%
  filter(dataset %in% dataset_levels, subgroup == "all", corruption_ratio > 0) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    type = case_when(
      corruption_type == "evidence_support_corruption" ~ "Evidence",
      corruption_type == "degree_matched_random_support" ~ "Deg-rand",
      corruption_type == "sequence_shuffled_support" ~ "Seq-shuf",
      corruption_type == "self_only_fallback" ~ "Self-only",
      TRUE ~ corruption_type
    ),
    type = factor(type, levels = c("Evidence", "Deg-rand", "Seq-shuf", "Self-only"))
  ) %>%
  group_by(dataset, type, corruption_ratio) %>%
  summarise(
    auc_drop = mean(auc_drop_from_clean, na.rm = TRUE),
    bce_inc = mean(bce_increase_from_clean, na.rm = TRUE),
    gap_auc = mean(evidence_minus_degree_random_auc_drop, na.rm = TRUE),
    .groups = "drop"
  )

if (nrow(support) > 0) {
  p_auc <- ggplot(support, aes(corruption_ratio, auc_drop, color = type)) +
    geom_line(linewidth = 0.62) +
    geom_point(size = 1.45) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = method_pal) +
    scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0.25, 0.5, 0.75, 1)) +
    labs(x = NULL, y = "AUC drop", title = "Support corruption curves", subtitle = "Inference-time perturbation of CRG support; larger drop indicates stronger support dependence.") +
    theme(legend.position = "bottom")

  p_bce <- ggplot(support, aes(corruption_ratio, bce_inc, color = type)) +
    geom_line(linewidth = 0.62) +
    geom_point(size = 1.45) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = method_pal) +
    scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0.25, 0.5, 0.75, 1)) +
    labs(x = "Corruption ratio", y = "BCE increase", title = NULL) +
    theme(legend.position = "none")

  gap <- support %>%
    filter(type == "Evidence", corruption_ratio == 1) %>%
    mutate(
      status = case_when(
        dataset == "assist_17" ~ "evidence gap",
        dataset == "assist_09" ~ "support-only",
        TRUE ~ "weak"
      )
    )

  p_gap <- ggplot(gap, aes(dataset, gap_auc, fill = dataset)) +
    geom_col(width = 0.58) +
    geom_hline(yintercept = 0, linewidth = 0.25, colour = "grey30") +
    geom_text(aes(label = status), size = 2.35, vjust = ifelse(gap$gap_auc >= 0, -0.45, 1.25), colour = "grey20") +
    scale_fill_manual(values = dataset_pal) +
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.22))) +
    coord_cartesian(clip = "off") +
    labs(x = NULL, y = "Evidence - deg-rand\nAUC-drop gap", title = "Dataset-dependent evidence gap") +
    theme(legend.position = "none", panel.grid.major.x = element_blank())

  fig3 <- (p_auc / p_bce / p_gap) + plot_layout(heights = c(1.0, 0.9, 0.75))
  save_pub(fig3, "fig3_nature_crg_support_corruption", 180, 150)
}

# -------------------------------------------------------------------------
# Figure 4: LCRF learner-state counterfactual.
# Claim: assist_09/assist_17 show state-dependence; Junyi is weak and kept grey.
# -------------------------------------------------------------------------
cf <- read_csv_silent(file.path(root, "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv")) %>%
  filter(dataset %in% dataset_levels) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    variant_label = case_when(
      variant == "no_filter" ~ "No filter",
      variant == "mean_state" ~ "Mean state",
      variant == "shuffle_state" ~ "Shuffle state",
      TRUE ~ variant
    ),
    variant_label = factor(variant_label, levels = c("No filter", "Mean state", "Shuffle state")),
    fill_key = if_else(dataset == "junyi", "junyi", as.character(variant_label))
  )

if (nrow(cf) > 0) {
  cf_pal <- c("No filter" = unname(pal["blue2"]), "Mean state" = unname(pal["orange"]), "Shuffle state" = unname(pal["red"]), "junyi" = "#B8BDC5")

  p4 <- ggplot(cf, aes(variant_label, auc_drop_from_full, fill = fill_key)) +
    geom_col(width = 0.62) +
    facet_wrap(~ dataset, nrow = 1) +
    geom_text(
      data = cf %>% filter(dataset == "junyi", variant == "shuffle_state"),
      aes(x = variant_label, label = "weak", y = auc_drop_from_full + 0.012),
      inherit.aes = FALSE,
      size = 2.4,
      colour = "grey35"
    ) +
    scale_fill_manual(values = cf_pal) +
    scale_y_continuous(limits = c(0, max(cf$auc_drop_from_full, na.rm = TRUE) * 1.18), expand = expansion(mult = c(0, 0.03))) +
    labs(x = NULL, y = expression(Delta*"AUC from full"), title = "Learner-state counterfactual", subtitle = "Shared y-axis prevents weak Junyi effects from appearing visually inflated.") +
    theme(legend.position = "none", axis.text.x = element_text(angle = 35, hjust = 1))

  save_pub(p4, "fig4_nature_lcrf_counterfactual_delta", 180, 78)
}

# -------------------------------------------------------------------------
# Figure 5: same-query posterior mechanism.
# Claim: under identical CRG support, learners receive distinct posterior
# route weights and different prediction shifts.
# -------------------------------------------------------------------------
sameq <- read_csv_silent(file.path(root, "lcrf_same_query", "lcrf_same_query_annotated_core3.csv"))
twostu <- read_csv_silent(file.path(root, "lcrf_same_query", "lcrf_two_student_path_case_core3.csv"))

if (nrow(sameq) > 0) {
  case_df <- sameq %>%
    filter(dataset == "assist_17") %>%
    mutate(support_concept_name = if_else(is.na(support_concept_name) | support_concept_name == "", paste0("C", support_concept_id), support_concept_name))
  if (nrow(case_df) == 0) {
    case_df <- sameq %>% filter(dataset == "assist_09")
  }
  case_id <- case_df %>%
    count(case_id, sort = TRUE) %>%
    slice(1) %>%
    pull(case_id)
  case_df <- case_df %>% filter(case_id == !!case_id)

  learner_order <- case_df %>%
    distinct(learner_id_anonymized, pred_full, true_label, query_mastery, query_recent_mastery) %>%
    arrange(desc(pred_full)) %>%
    pull(learner_id_anonymized)
  support_order <- case_df %>%
    group_by(support_concept_name) %>%
    summarise(global_support_prob = mean(global_support_prob, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(global_support_prob)) %>%
    pull(support_concept_name)
  shown_learners <- head(learner_order, 10)
  shown_support <- head(support_order, 10)
  plot_df <- case_df %>%
    filter(learner_id_anonymized %in% shown_learners, support_concept_name %in% shown_support) %>%
    mutate(
      learner_id_anonymized = factor(learner_id_anonymized, levels = rev(shown_learners)),
      support_concept_name = factor(support_concept_name, levels = shown_support)
    )

  p_support <- case_df %>%
    filter(support_concept_name %in% shown_support) %>%
    group_by(support_concept_name) %>%
    summarise(global_support_prob = mean(global_support_prob, na.rm = TRUE), .groups = "drop") %>%
    mutate(support_concept_name = factor(support_concept_name, levels = shown_support)) %>%
    ggplot(aes(support_concept_name, global_support_prob)) +
    geom_col(fill = pal["blue"], width = 0.65) +
    labs(x = NULL, y = "CRG prior", title = "Fixed CRG support") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid.major.x = element_blank())

  p_heat <- ggplot(plot_df, aes(support_concept_name, learner_id_anonymized, fill = posterior_minus_global)) +
    geom_tile(color = "white", linewidth = 0.28) +
    scale_fill_gradient2(low = "#3B6FB6", mid = "white", high = "#D65F5F", midpoint = 0, name = "Posterior\nminus prior") +
    labs(x = NULL, y = NULL, title = "Learner-conditioned posterior shift") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "right")

  learner_annot <- case_df %>%
    distinct(learner_id_anonymized, query_mastery, query_recent_mastery, true_label, pred_full) %>%
    filter(learner_id_anonymized %in% shown_learners) %>%
    mutate(
      learner_id_anonymized = factor(learner_id_anonymized, levels = rev(shown_learners)),
      label = sprintf("m %.2f | r %.2f | y %.0f | p %.2f", query_mastery, query_recent_mastery, true_label, pred_full)
    )
  p_annot <- ggplot(learner_annot, aes(0, learner_id_anonymized, label = label)) +
    geom_text(hjust = 0, size = 2.15, colour = "grey25") +
    xlim(0, 1) +
    labs(x = NULL, y = NULL, title = "Learner state") +
    theme_void() +
    theme(plot.title = element_text(size = 7.4, face = "bold"))

  shift_df <- case_df %>%
    distinct(learner_id_anonymized, pred_global, pred_no_filter, pred_full, true_label) %>%
    filter(learner_id_anonymized %in% shown_learners) %>%
    mutate(learner_id_anonymized = factor(learner_id_anonymized, levels = rev(shown_learners)))
  p_shift <- ggplot(shift_df) +
    geom_segment(aes(x = pred_global, xend = pred_full, y = learner_id_anonymized, yend = learner_id_anonymized), color = "grey76", linewidth = 0.45) +
    geom_point(aes(pred_global, learner_id_anonymized), color = pal["grey"], size = 1.55) +
    geom_point(aes(pred_full, learner_id_anonymized, color = factor(true_label)), size = 1.8) +
    scale_color_manual(values = c("0" = unname(pal["red"]), "1" = unname(pal["green"])), name = "Label") +
    labs(x = "Prediction", y = NULL, title = "Global to LCRF prediction shift") +
    theme(legend.position = "bottom")

  if (nrow(twostu) > 0) {
    two_df <- twostu %>%
      filter(dataset == unique(case_df$dataset)[1]) %>%
      mutate(support_concept_name = if_else(is.na(support_concept_name) | support_concept_name == "", paste0("C", support_concept_id), support_concept_name)) %>%
      group_by(learner_id_anonymized) %>%
      slice_max(order_by = posterior_prob, n = 3, with_ties = FALSE) %>%
      ungroup() %>%
      mutate(learner_id_anonymized = factor(learner_id_anonymized))
    p_two <- ggplot(two_df, aes(reorder(support_concept_name, posterior_prob), posterior_prob, fill = learner_id_anonymized)) +
      geom_col(position = position_dodge(width = 0.72), width = 0.65) +
      coord_flip() +
      scale_fill_manual(values = c(unname(pal["blue"]), unname(pal["red"]), unname(pal["green"]), unname(pal["orange"]))) +
      labs(x = NULL, y = "Top posterior route", title = "Two-student route contrast", subtitle = "Same query/support; top posterior routes diverge.") +
      theme(legend.position = "bottom", panel.grid.major.y = element_blank())
  } else {
    p_two <- plot_spacer()
  }

  fig5 <- ((p_support | p_annot) / (p_heat | p_shift) / p_two) +
    plot_layout(widths = c(1.25, 1.1), heights = c(0.55, 1.55, 0.85), guides = "collect")
  save_pub(fig5, "fig5_nature_lcrf_same_query_posterior", 180, 175)
}

message("Nature-style CRG/LCRF figures written to: ", normalizePath(out_dir, winslash = "/"))
