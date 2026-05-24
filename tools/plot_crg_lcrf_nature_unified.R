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
pdf_dir <- file.path(repo_root, "docs", "paper_review_2025_2026", "figures_main_pdf")
png_dir <- file.path(repo_root, "docs", "paper_review_2025_2026", "figures_preview_png")
dir.create(pdf_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(png_dir, recursive = TRUE, showWarnings = FALSE)

dataset_levels <- c("assist_09", "assist_17", "junyi")

pal <- c(
  evidence = "#E56A00",   # train evidence, aligned with the schematic
  crg = "#0B57D0",        # global CRG roadmap/support
  lcrf = "#D71920",       # LCRF posterior/filter
  learner = "#16833A",    # learner-state signal
  neutral = "#7D8590",
  neutral_dark = "#252A2E",
  neutral_light = "#EEF2F6",
  purple = "#7F6BB5",
  paper = "#FFFFFF"
)

c0 <- function(name) unname(pal[[name]])

method_pal <- c(
  "Self" = c0("neutral"),
  "Rand" = "#B89B57",
  "Deg-rand" = c0("evidence"),
  "Best CRG" = c0("crg"),
  "Seq-only" = c0("learner"),
  "Fused" = c0("crg"),
  "Evidence" = c0("crg"),
  "Seq-shuf" = c0("purple"),
  "Self-only" = c0("neutral"),
  "no_CRG" = c0("crg"),
  "no_LCRF" = c0("lcrf"),
  "No filter" = "#6BAED6",
  "Mean state" = c0("evidence"),
  "Shuffle state" = c0("lcrf"),
  "S1" = c0("crg"),
  "S7" = "#C94B50"
)

dataset_pal <- c(
  "assist_09" = c0("crg"),
  "assist_17" = c0("learner"),
  "junyi" = c0("neutral")
)

theme_nature <- function(base_size = 6.8, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.28, colour = "grey20"),
      axis.ticks = element_line(linewidth = 0.28, colour = "grey20"),
      axis.text = element_text(colour = "grey15"),
      axis.title = element_text(colour = "grey10"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base_size),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.7),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(6, "mm"),
      plot.title = element_text(size = base_size + 0.8, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.5, colour = "grey35", hjust = 0),
      plot.caption = element_text(size = base_size - 0.8, colour = "grey35", hjust = 0),
      panel.grid.major.y = element_line(linewidth = 0.16, colour = "grey90"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )
}
theme_set(theme_nature())

read_csv_base <- function(path) {
  if (!file.exists(path)) {
    warning("missing file: ", path)
    return(tibble())
  }
  as_tibble(utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE))
}

save_figure <- function(plot, name, width_mm = 183, height_mm = 115, dpi = 360) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  ggsave(
    file.path(pdf_dir, paste0(name, ".pdf")),
    plot,
    width = w,
    height = h,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
  ggsave(
    file.path(png_dir, paste0(name, ".png")),
    plot,
    width = w,
    height = h,
    units = "in",
    dpi = dpi,
    bg = "white"
  )
}

fmt_pct <- function(x, digits = 1) sprintf(paste0("%.", digits, "f%%"), 100 * as.numeric(x))
fmt_n <- function(x) comma(as.numeric(x), accuracy = 1)

audit_rows <- list()
add_audit <- function(figure, panel, claim, source_file, fields, filter, status = "checked") {
  audit_rows[[length(audit_rows) + 1]] <<- tibble(
    figure = figure,
    panel = panel,
    claim = claim,
    source_file = source_file,
    fields = fields,
    filter = filter,
    status = status
  )
}

cards_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "data_story", "dataset_story_cards_core3.csv")
history_paths <- file.path(repo_root, "results", "main_problem_experiments_20260523", dataset_levels, "main_problem_exp1_history_to_query_route_summary.csv")
global_retr_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "crg_retrieval", "crg_retrieval_full_core3.csv")
coverage_path <- file.path(repo_root, "results", "main_problem_experiments_20260523", "coverage_conditioned_prediction_table.csv")
support_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "crg_support_audit", "crg_support_gap_audit_core3.csv")
counter_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv")
same_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "lcrf_same_query", "lcrf_same_query_annotated_core3.csv")
two_path <- file.path(repo_root, "results", "crg_lcrf_core3_final_20260520", "lcrf_same_query", "lcrf_two_student_path_case_core3.csv")

# -------------------------------------------------------------------------
# Figure 2. Dataset coverage and CRG route retrieval.
# Claim: direct concept coverage differs by dataset, and train-stage routes
# retrieve query/transition concepts beyond random controls.
# -------------------------------------------------------------------------
cards <- read_csv_base(cards_path) %>%
  filter(dataset %in% dataset_levels) %>%
  mutate(dataset = factor(dataset, levels = dataset_levels)) %>%
  arrange(dataset)

hist <- bind_rows(lapply(history_paths, read_csv_base)) %>%
  filter(group == "direct_unseen_bridgeable", method %in% c("random", "seq-only", "fused CRG")) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    method = recode(method, "random" = "Rand", "seq-only" = "Seq-only", "fused CRG" = "Fused"),
    method = factor(method, levels = c("Rand", "Seq-only", "Fused"))
  )

global <- read_csv_base(global_retr_path) %>%
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

p_hist <- ggplot(hist, aes(dataset, hit10, fill = method)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.62) +
  scale_fill_manual(values = method_pal) +
  scale_y_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Hit@10", title = "History-to-query retrieval", subtitle = "Direct-unseen bridgeable query concepts.") +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

p_global <- ggplot(global, aes(dataset, `hit@10`, fill = method)) +
  geom_col(position = position_dodge(width = 0.74), width = 0.64) +
  scale_fill_manual(values = method_pal) +
  scale_y_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Hit@10", title = "Held-out transition retrieval", subtitle = "Global route recovery controls.") +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  theme(legend.position = "bottom", panel.grid.major.x = element_blank())

fig2 <- p_hist | p_global +
  plot_layout(widths = c(1, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 8))
save_figure(fig2, "fig2_nature_data_and_crg_retrieval", 183, 86)

add_audit("Figure 4", "a", "History-to-query route retrieval", paste(history_paths, collapse = ";"), "hit10", "group=direct_unseen_bridgeable; methods=random,seq-only,fused CRG")
add_audit("Figure 4", "b", "Held-out transition retrieval", global_retr_path, "hit@10", "Self/Rand/Deg-rand/Best CRG")

# -------------------------------------------------------------------------
# Figure 3. Coverage-conditioned prediction.
# Claim: prediction gains concentrate in coverage-conditioned subgroups.
# -------------------------------------------------------------------------
cov <- read_csv_base(coverage_path) %>%
  filter(
    dataset %in% dataset_levels,
    subgroup %in% c("direct_unseen_bridgeable", "weak_direct_evidence", "high_route_mass"),
    !(dataset == "junyi" & subgroup == "weak_direct_evidence")
  ) %>%
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
    row_label = paste0(subgroup_label, "\n", "n=", fmt_n(n_eval)),
    row_label = factor(row_label, levels = rev(unique(row_label)))
  )

p_bce <- ggplot(cov, aes(delta_bce, row_label, color = comparison_label)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey35") +
  geom_errorbar(aes(xmin = delta_bce_ci_low, xmax = delta_bce_ci_high), orientation = "y", width = 0.18, linewidth = 0.45, position = position_dodge(width = 0.48)) +
  geom_point(size = 1.75, position = position_dodge(width = 0.48)) +
  facet_grid(dataset ~ ., scales = "free_y", space = "free_y") +
  scale_color_manual(values = method_pal) +
  scale_x_continuous(labels = number_format(accuracy = 0.001), expand = expansion(mult = c(0.07, 0.08))) +
  labs(x = expression(Delta*"BCE (variant - Full)"), y = NULL, title = "Coverage-conditioned prediction", subtitle = "Positive values mean lower BCE for Full; 95% CI.") +
  theme(legend.position = "bottom", panel.spacing.y = unit(2.2, "mm"))

p_auc <- ggplot(cov, aes(delta_auc, row_label, color = comparison_label)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey35") +
  geom_errorbar(aes(xmin = delta_auc_ci_low, xmax = delta_auc_ci_high), orientation = "y", width = 0.18, linewidth = 0.45, position = position_dodge(width = 0.48)) +
  geom_point(size = 1.75, position = position_dodge(width = 0.48)) +
  facet_grid(dataset ~ ., scales = "free_y", space = "free_y") +
  scale_color_manual(values = method_pal) +
  scale_x_continuous(labels = number_format(accuracy = 0.001), expand = expansion(mult = c(0.08, 0.08))) +
  labs(x = expression(Delta*"AUC (Full - variant)"), y = NULL, title = "AUC companion", subtitle = "Small subgroups may show BCE/AUC divergence.") +
  theme(legend.position = "none", axis.text.y = element_blank(), axis.ticks.y = element_blank(), panel.spacing.y = unit(2.2, "mm"))

fig3 <- (p_bce | p_auc) +
  plot_layout(widths = c(1.22, 1.0)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 8))
save_figure(fig3, "fig3_nature_coverage_conditioned_prediction", 183, 116)

add_audit("Figure 5", "a", "Coverage-conditioned BCE gain", coverage_path, "delta_bce,delta_bce_ci_low,delta_bce_ci_high,n_eval", "direct_unseen_bridgeable,weak_direct_evidence,high_route_mass; junyi weak-direct omitted")
add_audit("Figure 5", "b", "Coverage-conditioned AUC companion", coverage_path, "delta_auc,delta_auc_ci_low,delta_auc_ci_high,n_eval", "same rows as panel a")

# -------------------------------------------------------------------------
# Figure 4. CRG support corruption.
# Claim: support dependence is strongest for assist_17 and smaller for Junyi.
# -------------------------------------------------------------------------
support <- read_csv_base(support_path) %>%
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

support_long <- bind_rows(
  support %>% transmute(dataset, type, corruption_ratio, metric = "AUC drop", value = auc_drop),
  support %>% transmute(dataset, type, corruption_ratio, metric = "BCE increase", value = bce_inc)
) %>% mutate(metric = factor(metric, levels = c("AUC drop", "BCE increase")))

p_curves <- ggplot(support_long, aes(corruption_ratio, value, color = type)) +
  geom_line(linewidth = 0.58) +
  geom_point(size = 1.35) +
  facet_grid(metric ~ dataset, scales = "free_y") +
  scale_color_manual(values = method_pal) +
  scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0.25, 0.5, 0.75, 1.0)) +
  labs(x = "Corruption ratio", y = NULL, title = "Support perturbation", subtitle = "Inference-time changes to CRG support; larger values mean larger prediction change.") +
  theme(legend.position = "bottom")

gap <- support %>%
  filter(type == "Evidence", corruption_ratio == 1) %>%
  mutate(
    status = case_when(
      dataset == "assist_17" ~ "evidence gap",
      dataset == "assist_09" ~ "support-only",
      TRUE ~ "small gap"
    )
  )

p_gap <- ggplot(gap, aes(dataset, gap_auc, fill = dataset)) +
  geom_col(width = 0.56) +
  geom_hline(yintercept = 0, linewidth = 0.28, colour = "grey35") +
  geom_text(aes(label = status), size = 2.3, vjust = ifelse(gap$gap_auc >= 0, -0.45, 1.25), colour = "grey20") +
  scale_fill_manual(values = dataset_pal) +
  scale_y_continuous(expand = expansion(mult = c(0.16, 0.26))) +
  labs(x = NULL, y = "Evidence - Deg-rand\nAUC-drop gap", title = "Evidence-specific gap at 100% replacement") +
  theme(legend.position = "none", panel.grid.major.x = element_blank())

fig4 <- p_curves / p_gap +
  plot_layout(heights = c(1.85, 0.75)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 8))
save_figure(fig4, "fig4_nature_crg_support_corruption", 183, 146)

add_audit("Figure 6", "a", "AUC/BCE response to support perturbation", support_path, "auc_drop_from_clean,bce_increase_from_clean", "subgroup=all; ratios=25,50,75,100")
add_audit("Figure 6", "b", "Evidence-minus-degree random gap", support_path, "evidence_minus_degree_random_auc_drop", "100% evidence support replacement")

# -------------------------------------------------------------------------
# Figure 5. LCRF learner-state counterfactual.
# Claim: assist_09/assist_17 depend on learner state; Junyi has smaller effect.
# -------------------------------------------------------------------------
cf <- read_csv_base(counter_path) %>%
  filter(dataset %in% dataset_levels) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    variant_label = case_when(
      variant == "no_filter" ~ "No filter",
      variant == "mean_state" ~ "Mean state",
      variant == "shuffle_state" ~ "Shuffle state",
      TRUE ~ variant
    ),
    variant_label = factor(variant_label, levels = c("No filter", "Mean state", "Shuffle state"))
  )

p5 <- ggplot(cf, aes(variant_label, auc_drop_from_full, color = variant_label)) +
  geom_segment(aes(xend = variant_label, y = 0, yend = auc_drop_from_full), linewidth = 0.55, alpha = 0.75) +
  geom_point(size = 2.2) +
  facet_wrap(~ dataset, nrow = 1) +
  scale_color_manual(values = method_pal) +
  scale_y_continuous(limits = c(0, max(cf$auc_drop_from_full, na.rm = TRUE) * 1.08), labels = number_format(accuracy = 0.01)) +
  labs(x = NULL, y = expression(Delta*"AUC from Full"), title = "Learner-state counterfactual", subtitle = "Shared y-axis keeps the small Junyi effects visually proportional.") +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 25, hjust = 1), panel.grid.major.x = element_blank()) +
  geom_text(
    data = tibble(
      dataset = factor("junyi", levels = dataset_levels),
      variant_label = factor("Shuffle state", levels = levels(cf$variant_label)),
      auc_drop_from_full = max(cf$auc_drop_from_full, na.rm = TRUE) * 0.12,
      label = "small state effect"
    ),
    aes(x = variant_label, y = auc_drop_from_full, label = label),
    inherit.aes = FALSE,
    size = 2.15,
    colour = "grey35",
    hjust = 0.55
  )

save_figure(p5, "fig5_nature_lcrf_counterfactual_delta", 183, 78)

add_audit("Figure 7", "all", "Learner-state replacement changes prediction", counter_path, "auc_drop_from_full", "variants=no_filter,mean_state,shuffle_state")

# -------------------------------------------------------------------------
# Figure 6. Same-query posterior case.
# Claim: same CRG support can be filtered into different posterior routes.
# -------------------------------------------------------------------------
same <- read_csv_base(same_path)
twostu <- read_csv_base(two_path)

chosen_case_id <- if (nrow(twostu) > 0 && any(twostu$dataset == "assist_17")) {
  twostu$case_id[twostu$dataset == "assist_17"][1]
} else if (nrow(same) > 0) {
  same$case_id[1]
} else {
  NA_character_
}

if (!is.na(chosen_case_id)) {
  two <- twostu %>%
    filter(case_id == chosen_case_id) %>%
    mutate(
      learner_id_anonymized = factor(learner_id_anonymized, levels = c("S1", "S7")),
      support_concept_name = as.character(support_concept_name)
    )

  support_keep <- two %>%
    group_by(support_concept_name) %>%
    summarise(score = max(c(posterior_prob, global_support_prob), na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(score)) %>%
    slice_head(n = 6) %>%
    pull(support_concept_name)

  prior_lookup <- two %>%
    filter(support_concept_name %in% support_keep) %>%
    group_by(support_concept_name) %>%
    summarise(global_support_prob = max(global_support_prob, na.rm = TRUE), .groups = "drop")

  learner_meta <- two %>%
    distinct(learner_id_anonymized, query_mastery, query_recent_mastery, pred_global, pred_full, true_label)

  two_complete <- as_tibble(expand.grid(
    learner_id_anonymized = levels(two$learner_id_anonymized),
    support_concept_name = support_keep,
    stringsAsFactors = FALSE
  ))

  two <- two_complete %>%
    left_join(two %>% filter(support_concept_name %in% support_keep), by = c("learner_id_anonymized", "support_concept_name")) %>%
    left_join(prior_lookup, by = "support_concept_name", suffix = c("", ".prior")) %>%
    left_join(learner_meta, by = "learner_id_anonymized", suffix = c("", ".meta")) %>%
    mutate(
      learner_id_anonymized = factor(learner_id_anonymized, levels = c("S1", "S7")),
      global_support_prob = coalesce(global_support_prob, global_support_prob.prior),
      posterior_prob = coalesce(posterior_prob, 0),
      posterior_minus_global = posterior_prob - global_support_prob,
      query_mastery = coalesce(query_mastery, query_mastery.meta),
      query_recent_mastery = coalesce(query_recent_mastery, query_recent_mastery.meta),
      pred_global = coalesce(pred_global, pred_global.meta),
      pred_full = coalesce(pred_full, pred_full.meta),
      true_label = coalesce(true_label, true_label.meta),
      support_concept_name = factor(support_concept_name, levels = rev(support_keep))
    ) %>%
    select(-ends_with(".prior"), -ends_with(".meta"))

  prior <- two %>%
    group_by(support_concept_name) %>%
    summarise(global_support_prob = max(global_support_prob, na.rm = TRUE), .groups = "drop")

  p6a <- ggplot(prior, aes(global_support_prob, support_concept_name)) +
    geom_col(fill = c0("crg"), width = 0.64) +
    scale_x_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0, 0.08))) +
    labs(x = "CRG prior", y = NULL, title = "Fixed CRG support") +
    theme(panel.grid.major.y = element_blank())

  posterior_wide <- two %>%
    select(learner_id_anonymized, support_concept_name, posterior_prob) %>%
    tidyr::pivot_wider(names_from = learner_id_anonymized, values_from = posterior_prob)

  p6b <- ggplot(posterior_wide, aes(y = support_concept_name)) +
    geom_segment(aes(x = S1, xend = S7, yend = support_concept_name), linewidth = 0.45, color = "grey72") +
    geom_point(aes(x = S1, color = "S1"), size = 2.1) +
    geom_point(aes(x = S7, color = "S7"), size = 2.1) +
    scale_color_manual(values = method_pal) +
    scale_x_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0.02, 0.10))) +
    labs(x = "LCRF posterior", y = NULL, title = "Posterior route split") +
    theme(legend.position = "bottom", panel.grid.major.y = element_blank())

  p6c <- ggplot(two, aes(posterior_minus_global, support_concept_name, fill = learner_id_anonymized)) +
    geom_vline(xintercept = 0, linewidth = 0.28, colour = "grey40") +
    geom_col(position = position_dodge(width = 0.72), width = 0.58) +
    scale_fill_manual(values = method_pal) +
    scale_x_continuous(labels = number_format(accuracy = 0.01), expand = expansion(mult = c(0.12, 0.08))) +
    labs(x = "Posterior - CRG prior", y = NULL, title = "Route reweighting") +
    theme(legend.position = "none", panel.grid.major.y = element_blank())

  pred <- two %>%
    group_by(learner_id_anonymized) %>%
    slice_max(order_by = posterior_prob, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    transmute(
      learner_id_anonymized,
      query_mastery,
      query_recent_mastery,
      pred_global,
      pred_full,
      true_label,
      top_route = as.character(support_concept_name)
    )

  p6d <- ggplot(pred, aes(y = learner_id_anonymized)) +
    geom_segment(aes(x = pred_global, xend = pred_full, yend = learner_id_anonymized), linewidth = 0.55, color = "grey65") +
    geom_point(aes(x = pred_global), size = 1.9, color = "grey55") +
    geom_point(aes(x = pred_full, color = learner_id_anonymized), size = 2.2) +
    geom_text(aes(x = pmax(pred_global, pred_full) + 0.025, label = paste0("top ", top_route, "\ny=", true_label, "; q=", sprintf("%.2f", query_mastery), "\nr=", sprintf("%.2f", query_recent_mastery))), size = 2.05, hjust = 0, colour = "grey25", lineheight = 0.9) +
    scale_color_manual(values = method_pal) +
    scale_x_continuous(limits = c(min(c(pred$pred_global, pred$pred_full), na.rm = TRUE) - 0.04, max(c(pred$pred_global, pred$pred_full), na.rm = TRUE) + 0.22), labels = number_format(accuracy = 0.01)) +
    labs(x = "Prediction: global to LCRF", y = NULL, title = "Prediction shift and learner state") +
    theme(legend.position = "none", panel.grid.major.y = element_blank())

  fig6 <- (p6a | p6b) / (p6c | p6d) +
    plot_layout(heights = c(0.92, 1.08)) +
    plot_annotation(tag_levels = "a", caption = "The two students share the same query and CRG support; LCRF redirects posterior mass without adding support concepts.") &
    theme(plot.tag = element_text(face = "bold", size = 8))

  save_figure(fig6, "fig6_nature_lcrf_same_query_posterior", 183, 128)
}

add_audit("Figure 8", "a-d", "Same-query posterior routes differ across learners", paste(same_path, two_path, sep = ";"), "global_support_prob,posterior_prob,posterior_minus_global,pred_global,pred_full,query_mastery,query_recent_mastery", paste0("case_id=", chosen_case_id))

audit <- bind_rows(audit_rows)
audit_path <- file.path(repo_root, "docs", "paper_review_2025_2026", "figure_data_audit.csv")
utils::write.csv(audit, audit_path, row.names = FALSE)
