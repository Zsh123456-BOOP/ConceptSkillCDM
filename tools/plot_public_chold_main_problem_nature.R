#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(scales)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
repo_root <- if (length(args) >= 1) args[[1]] else "."
run_id <- if (length(args) >= 2) args[[2]] else "public_chold_full_20260529_v1"

run_dir <- file.path(repo_root, "results", run_id)
fig_dir <- file.path(run_dir, "paper_figures")
archive_dir <- file.path(fig_dir, "archive")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(archive_dir, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  blue = "#244C8F",
  blue_mid = "#5E84BF",
  teal = "#3A9D9A",
  teal_light = "#9ED5CF",
  grey = "#B8B8B8",
  grey_dark = "#555555",
  orange = "#D98C32",
  red = "#C8524A",
  paper = "#FFFFFF"
)
c0 <- function(name) unname(pal[[name]])

dataset_levels <- c("assist_09_chold", "assist_17_chold", "junyi_chold")
dataset_labels <- c(
  assist_09_chold = "ASSIST09",
  assist_17_chold = "ASSIST17",
  junyi_chold = "Junyi"
)
route_levels <- c("low_route", "mid_route", "high_route")
route_labels <- c(
  low_route = "Low",
  mid_route = "Medium",
  high_route = "High"
)
variant_labels <- c(
  no_CRG = "w/o CRG",
  self_only = "Self-only",
  degree_random_support = "Degree-random"
)
variant_pal <- c(
  "w/o CRG" = c0("blue"),
  "Self-only" = c0("grey_dark"),
  "Degree-random" = c0("orange")
)

theme_nature <- function(base_size = 6.8, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.30, colour = "black"),
      axis.text = element_text(size = base_size - 0.4, colour = "black"),
      axis.title = element_text(size = base_size, colour = "black"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base_size - 0.2),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.7),
      legend.key.height = unit(3.5, "mm"),
      legend.key.width = unit(6.0, "mm"),
      legend.position = "bottom",
      plot.title = element_text(size = base_size + 0.8, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = base_size - 0.5, colour = c0("grey_dark"), hjust = 0),
      plot.caption = element_text(size = base_size - 0.8, colour = c0("grey_dark"), hjust = 0),
      panel.grid = element_blank(),
      plot.margin = margin(3, 3, 3, 3)
    )
}
theme_set(theme_nature())

read_csv_base <- function(path) {
  if (!file.exists(path)) {
    stop("missing file: ", path)
  }
  as_tibble(utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE))
}

save_figure <- function(plot, name, width_mm = 183, height_mm = 120, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  ggsave(
    file.path(fig_dir, paste0(name, ".pdf")),
    plot,
    width = w,
    height = h,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
  ggsave(
    file.path(fig_dir, paste0(name, ".png")),
    plot,
    width = w,
    height = h,
    units = "in",
    dpi = dpi,
    bg = "white"
  )
}

save_archive <- function(plot, name, width_mm = 183, height_mm = 70, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  ggsave(
    file.path(archive_dir, paste0(name, ".pdf")),
    plot,
    width = w,
    height = h,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
  ggsave(
    file.path(archive_dir, paste0(name, ".png")),
    plot,
    width = w,
    height = h,
    units = "in",
    dpi = dpi,
    bg = "white"
  )
}

profile_path <- file.path(run_dir, "public_chold_gap_profile.csv")
metric_path <- file.path(run_dir, "main_problem_analysis", "main_problem_exp2_coverage_conditioned_metrics_all.csv")
route_path <- file.path(run_dir, "route_bin_recovery_summary.csv")
agg_path <- file.path(run_dir, "mechanism_results.csv")

if (!file.exists(profile_path)) {
  stop(
    "missing aggregate profile file: ", profile_path,
    "\nRun tools/plot_public_concept_heldout_results.py before this script."
  )
}

profile <- read_csv_base(profile_path) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    dataset_label = dataset_labels[as.character(dataset)]
  )

auc_rank <- function(y, p) {
  y <- as.numeric(y)
  p <- as.numeric(p)
  keep <- is.finite(y) & is.finite(p)
  y <- y[keep]
  p <- p[keep]
  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

bce_score <- function(y, p) {
  y <- as.numeric(y)
  p <- pmin(pmax(as.numeric(p), 1e-7), 1 - 1e-7)
  mean(-(y * log(p) + (1 - y) * log(1 - p)))
}

rmse_score <- function(y, p) sqrt(mean((as.numeric(y) - as.numeric(p))^2))

if (!file.exists(route_path)) {
  pred_files <- list.files(
    file.path(run_dir, "main_problem_analysis"),
    pattern = "main_problem_exp2_coverage_conditioned_predictions\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(pred_files) == 0) {
    stop("missing route summary and prediction CSVs under ", file.path(run_dir, "main_problem_analysis"))
  }
  route_rows <- lapply(pred_files, function(path) {
    dataset <- basename(dirname(path))
    pred <- read_csv_base(path) %>%
      filter(as.logical(direct_unseen_bridgeable), route_mass_to_query > 0)
    if (nrow(pred) == 0) return(tibble())
    qs <- as.numeric(stats::quantile(pred$route_mass_to_query, probs = c(1 / 3, 2 / 3), na.rm = TRUE))
    pred <- pred %>%
      mutate(
        route_bin = case_when(
          route_mass_to_query <= qs[[1]] ~ "low_route",
          route_mass_to_query <= qs[[2]] ~ "mid_route",
          TRUE ~ "high_route"
        )
      )
    bind_rows(lapply(route_levels, function(bin_name) {
      sub <- pred %>% filter(route_bin == bin_name)
      if (nrow(sub) < 500) return(tibble())
      y <- sub$label_eval
      pf <- sub$prob_full
      full_auc <- auc_rank(y, pf)
      full_bce <- bce_score(y, pf)
      full_rmse <- rmse_score(y, pf)
      bind_rows(lapply(c("no_CRG", "self_only", "degree_random_support"), function(v) {
        col <- paste0("prob_", v)
        if (!col %in% names(sub)) return(tibble())
        pv <- sub[[col]]
        tibble(
          dataset = dataset,
          route_bin = bin_name,
          variant = v,
          n_eval = nrow(sub),
          route_mass_mean = mean(sub$route_mass_to_query, na.rm = TRUE),
          full_auc = full_auc,
          variant_auc = auc_rank(y, pv),
          auc_recovery = full_auc - auc_rank(y, pv),
          full_bce = full_bce,
          variant_bce = bce_score(y, pv),
          bce_recovery = bce_score(y, pv) - full_bce,
          full_rmse = full_rmse,
          variant_rmse = rmse_score(y, pv),
          rmse_recovery = rmse_score(y, pv) - full_rmse
        )
      }))
    }))
  }) %>% bind_rows()
  utils::write.csv(route_rows, route_path, row.names = FALSE)
}

route <- read_csv_base(route_path) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    dataset_label = dataset_labels[as.character(dataset)],
    route_bin = factor(route_bin, levels = route_levels, labels = route_labels[route_levels]),
    variant = factor(variant, levels = names(variant_labels), labels = variant_labels)
  )

agg <- read_csv_base(agg_path) %>%
  filter(dataset %in% dataset_levels) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    dataset_label = dataset_labels[as.character(dataset)],
    variant_label = recode(
      variant,
      full = "Full",
      no_A = "w/o CRG",
      A_self_only = "Self-only",
      A_degree_random = "Degree-random",
      .default = variant
    )
  )

target_metrics <- list.files(
  file.path(run_dir, "main_problem_analysis"),
  pattern = "main_problem_exp2_coverage_conditioned_metrics\\.csv$",
  recursive = TRUE,
  full.names = TRUE
) %>%
  lapply(read_csv_base) %>%
  bind_rows() %>%
  filter(
    subgroup %in% c("direct_unseen_bridgeable", "high_route_mass"),
    variant %in% c("no_CRG", "self_only", "degree_random_support")
  ) %>%
  mutate(
    dataset = factor(dataset, levels = dataset_levels),
    dataset_label = dataset_labels[as.character(dataset)],
    subgroup_label = recode(
      subgroup,
      direct_unseen_bridgeable = "Direct-unseen\nbridgeable",
      high_route_mass = "High-route\ngap"
    ),
    variant = factor(variant, levels = names(variant_labels), labels = variant_labels)
  )

profile_long <- profile %>%
  select(dataset, dataset_label, train_ratio, valid_ratio, test_ratio) %>%
  pivot_longer(
    cols = c(train_ratio, valid_ratio, test_ratio),
    names_to = "split",
    values_to = "ratio"
  ) %>%
  mutate(
    split = factor(
      split,
      levels = c("train_ratio", "valid_ratio", "test_ratio"),
      labels = c("Train", "Valid", "Test")
    )
  )

p_gap_split <- ggplot(profile_long, aes(dataset_label, ratio, fill = split)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.18) +
  geom_hline(yintercept = c(0.7, 0.8), linewidth = 0.22, linetype = c("dashed", "dotted"), colour = "grey45") +
  scale_fill_manual(values = c(Train = c0("blue"), Valid = c0("orange"), Test = c0("teal"))) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.02), expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "Row share", title = "A  Student-conditioned concept-heldout split") +
  guides(fill = guide_legend(nrow = 1)) +
  theme(legend.position = "bottom")

p_gap_rate <- ggplot(profile, aes(dataset_label)) +
  geom_col(aes(y = test_direct_unseen_rate), width = 0.56, fill = c0("blue"), alpha = 0.92) +
  geom_point(aes(y = test_train_concept_overlap_rate), size = 1.8, colour = c0("orange")) +
  geom_text(
    aes(y = pmin(test_direct_unseen_rate + 0.045, 1.04), label = paste0("n=", comma(test_direct_unseen_rows))),
    size = 2.0,
    family = "Arial"
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.10), expand = c(0, 0)) +
  labs(
    x = NULL,
    y = "Rate in test set",
    title = "B  Query-level target-concept gap",
    subtitle = "Bars: direct-unseen; dots: train-test concept overlap"
  )

fig_ab <- p_gap_split | p_gap_rate
save_archive(fig_ab, "fig_public_chold_gap_profile_ab", width_mm = 183, height_mm = 62)

p_route <- ggplot(route, aes(route_bin, auc_recovery, fill = variant)) +
  geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey35") +
  geom_col(position = position_dodge(width = 0.72), width = 0.62, colour = "white", linewidth = 0.12) +
  facet_wrap(~ dataset_label, nrow = 1) +
  scale_fill_manual(values = variant_pal, drop = FALSE) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    x = "Route-support strength",
    y = "AUC recovery\nFull - control",
    title = "C  Route-conditioned recovery",
    subtitle = "Direct-unseen bridgeable queries"
  ) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom"
  )

p_target <- ggplot(target_metrics, aes(subgroup_label, auc_gap_full_minus_variant, fill = variant)) +
  geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey35") +
  geom_col(position = position_dodge(width = 0.70), width = 0.62, colour = "white", linewidth = 0.12) +
  facet_wrap(~ dataset_label, nrow = 1) +
  scale_fill_manual(values = variant_pal, drop = FALSE) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    x = NULL,
    y = "AUC recovery\nFull - control",
    title = "D  Problem-defined cohorts"
  ) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.position = "bottom"
  )

fig_cd <- p_route / p_target +
  plot_layout(heights = c(1.03, 0.97), guides = "collect") &
  theme(legend.position = "bottom")

save_figure(fig_cd, "fig_public_chold_cd_gap_recovery", width_mm = 183, height_mm = 105)

auc_summary <- route %>%
  group_by(dataset_label, variant) %>%
  summarise(mean_auc_recovery = mean(auc_recovery, na.rm = TRUE), .groups = "drop")

p_compact <- ggplot(route, aes(route_bin, auc_recovery, group = variant, colour = variant)) +
  geom_hline(yintercept = 0, linewidth = 0.22, colour = "grey35") +
  geom_line(linewidth = 0.42) +
  geom_point(size = 1.7) +
  facet_wrap(~ dataset_label, nrow = 1) +
  scale_colour_manual(values = variant_pal, drop = FALSE) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    x = "Route-support strength",
    y = "AUC recovery (Full - control)",
    title = "Route-conditioned gap recovery",
    subtitle = "Direct-unseen bridgeable queries"
  ) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

save_figure(p_compact, "fig_public_chold_route_conditioned_auc_recovery", width_mm = 100, height_mm = 58)

audit <- bind_rows(
  profile %>%
    transmute(
      dataset = as.character(dataset),
      evidence = "gap_existence",
      metric = "test_direct_unseen_rate",
      value = test_direct_unseen_rate,
      n_eval = test_rows,
      note = "higher values show the student-level target-concept gap is present"
    ),
  target_metrics %>%
    filter(variant == "w/o CRG", subgroup_label == "Direct-unseen bridgeable") %>%
    transmute(
      dataset = as.character(dataset),
      evidence = "gap_recovery",
      metric = "auc_full_minus_no_CRG",
      value = auc_gap_full_minus_variant,
      n_eval = n_eval,
      note = "positive values favor Full over the route-absent control"
    ),
  route %>%
    filter(variant == "w/o CRG", route_bin == "High route") %>%
    transmute(
      dataset = as.character(dataset),
      evidence = "high_route_recovery",
      metric = "auc_full_minus_no_CRG",
      value = auc_recovery,
      n_eval = n_eval,
      note = "positive values in the high-route bin support route-conditioned recovery"
    )
)
utils::write.csv(audit, file.path(run_dir, "public_chold_main_problem_figure_audit.csv"), row.names = FALSE)
utils::write.csv(auc_summary, file.path(run_dir, "public_chold_route_auc_recovery_summary.csv"), row.names = FALSE)

message("Wrote figures to ", normalizePath(fig_dir, winslash = "/"))
