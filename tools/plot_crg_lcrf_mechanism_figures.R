#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(dplyr)
  library(tidyr)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "results/crg_lcrf_small_core_20260519_compact"
root <- normalizePath(root, mustWork = TRUE)
out_dir <- if (length(args) >= 2) args[[2]] else file.path(root, "paper_figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c("assist_09", "junyi", "assist_17", "nips34")
dataset_labels <- c(assist_09 = "ASSIST09", junyi = "Junyi", assist_17 = "ASSIST17", nips34 = "NIPS34")
pal <- c(
  evidence = "#D99A00",
  crg = "#2C6DA4",
  lcrf = "#B2182B",
  seq = "#1B9E77",
  random = "#8A99A3",
  self = "#4D4D4D",
  mean = "#EF8A62",
  ink = "#1E2A35",
  grid = "#E8EBEF"
)

theme_pub <- function(base = 6.5) {
  theme_bw(base_size = base) +
    theme(
      text = element_text(family = "sans", colour = pal[["ink"]]),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      axis.title = element_text(face = "bold", size = base + 0.2),
      axis.text = element_text(size = base - 0.6),
      strip.background = element_rect(fill = "#F3F5F7", colour = "#B8C2CC", linewidth = 0.25),
      strip.text = element_text(face = "bold", size = base),
      legend.title = element_blank(),
      legend.text = element_text(size = base - 0.6),
      legend.position = "bottom",
      legend.key.height = unit(0.12, "in"),
      legend.key.width = unit(0.25, "in"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = pal[["grid"]], linewidth = 0.25),
      panel.border = element_rect(colour = "#B8C2CC", fill = NA, linewidth = 0.35),
      plot.margin = margin(3, 4, 3, 4)
    )
}
theme_set(theme_pub())

read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

save_plot <- function(p, name, width, height) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height, dpi = 360, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height, bg = "white")
}

label_ds <- function(x) factor(dataset_labels[x], levels = dataset_labels[datasets])

# Figure 1: vector mechanism diagram.
nodes <- data.frame(
  x = c(1, 2.5, 4.0, 5.5, 7.0),
  y = c(1, 1, 1, 1, 1),
  label = c("train-only\nevidence", "CRG\nroadmap", "fixed\nsupport", "LCRF\nposterior", "prediction"),
  kind = c("evidence", "crg", "crg", "lcrf", "lcrf")
)
edges <- data.frame(x = nodes$x[-nrow(nodes)] + 0.45, xend = nodes$x[-1] - 0.45, y = 1, yend = 1)
fig1 <- ggplot() +
  geom_segment(
    data = edges,
    aes(x = x, xend = xend, y = y, yend = yend),
    arrow = arrow(length = unit(0.08, "in"), type = "closed"),
    linewidth = 0.45,
    colour = "#6B7780"
  ) +
  geom_rect(
    data = nodes,
    aes(xmin = x - 0.48, xmax = x + 0.48, ymin = y - 0.24, ymax = y + 0.24, fill = kind),
    colour = "white",
    linewidth = 0.35,
    radius = unit(0.02, "in")
  ) +
  geom_text(data = nodes, aes(x = x, y = y, label = label), size = 2.4, fontface = "bold", lineheight = 0.88) +
  annotate("text", x = 1, y = 0.55, label = "item + sequence + self", size = 2.0, colour = "#52616B") +
  annotate("text", x = 5.5, y = 0.55, label = "learner state filters routes", size = 2.0, colour = "#52616B") +
  scale_fill_manual(values = c(evidence = "#F1C46B", crg = "#8DB8D8", lcrf = "#E5A1AA")) +
  coord_cartesian(xlim = c(0.35, 7.65), ylim = c(0.35, 1.45), expand = FALSE) +
  theme_void(base_size = 7) +
  theme(legend.position = "none", plot.margin = margin(4, 4, 4, 4))
save_plot(fig1, "fig1_mechanism_crg_lcrf", 6.6, 1.35)

# Figure 2: data cards + CRG retrieval dumbbell.
phen <- read_csv(file.path(root, "data_phenomenon", "crg_lcrf_data_readiness.csv"))
retrieval <- bind_rows(lapply(c("assist_09", "junyi", "assist_17"), function(ds) {
  dat <- read_csv(file.path(root, "crg_retrieval", ds, "crg_transition_retrieval.csv"))
  if (is.null(dat)) return(NULL)
  dat$dataset <- ds
  dat
}))

phen_plot <- phen %>%
  filter(dataset %in% datasets) %>%
  mutate(
    dataset_label = label_ds(dataset),
    single = 1 - multi_concept_item_rate,
    bridge = test_e_bridge_only_rate,
    seq = seq_density
  ) %>%
  select(dataset, dataset_label, single, bridge, seq) %>%
  pivot_longer(c(single, bridge, seq), names_to = "signal", values_to = "value") %>%
  mutate(signal = factor(signal, levels = c("single", "bridge", "seq"), labels = c("single", "bridge", "seq edge")))

p2a <- ggplot(phen_plot, aes(value, dataset_label, colour = signal)) +
  geom_segment(aes(x = 0, xend = value, y = dataset_label, yend = dataset_label), colour = "#DDE3E8", linewidth = 0.45) +
  geom_point(size = 1.6) +
  facet_wrap(~signal, nrow = 1) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.02)) +
  scale_colour_manual(values = c(single = pal[["crg"]], bridge = pal[["lcrf"]], `seq edge` = pal[["seq"]])) +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "none", panel.grid.major.y = element_blank(), panel.spacing.x = unit(0.08, "in"))

ret_best <- retrieval %>%
  filter(variant %in% c("CRG_fused_prior", "CRG_seq_only", "CRG_item_only", "CRG_degree_random", "CRG_self_only")) %>%
  mutate(role = case_when(
    variant == "CRG_self_only" ~ "self",
    variant == "CRG_degree_random" ~ "random",
    TRUE ~ "best CRG"
  )) %>%
  group_by(dataset, role) %>%
  summarise(hit10 = max(`hit@10`, na.rm = TRUE), .groups = "drop") %>%
  mutate(dataset_label = label_ds(dataset), role = factor(role, levels = c("self", "random", "best CRG")))

p2b <- ggplot(ret_best, aes(role, hit10, colour = role, group = dataset_label)) +
  geom_line(aes(group = dataset_label), colour = "#B8C2CC", linewidth = 0.4) +
  geom_point(size = 1.8) +
  facet_wrap(~dataset_label, nrow = 1) +
  scale_colour_manual(values = c(self = pal[["self"]], random = pal[["random"]], `best CRG` = pal[["crg"]])) +
  coord_cartesian(ylim = c(0, 0.43)) +
  labs(x = NULL, y = "Hit@10") +
  theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")

save_plot(p2a / p2b + plot_layout(heights = c(0.9, 1.1)), "fig2_data_and_crg_retrieval", 6.9, 4.1)

# Figure 3: CRG support necessity controls.
control <- bind_rows(lapply(c("assist_09", "assist_17", "junyi"), function(ds) {
  dat <- read_csv(file.path(root, "crg_support_corruption_control", ds, "crg_support_corruption_control.csv"))
  if (is.null(dat)) return(NULL)
  dat
}))
if (!is.null(control) && nrow(control) > 0) {
  control_sum <- control %>%
    filter(group %in% c("all", "high_support_mass", "query_seq_top5_q4_high")) %>%
    group_by(dataset, corruption_type, corruption_ratio, group) %>%
    summarise(
      auc_drop = mean(auc_drop, na.rm = TRUE),
      bce_increase = mean(bce_increase, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      dataset_label = label_ds(dataset),
      corruption_label = recode(
        corruption_type,
        evidence_support_corruption = "evidence",
        degree_matched_random_support = "degree-rand",
        sequence_shuffled_support = "seq-shuffle",
        self_only_fallback = "self-only"
      )
    )
  metric_long <- control_sum %>%
    filter(group == "all") %>%
    pivot_longer(c(auc_drop, bce_increase), names_to = "metric", values_to = "value") %>%
    mutate(metric = factor(metric, levels = c("auc_drop", "bce_increase"), labels = c("AUC drop", "BCE inc.")))
  fig3 <- ggplot(metric_long, aes(corruption_ratio, value, colour = corruption_label)) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.3) +
    facet_grid(metric ~ dataset_label, scales = "free_y") +
    scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0, 0.5, 1)) +
    scale_colour_manual(values = c(evidence = pal[["crg"]], `degree-rand` = pal[["random"]], `seq-shuffle` = pal[["seq"]], `self-only` = pal[["self"]])) +
    labs(x = "corruption ratio", y = NULL, colour = NULL) +
    theme(legend.position = "bottom")
  save_plot(fig3, "fig3_crg_support_necessity_controls", 6.9, 3.6)
}

# Figure 4: LCRF counterfactual delta AUC.
lcrf <- bind_rows(lapply(datasets, function(ds) {
  dat <- read_csv(file.path(root, "lcrf_case_studies", ds, "metrics_check.csv"))
  if (is.null(dat)) return(NULL)
  dat$dataset <- ds
  dat
}))
if (!is.null(lcrf) && nrow(lcrf) > 0) {
  lcrf_delta <- lcrf %>%
    filter(variant %in% c("full", "no_LCRF", "LCRF_mean", "LCRF_shuffle")) %>%
    select(dataset, variant, auc) %>%
    pivot_wider(names_from = variant, values_from = auc) %>%
    transmute(
      dataset,
      `no filter` = full - no_LCRF,
      mean = full - LCRF_mean,
      shuffle = full - LCRF_shuffle
    ) %>%
    pivot_longer(-dataset, names_to = "counterfactual", values_to = "delta_auc") %>%
    mutate(dataset_label = label_ds(dataset), counterfactual = factor(counterfactual, levels = c("no filter", "mean", "shuffle")))
  fig4 <- ggplot(lcrf_delta, aes(counterfactual, delta_auc, fill = counterfactual)) +
    geom_col(width = 0.62, colour = "white", linewidth = 0.2) +
    facet_wrap(~dataset_label, nrow = 1) +
    scale_fill_manual(values = c(`no filter` = pal[["mean"]], mean = pal[["self"]], shuffle = pal[["random"]])) +
    labs(x = NULL, y = expression(Delta * "AUC from full")) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1), legend.position = "none")
  save_plot(fig4, "fig4_lcrf_counterfactual_delta_auc", 6.8, 2.25)
}

# Figure 5: LCRF same-query posterior.
same_files <- list.files(file.path(root, "lcrf_same_query_posterior"), pattern = "_same_query_learner_posterior_topk.csv$", full.names = TRUE)
same <- bind_rows(lapply(same_files, read_csv))
if (!is.null(same) && nrow(same) > 0) {
  same <- same %>%
    mutate(
      dataset_label = label_ds(dataset),
      learner_label = paste0("L", learner_rank)
    )
  # Prefer one sparse core case and one dense contrast case.
  keep_cases <- same %>%
    distinct(dataset, case_id, dataset_label, mean_pairwise_l1, mean_pairwise_js) %>%
    arrange(desc(dataset %in% c("assist_09", "assist_17", "nips34")), desc(mean_pairwise_l1 + 2 * mean_pairwise_js)) %>%
    slice_head(n = 2)
  same_keep <- same %>% semi_join(keep_cases, by = c("dataset", "case_id"))
  support_rank <- same_keep %>%
    group_by(dataset, dataset_label, case_id, support_concept_id) %>%
    summarise(global_support_prob = mean(global_support_prob), posterior_prob = mean(posterior_prob), .groups = "drop") %>%
    group_by(dataset, case_id) %>%
    arrange(desc(global_support_prob), .by_group = TRUE) %>%
    mutate(
      support_order = row_number(),
      support_label = paste0(support_order, ":C", support_concept_id),
      case_panel = paste0(dataset_label, " / ", case_id)
    ) %>%
    ungroup()
  same_keep <- same_keep %>%
    left_join(support_rank %>% select(dataset, case_id, support_concept_id, support_order, support_label, case_panel), by = c("dataset", "case_id", "support_concept_id")) %>%
    group_by(dataset, case_id) %>%
    mutate(support_label = factor(support_label, levels = support_rank$support_label[support_rank$dataset == first(dataset) & support_rank$case_id == first(case_id)])) %>%
    ungroup()

  p5a <- ggplot(support_rank, aes(support_label, global_support_prob)) +
    geom_col(fill = pal[["crg"]], width = 0.65, colour = "white", linewidth = 0.2) +
    facet_wrap(~case_panel, scales = "free_x", nrow = 1) +
    labs(x = NULL, y = "CRG prob") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  p5b <- ggplot(same_keep, aes(support_label, learner_label, fill = posterior_prob)) +
    geom_tile(colour = "white", linewidth = 0.15) +
    facet_wrap(~case_panel, scales = "free_x", nrow = 1) +
    scale_fill_gradient(low = "#F7F7F7", high = pal[["lcrf"]]) +
    labs(x = NULL, y = NULL, fill = "posterior") +
    theme(axis.text.x = element_text(angle = 35, hjust = 1), legend.position = "right")

  pred_shift <- same_keep %>%
    distinct(dataset, dataset_label, case_id, case_panel, learner_rank, learner_label, pred_global, pred_full, true_label) %>%
    pivot_longer(c(pred_global, pred_full), names_to = "pred_type", values_to = "prob") %>%
    mutate(pred_type = factor(pred_type, levels = c("pred_global", "pred_full"), labels = c("global", "full")))
  p5c <- ggplot(pred_shift, aes(pred_type, prob, group = learner_label, colour = true_label > 0.5)) +
    geom_line(alpha = 0.45, linewidth = 0.35) +
    geom_point(size = 1.0) +
    facet_wrap(~case_panel, nrow = 1) +
    scale_colour_manual(values = c(`FALSE` = "#8A99A3", `TRUE` = pal[["lcrf"]])) +
    labs(x = NULL, y = "pred.", colour = "label=1") +
    theme(legend.position = "bottom")
  save_plot((p5a / p5b / p5c) + plot_layout(heights = c(0.75, 1.45, 0.9)), "fig5_lcrf_same_query_posterior", 6.9, 5.1)
}

message("Wrote mechanism figures to: ", out_dir)
