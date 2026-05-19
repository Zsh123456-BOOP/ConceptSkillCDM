suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(ggrepel)
  library(scales)
  library(dplyr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "results/crg_lcrf_small_core_20260519_compact"
root <- normalizePath(root, mustWork = TRUE)
out_dir <- file.path(root, "paper_figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

datasets <- c("assist_09", "junyi", "assist_17", "nips34")
dataset_labels <- c(assist_09 = "ASSIST09", junyi = "Junyi", assist_17 = "ASSIST17", nips34 = "NIPS34")

pal <- c(
  crg = "#2C6DA4",
  seq = "#1B9E77",
  item = "#D99A00",
  lcrf = "#B2182B",
  lcrf2 = "#EF8A62",
  random = "#8A99A3",
  self = "#4D4D4D",
  gray = "#D6DBDF",
  ink = "#1E2A35"
)

theme_pub <- function(base = 6) {
  theme_bw(base_size = base) +
    theme(
      text = element_text(family = "sans", colour = pal[["ink"]]),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.caption = element_text(size = base - 1, colour = "#52616B", hjust = 0),
      axis.title = element_text(face = "bold", size = base + 0.2),
      axis.text = element_text(size = base - 1),
      strip.background = element_rect(fill = "#F3F5F7", colour = "#B8C2CC", linewidth = 0.25),
      strip.text = element_text(face = "bold", size = base),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = base - 1),
      legend.key.height = unit(0.12, "in"),
      legend.key.width = unit(0.24, "in"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#E8EBEF", linewidth = 0.25),
      panel.border = element_rect(colour = "#B8C2CC", fill = NA, linewidth = 0.35),
      plot.margin = margin(3, 4, 3, 4)
    )
}

save_plot <- function(p, name, width, height) {
  ggsave(file.path(out_dir, paste0(name, ".png")), p, width = width, height = height, dpi = 360, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), p, width = width, height = height, bg = "white")
}

read_csv_base <- function(path) read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)

phen <- read_csv_base(file.path(root, "data_phenomenon", "crg_lcrf_data_readiness.csv")) %>%
  filter(dataset %in% datasets) %>%
  mutate(
    dataset_label = factor(dataset_labels[dataset], levels = dataset_labels[datasets]),
    single_rate = 1 - multi_concept_item_rate,
    direct_unseen = test_e_direct_unseen_rate,
    bridge_only = test_e_bridge_only_rate
  )

retrieval <- bind_rows(lapply(c("assist_09", "junyi", "assist_17"), function(ds) {
  path <- file.path(root, "crg_retrieval", ds, "crg_transition_retrieval.csv")
  if (!file.exists(path)) return(NULL)
  read_csv_base(path) %>%
    mutate(dataset = ds, dataset_label = factor(dataset_labels[ds], levels = dataset_labels[datasets]))
}))

corruption <- bind_rows(lapply(c("assist_09", "junyi", "assist_17"), function(ds) {
  path <- file.path(root, "crg_support_corruption", ds, "crg_support_corruption_aggregate.csv")
  if (!file.exists(path)) return(NULL)
  read_csv_base(path) %>%
    mutate(dataset = ds, dataset_label = factor(dataset_labels[ds], levels = dataset_labels[datasets]))
}))

lcrf_metrics <- bind_rows(lapply(datasets, function(ds) {
  path <- file.path(root, "lcrf_case_studies", ds, "metrics_check.csv")
  if (!file.exists(path)) return(NULL)
  read_csv_base(path) %>%
    mutate(dataset = ds, dataset_label = factor(dataset_labels[ds], levels = dataset_labels[datasets]))
}))

theme_set(theme_pub())

# Fig. 1: compact data fact panel, close to AAAI/KDD style small-multiple evidence plots.
data_fact <- phen %>%
  transmute(
    dataset,
    dataset_label,
    `single item` = single_rate,
    `item edge` = item_density,
    `sequence edge` = seq_density,
    `bridge only` = bridge_only
  ) %>%
  pivot_longer(-c(dataset, dataset_label), names_to = "signal", values_to = "value") %>%
  mutate(
    signal = factor(signal, levels = c("single item", "item edge", "sequence edge", "bridge only")),
    role = case_when(
      dataset %in% c("assist_09", "junyi", "assist_17") ~ "core sparse-route datasets",
      TRUE ~ "dense contrast"
    )
  )

p_data <- ggplot(data_fact, aes(value, dataset_label, colour = signal, shape = role)) +
  geom_segment(aes(x = 0, xend = value, y = dataset_label, yend = dataset_label), colour = "#DDE3E8", linewidth = 0.55) +
  geom_point(size = 2.1, stroke = 0.8) +
  facet_wrap(~signal, nrow = 1) +
  scale_x_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1.02)) +
  scale_colour_manual(values = c("single item" = pal[["crg"]], "item edge" = pal[["item"]], "sequence edge" = pal[["seq"]], "bridge only" = pal[["lcrf"]])) +
  labs(
    x = NULL,
    y = NULL,
    shape = NULL,
    colour = NULL
  ) +
  theme(legend.position = "none", panel.grid.major.y = element_blank(), panel.spacing.x = unit(0.08, "in"))
save_plot(p_data, "fig1_dataset_reachability_compact", 7.1, 1.75)

# Fig. 2: CRG retrieval as a compact ablation grid.
ret_keep <- c("CRG_fused_prior", "CRG_seq_only", "CRG_item_only", "CRG_degree_random", "CRG_self_only")
ret_lab <- c(
  CRG_fused_prior = "Fused",
  CRG_seq_only = "Seq",
  CRG_item_only = "Item",
  CRG_degree_random = "Rand",
  CRG_self_only = "Self"
)
ret_col <- c(Fused = pal[["crg"]], Seq = pal[["seq"]], Item = pal[["item"]], Rand = pal[["random"]], Self = pal[["self"]])

ret_plot <- retrieval %>%
  filter(variant %in% ret_keep) %>%
  mutate(variant_label = factor(ret_lab[variant], levels = c("Fused", "Seq", "Item", "Rand", "Self")))

p_ret <- ggplot(ret_plot, aes(variant_label, `hit@10`, fill = variant_label)) +
  geom_col(width = 0.62, colour = "white", linewidth = 0.2) +
  facet_wrap(~dataset_label, nrow = 1) +
  scale_fill_manual(values = ret_col) +
  coord_cartesian(ylim = c(0, 0.43)) +
  labs(
    x = NULL,
    y = "Hit@10"
  ) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1), panel.spacing.x = unit(0.1, "in"))
save_plot(p_ret, "fig2_crg_retrieval_ablation", 6.8, 2.2)

# Fig. 3: support corruption, one line chart only.
corrupt_all <- corruption %>% filter(group == "all")
p_corrupt <- ggplot(corrupt_all, aes(fraction, auc_drop_mean, colour = dataset_label, fill = dataset_label)) +
  geom_ribbon(aes(ymin = pmax(0, auc_drop_mean - ifelse(is.na(auc_drop_std), 0, auc_drop_std)), ymax = auc_drop_mean + ifelse(is.na(auc_drop_std), 0, auc_drop_std)), alpha = 0.12, colour = NA) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 1.7) +
  scale_x_continuous(labels = percent_format(accuracy = 1), breaks = c(0, 0.25, 0.5, 0.75, 1)) +
  scale_y_continuous(labels = number_format(accuracy = 0.001)) +
  scale_colour_manual(values = c(ASSIST09 = pal[["crg"]], Junyi = pal[["seq"]], ASSIST17 = pal[["lcrf2"]])) +
  scale_fill_manual(values = c(ASSIST09 = pal[["crg"]], Junyi = pal[["seq"]], ASSIST17 = pal[["lcrf2"]])) +
  labs(
    x = "corrupted support ratio",
    y = "AUC drop"
  ) +
  theme(legend.position = c(0.34, 0.86), legend.background = element_rect(fill = "white", colour = "#D6DBDF", linewidth = 0.25))
save_plot(p_corrupt, "fig3_crg_support_corruption", 3.45, 2.25)

# Fig. 4: LCRF counterfactual as a dumbbell-like small multiple.
lcrf_order <- c("full", "no_LCRF", "LCRF_shuffle", "LCRF_mean")
lcrf_lab <- c(full = "Actual", no_LCRF = "No filter", LCRF_shuffle = "Shuffle", LCRF_mean = "Mean")
lcrf_col <- c(Actual = pal[["lcrf"]], `No filter` = pal[["lcrf2"]], Shuffle = pal[["random"]], Mean = pal[["self"]])

lcrf_long <- lcrf_metrics %>%
  filter(variant %in% lcrf_order) %>%
  mutate(variant_label = factor(lcrf_lab[variant], levels = c("Actual", "No filter", "Shuffle", "Mean")))

p_lcrf <- ggplot(lcrf_long, aes(variant_label, auc, group = dataset_label)) +
  geom_line(colour = "#B8C2CC", linewidth = 0.45) +
  geom_point(aes(colour = variant_label), size = 2.2) +
  facet_wrap(~dataset_label, nrow = 1) +
  scale_colour_manual(values = lcrf_col) +
  coord_cartesian(ylim = c(0.48, 0.85)) +
  labs(
    x = NULL,
    y = "AUC"
  ) +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 30, hjust = 1), panel.spacing.x = unit(0.1, "in"))
save_plot(p_lcrf, "fig4_lcrf_counterfactual_dots", 6.9, 2.25)

# Fig. 5: evidence matrix, but as a small numeric heatmap with clear semantics.
crg_lift <- retrieval %>%
  group_by(dataset) %>%
  summarise(
    dataset_label = first(dataset_labels[dataset]),
    best_crg_hit10 = max(`hit@10`[variant %in% c("CRG_fused_prior", "CRG_seq_only")], na.rm = TRUE),
    random_hit10 = `hit@10`[variant == "CRG_degree_random"][1],
    self_hit10 = `hit@10`[variant == "CRG_self_only"][1],
    .groups = "drop"
  ) %>%
  mutate(retrieval_lift = best_crg_hit10 - random_hit10)

corrupt_100 <- corruption %>%
  filter(group == "all", abs(fraction - 1) < 1e-9) %>%
  select(dataset, corruption_drop = auc_drop_mean)

lcrf_drop <- lcrf_long %>%
  select(dataset, dataset_label, variant, auc) %>%
  pivot_wider(names_from = variant, values_from = auc) %>%
  mutate(no_filter_drop = full - no_LCRF, shuffle_drop = full - LCRF_shuffle)

matrix_df <- phen %>%
  transmute(dataset, dataset_label, `Sparse item evidence` = 1 - item_density, `Bridge-only samples` = bridge_only) %>%
  left_join(crg_lift %>% select(dataset, `CRG retrieval lift` = retrieval_lift), by = "dataset") %>%
  left_join(corrupt_100 %>% select(dataset, `CRG corruption drop` = corruption_drop), by = "dataset") %>%
  left_join(lcrf_drop %>% select(dataset, `LCRF no-filter drop` = no_filter_drop, `LCRF shuffle drop` = shuffle_drop), by = "dataset")

matrix_long <- matrix_df %>%
  pivot_longer(-c(dataset, dataset_label), names_to = "claim", values_to = "value") %>%
  group_by(claim) %>%
  mutate(strength = ifelse(all(is.na(value)) || diff(range(value, na.rm = TRUE)) == 0, NA_real_, (value - min(value, na.rm = TRUE)) / diff(range(value, na.rm = TRUE)))) %>%
  ungroup() %>%
  mutate(
    claim = factor(claim, levels = c("Sparse item evidence", "Bridge-only samples", "CRG retrieval lift", "CRG corruption drop", "LCRF no-filter drop", "LCRF shuffle drop")),
    dataset_label = factor(dataset_label, levels = rev(dataset_labels[datasets]))
  )

p_matrix <- ggplot(matrix_long, aes(claim, dataset_label, fill = strength)) +
  geom_tile(colour = "white", linewidth = 0.45) +
  geom_text(aes(label = ifelse(is.na(value), "", sprintf("%.3f", value))), size = 1.8, fontface = "bold") +
  scale_fill_gradient(low = "#F0F4C3", high = "#225EA8", na.value = "#F2F4F7") +
  labs(
    x = NULL,
    y = NULL,
    fill = "strength"
  ) +
  theme(axis.text.x = element_text(angle = 28, hjust = 1), legend.position = "right")
save_plot(p_matrix, "fig5_module_evidence_matrix", 6.6, 2.6)

write.csv(matrix_df, file.path(out_dir, "module_evidence_matrix.csv"), row.names = FALSE)
write.csv(
  phen %>%
    select(dataset, single_rate, multi_concept_item_rate, item_density, seq_density, direct_unseen, bridge_only, student_train_count_median) %>%
    left_join(crg_lift, by = "dataset") %>%
    left_join(corrupt_100, by = "dataset") %>%
    left_join(lcrf_drop %>% select(dataset, full, no_LCRF, LCRF_shuffle, LCRF_mean, no_filter_drop, shuffle_drop), by = "dataset"),
  file.path(out_dir, "paper_figure_summary.csv"),
  row.names = FALSE
)

message("Wrote compact R paper figures to: ", out_dir)
