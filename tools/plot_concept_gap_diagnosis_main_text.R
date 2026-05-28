#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "."
out_dir <- file.path(root, "results", "main_problem_experiments_20260523", "main_text")
fig_base <- file.path(out_dir, "fig_concept_gap_diagnosis")

read_csv <- function(name) {
  read.csv(file.path(out_dir, name), stringsAsFactors = FALSE, check.names = FALSE)
}

src <- read_csv("crg_evidence_source_decomposition.csv")

dataset_labels <- c(assist_09 = "ASSIST09", junyi = "Junyi", assist_17 = "ASSIST17")
method_levels <- c("Self", "Rand", "Deg-rand", "Seq", "Fused")
pal_methods <- c(
  Self = "#d9d9d9",
  Rand = "#b7b7b7",
  `Deg-rand` = "#a6a6a6",
  Seq = "#2f9f9b",
  Fused = "#1f4e8c"
)

theme_pub <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.3, colour = "black"),
      axis.text = element_text(colour = "black"),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = base_size),
      legend.position = "top",
      legend.title = element_blank(),
      legend.key.width = unit(7, "pt"),
      legend.key.height = unit(5, "pt"),
      legend.text = element_text(size = 5.5),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      panel.spacing = unit(4, "pt"),
      plot.margin = margin(2, 2, 2, 2),
      plot.title = element_blank()
    )
}

retr <- src[src$task == "history_to_query" & src$method %in% c("random", "seq-only", "fused CRG"), ]
retr <- retr[retr$dataset %in% c("assist_09", "junyi"), ]
retr$dataset_label <- dataset_labels[retr$dataset]
retr$method_label <- factor(
  c(random = "Rand", `seq-only` = "Seq", `fused CRG` = "Fused")[retr$method],
  levels = method_levels
)

p_a <- ggplot(retr, aes(dataset_label, hit10, fill = method_label)) +
  geom_col(width = 0.62, position = position_dodge(width = 0.66), colour = "black", linewidth = 0.18) +
  scale_fill_manual(values = pal_methods, breaks = method_levels, limits = method_levels, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Hit@10") +
  theme_pub(6.6) +
  theme(legend.position = "none")

held <- src[src$task == "heldout_transition" &
              src$method %in% c("CRG_self_only", "CRG_degree_random", "CRG_seq_only", "CRG_fused_prior"), ]
held$dataset_label <- dataset_labels[held$dataset]
held$method_label <- factor(
  c(
    CRG_self_only = "Self",
    CRG_degree_random = "Deg-rand",
    CRG_seq_only = "Seq",
    CRG_fused_prior = "Fused"
  )[held$method],
  levels = method_levels
)

p_b <- ggplot(held, aes(dataset_label, hit10, fill = method_label)) +
  geom_col(width = 0.64, position = position_dodge(width = 0.68), colour = "black", linewidth = 0.18) +
  scale_fill_manual(values = pal_methods, breaks = method_levels, limits = method_levels, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Hit@10") +
  guides(fill = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_pub(6.6)

tag_theme <- theme(
  plot.tag = element_text(face = "bold", size = 7.2, family = "Arial"),
  plot.tag.position = c(0.01, 0.98)
)

fig <- (p_a + labs(tag = "A") + tag_theme) /
  (p_b + labs(tag = "B") + tag_theme) +
  plot_layout(heights = c(1.0, 1.08))

ggsave(paste0(fig_base, ".pdf"), fig, width = 3.48, height = 3.2, units = "in", device = cairo_pdf)
ggsave(paste0(fig_base, ".png"), fig, width = 3.48, height = 3.2, units = "in", dpi = 600, bg = "white")
