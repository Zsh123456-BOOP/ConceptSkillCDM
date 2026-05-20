#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else "results/crg_lcrf_core3_final_20260520"
fig_dir <- file.path(root, "paper_figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

read_csv <- function(path) {
  if (!file.exists(path)) {
    return(data.frame())
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

save_both <- function(plot, name, width = 7.2, height = 4.2) {
  ggsave(file.path(fig_dir, paste0(name, ".pdf")), plot, width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(file.path(fig_dir, paste0(name, ".png")), plot, width = width, height = height, units = "in", dpi = 300)
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
      axis.text = element_text(size = base_size - 1)
    )
}

palette <- c(
  "CRG" = "#2878B5",
  "best_CRG" = "#2878B5",
  "self" = "#7F7F7F",
  "degree_random" = "#B65F5F",
  "random_or_uniform" = "#D59A2D",
  "evidence_support_corruption" = "#2878B5",
  "degree_matched_random_support" = "#B65F5F",
  "sequence_shuffled_support" = "#D59A2D",
  "self_only_fallback" = "#7F7F7F",
  "no_filter" = "#7F7F7F",
  "mean_state" = "#D59A2D",
  "shuffle_state" = "#B65F5F"
)

short_type <- function(x) {
  dplyr_map <- c(
    "evidence_support_corruption" = "evidence",
    "degree_matched_random_support" = "degree-rnd",
    "sequence_shuffled_support" = "seq-shuf",
    "self_only_fallback" = "self"
  )
  ifelse(x %in% names(dplyr_map), dplyr_map[x], x)
}

# Fig 2: data cards and retrieval lift
cards <- read_csv(file.path(root, "data_story", "dataset_story_cards_core3.csv"))
retr <- read_csv(file.path(fig_dir, "fig2_core3_retrieval_summary.csv"))
if (nrow(cards) > 0 && nrow(retr) > 0) {
  metric_cols <- c("single_concept_rate", "item_edge_density", "seq_edge_density",
                   "direct_unseen_rate", "bridge_only_rate", "history_len_median")
  card_long <- reshape(cards[, c("dataset", metric_cols)], varying = metric_cols,
                       v.names = "value", timevar = "metric", times = metric_cols,
                       direction = "long")
  card_long$metric <- factor(card_long$metric, levels = metric_cols,
                             labels = c("single", "item edge", "seq edge", "unseen", "bridge", "hist med"))
  card_long$label <- ifelse(card_long$metric == "hist med",
                            sprintf("%.0f", card_long$value),
                            sprintf("%.2f", card_long$value))
  p1 <- ggplot(card_long, aes(metric, dataset, fill = value)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = label), size = 2.2) +
    scale_fill_gradient(low = "#F2F4F7", high = "#2878B5") +
    labs(x = NULL, y = NULL) +
    theme_core()

  retr$role2 <- ifelse(retr$role == "best_CRG", "best_CRG",
                       ifelse(retr$variant == "CRG_self_only", "self",
                              ifelse(retr$variant == "CRG_degree_random", "degree_random", "random_or_uniform")))
  retr$role_label <- factor(retr$role2,
                            levels = c("self", "random_or_uniform", "degree_random", "best_CRG"),
                            labels = c("self", "rnd/unif", "degree-rnd", "best CRG"))
  p2 <- ggplot(retr, aes(`hit@10`, role_label, color = role2)) +
    geom_point(size = 2.3) +
    geom_line(aes(group = dataset), color = "grey80", linewidth = 0.35) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "Hit@10", y = NULL) +
    theme_core()

  if (requireNamespace("gridExtra", quietly = TRUE)) {
    g <- gridExtra::arrangeGrob(p1, p2, ncol = 1, heights = c(1.05, 1.0))
    ggsave(file.path(fig_dir, "fig2_core3_data_and_crg_retrieval.pdf"), g, width = 7.2, height = 5.0, device = cairo_pdf)
    ggsave(file.path(fig_dir, "fig2_core3_data_and_crg_retrieval.png"), g, width = 7.2, height = 5.0, dpi = 300)
  } else {
    save_both(p2, "fig2_core3_data_and_crg_retrieval", 7.2, 3.2)
  }
}

# Fig 3: support corruption
support <- read_csv(file.path(root, "crg_support_audit", "crg_support_gap_audit_core3.csv"))
if (nrow(support) > 0) {
  all_rows <- subset(support, subgroup == "all" & corruption_ratio > 0)
  all_mean <- aggregate(cbind(auc_drop_from_clean, bce_increase_from_clean) ~ dataset + corruption_type + corruption_ratio,
                        all_rows, mean, na.rm = TRUE)
  all_mean$corruption_label <- short_type(all_mean$corruption_type)
  p_auc <- ggplot(all_mean, aes(corruption_ratio, auc_drop_from_clean, color = corruption_type)) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.4) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "corruption ratio", y = "AUC drop") +
    theme_core()
  p_bce <- ggplot(all_mean, aes(corruption_ratio, bce_increase_from_clean, color = corruption_type)) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.4) +
    facet_wrap(~ dataset, nrow = 1) +
    scale_color_manual(values = palette) +
    labs(x = "corruption ratio", y = "BCE increase") +
    theme_core()
  gap <- subset(all_rows, corruption_type == "evidence_support_corruption" & corruption_ratio == 1)
  gap <- aggregate(evidence_minus_degree_random_auc_drop ~ dataset + claim_status, gap, mean, na.rm = TRUE)
  p_gap <- ggplot(gap, aes(dataset, evidence_minus_degree_random_auc_drop, fill = claim_status)) +
    geom_col(width = 0.55) +
    geom_hline(yintercept = 0, linewidth = 0.25) +
    labs(x = NULL, y = "evidence - degree AUC drop") +
    theme_core()
  if (requireNamespace("gridExtra", quietly = TRUE)) {
    g <- gridExtra::arrangeGrob(p_auc, p_bce, p_gap, ncol = 1, heights = c(1, 1, 0.8))
    ggsave(file.path(fig_dir, "fig3_core3_support_corruption_auc_bce.pdf"), g, width = 7.2, height = 7.0, device = cairo_pdf)
    ggsave(file.path(fig_dir, "fig3_core3_support_corruption_auc_bce.png"), g, width = 7.2, height = 7.0, dpi = 300)
  } else {
    save_both(p_auc, "fig3_core3_support_corruption_auc_bce", 7.2, 3.0)
  }
}

subgroup <- read_csv(file.path(root, "crg_support_audit", "crg_subgroup_support_dependence_core3.csv"))
if (nrow(subgroup) > 0) {
  sg <- subset(subgroup, corruption_ratio == 1 & corruption_type %in% c("evidence_support_corruption", "degree_matched_random_support", "sequence_shuffled_support", "self_only_fallback"))
  sg <- aggregate(auc_drop_from_clean ~ dataset + requested_subgroup + corruption_type, sg, mean, na.rm = TRUE)
  p_sg <- ggplot(sg, aes(requested_subgroup, auc_drop_from_clean, fill = corruption_type)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = palette) +
    coord_flip() +
    labs(x = NULL, y = "AUC drop at 100%") +
    theme_core()
  save_both(p_sg, "figS_crg_subgroup_support_dependence_core3", 7.4, 4.6)
}

# Fig 4: LCRF counterfactual
cf <- read_csv(file.path(root, "lcrf_counterfactual", "lcrf_counterfactual_delta_core3.csv"))
if (nrow(cf) > 0) {
  p4 <- ggplot(cf, aes(variant, auc_drop_from_full, fill = variant)) +
    geom_col(width = 0.62) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_y") +
    scale_fill_manual(values = palette) +
    labs(x = NULL, y = expression(Delta * "AUC from full")) +
    theme_core()
  save_both(p4, "fig4_core3_lcrf_counterfactual_delta_auc_bce", 7.2, 3.1)
}

# Fig 5: same-query posterior
same <- read_csv(file.path(root, "lcrf_same_query", "lcrf_same_query_annotated_core3.csv"))
twostu <- read_csv(file.path(root, "lcrf_same_query", "lcrf_two_student_path_case_core3.csv"))
if (nrow(same) > 0) {
  choice <- subset(same, dataset %in% c("assist_17", "assist_09"))
  if (nrow(choice) > 0) {
    case_scores <- aggregate(mean_pairwise_l1 ~ dataset + case_id, choice, max, na.rm = TRUE)
    case_scores <- case_scores[order(-case_scores$mean_pairwise_l1), ]
    chosen <- case_scores[1, ]
    one <- subset(choice, dataset == chosen$dataset & case_id == chosen$case_id)
    one$learner_id_anonymized <- factor(one$learner_id_anonymized, levels = unique(one$learner_id_anonymized))
    support_mean <- aggregate(cbind(global_support_prob, posterior_prob) ~ support_concept_name, one, mean, na.rm = TRUE)
    p5a <- ggplot(support_mean, aes(reorder(support_concept_name, global_support_prob), global_support_prob)) +
      geom_col(fill = "#2878B5", width = 0.65) +
      coord_flip() +
      labs(x = NULL, y = "CRG prob") +
      theme_core()
    p5b <- ggplot(one, aes(support_concept_name, learner_id_anonymized, fill = posterior_minus_global)) +
      geom_tile(color = "white", linewidth = 0.25) +
      scale_fill_gradient2(low = "#B65F5F", mid = "white", high = "#2878B5") +
      labs(x = "support", y = "learner") +
      theme_core()
    shift <- unique(one[, c("learner_id_anonymized", "pred_global", "pred_full", "true_label")])
    shift <- head(shift, 12)
    p5c <- ggplot(shift, aes(y = learner_id_anonymized)) +
      geom_segment(aes(x = pred_global, xend = pred_full, yend = learner_id_anonymized), color = "grey65") +
      geom_point(aes(x = pred_global), color = "#7F7F7F", size = 1.7) +
      geom_point(aes(x = pred_full), color = "#2878B5", size = 1.7) +
      labs(x = "prediction", y = NULL) +
      theme_core()
    if (nrow(twostu) > 0) {
      p5d <- ggplot(twostu, aes(support_concept_name, posterior_prob, fill = learner_id_anonymized)) +
        geom_col(position = position_dodge(width = 0.72), width = 0.62) +
        labs(x = "top support", y = "posterior") +
        theme_core()
    } else {
      p5d <- p5c
    }
    if (requireNamespace("gridExtra", quietly = TRUE)) {
      g <- gridExtra::arrangeGrob(p5a, p5b, p5c, p5d, ncol = 2)
      ggsave(file.path(fig_dir, "fig5_core3_lcrf_same_query_posterior_annotated.pdf"), g, width = 7.4, height = 5.6, device = cairo_pdf)
      ggsave(file.path(fig_dir, "fig5_core3_lcrf_same_query_posterior_annotated.png"), g, width = 7.4, height = 5.6, dpi = 300)
    } else {
      save_both(p5b, "fig5_core3_lcrf_same_query_posterior_annotated", 6.0, 4.0)
    }
  }
}

# Appendix route, timeline, and source audit figures
route_edges <- read_csv(file.path(root, "crg_local_route_cases", "crg_local_route_case_edges_core3.csv"))
if (nrow(route_edges) > 0) {
  route_top <- subset(route_edges, fused_crg_prob == fused_crg_prob)
  route_top <- route_top[order(route_top$dataset, route_top$case_id, -route_top$fused_crg_prob), ]
  route_top <- do.call(rbind, by(route_top, route_top$case_id, head, 6))
  p_route <- ggplot(route_top, aes(target_concept_name, fused_crg_prob, fill = dataset)) +
    geom_col(width = 0.62) +
    facet_wrap(~ case_id, scales = "free_x") +
    coord_flip() +
    labs(x = NULL, y = "CRG prob") +
    theme_core()
  save_both(p_route, "figS_crg_local_route_cases_core3", 7.4, 5.0)
}

timeline <- read_csv(file.path(root, "lcrf_student_timeline", "lcrf_specific_student_timeline_core3.csv"))
if (nrow(timeline) > 0) {
  tl <- timeline[, c("dataset", "student_id_anonymized", "event_order", "pred_global", "pred_full", "pred_shuffle_state", "pred_mean_state")]
  tl_long <- reshape(tl, varying = c("pred_global", "pred_full", "pred_shuffle_state", "pred_mean_state"),
                     v.names = "pred", timevar = "variant",
                     times = c("global", "full", "shuffle", "mean"), direction = "long")
  p_tl <- ggplot(tl_long, aes(event_order, pred, color = variant)) +
    geom_line(linewidth = 0.45) +
    geom_point(size = 1.2) +
    facet_wrap(~ dataset + student_id_anonymized, scales = "free_x") +
    labs(x = "event", y = "prediction") +
    theme_core()
  save_both(p_tl, "figS_lcrf_specific_student_timeline_core3", 7.4, 4.4)
}

state <- read_csv(file.path(root, "lcrf_state_source_audit", "lcrf_state_source_audit_core3.csv"))
if (nrow(state) > 0) {
  p_state <- ggplot(state, aes(variant, auc_drop_from_full, fill = variant)) +
    geom_col(width = 0.62) +
    facet_wrap(~ dataset, nrow = 1, scales = "free_y") +
    coord_flip() +
    labs(x = NULL, y = "AUC drop") +
    theme_core()
  save_both(p_state, "figS_lcrf_state_source_audit_core3", 7.4, 3.8)
}

summary <- data.frame(
  figure = c(
    "fig2_core3_data_and_crg_retrieval",
    "fig3_core3_support_corruption_auc_bce",
    "figS_crg_subgroup_support_dependence_core3",
    "fig4_core3_lcrf_counterfactual_delta_auc_bce",
    "fig5_core3_lcrf_same_query_posterior_annotated",
    "figS_crg_local_route_cases_core3",
    "figS_lcrf_specific_student_timeline_core3",
    "figS_lcrf_state_source_audit_core3"
  ),
  claim = c(
    "data phenomenon and CRG retrieval sufficiency",
    "CRG support corruption necessity/control",
    "CRG subgroup support dependence",
    "LCRF actual-vs-counterfactual necessity",
    "LCRF same-query posterior sufficiency",
    "CRG local route cases",
    "LCRF specific student timeline",
    "LCRF learner-state source audit, limited"
  )
)
write.csv(summary, file.path(fig_dir, "paper_figure_summary_core3.csv"), row.names = FALSE)
