library(ggplot2)
library(scales)

out_dir <- "docs/paper_review_2025_2026/icdm2026_template/IEEEtran_CTAN/IEEEtran/figures"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

profile <- data.frame(
  dataset = rep(c("ASSIST09", "Junyi", "ASSIST17"), each = 5),
  metric = rep(c("Single", "Direct-unseen", "Bridge-only", "Item edge", "Seq edge"), times = 3),
  value = c(
    82.8, 3.1, 3.1, 0.9, 64.3,
    100.0, 100.0, 99.97, 0.0, 25.2,
    78.3, 2.8, 2.8, 6.2, 76.3
  ),
  family = rep(c("Coverage", "Gap", "Gap", "Route", "Route"), times = 3)
)

profile$dataset <- factor(profile$dataset, levels = c("ASSIST09", "Junyi", "ASSIST17"))
profile$metric <- factor(
  profile$metric,
  levels = c("Single", "Direct-unseen", "Bridge-only", "Item edge", "Seq edge")
)

scale_info <- data.frame(
  dataset = factor(c("ASSIST09", "Junyi", "ASSIST17"), levels = c("ASSIST09", "Junyi", "ASSIST17")),
  x = 0,
  y = 5.75,
  label = c(
    "2.5k L | 17.7k I | 123 C | 267k R | H=41",
    "10.0k L | 706 I | 706 C | 354k R | H=25",
    "1.7k L | 3.2k I | 102 C | 390k R | H=148"
  )
)

pal <- c(
  "Coverage" = "#BDBDBD",
  "Gap" = "#5D86C5",
  "Route" = "#3AA6A3"
)

p <- ggplot(profile, aes(x = value, y = metric, fill = family)) +
  geom_col(width = 0.54, colour = "white", linewidth = 0.18) +
  geom_text(
    aes(label = ifelse(value >= 99.95, "100", sprintf("%.1f", value))),
    hjust = -0.12,
    size = 2.05,
    family = "Arial",
    colour = "#2B2B2B"
  ) +
  geom_text(
    data = scale_info,
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 0.5,
    size = 1.75,
    family = "Arial",
    colour = "#4A4A4A"
  ) +
  facet_grid(dataset ~ ., switch = "y") +
  scale_fill_manual(values = pal, breaks = c("Coverage", "Gap", "Route")) +
  scale_x_continuous(
    limits = c(0, 108),
    breaks = c(0, 50, 100),
    labels = c("0", "50", "100%"),
    expand = expansion(mult = c(0, 0))
  ) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme_classic(base_family = "Arial", base_size = 6.2) +
  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 5.9, colour = "#2B2B2B"),
    axis.text.x = element_text(size = 5.8, colour = "#2B2B2B"),
    axis.ticks.x = element_line(linewidth = 0.25, colour = "#777777"),
    axis.line.x = element_line(linewidth = 0.3, colour = "#777777"),
    strip.placement = "outside",
    strip.background = element_blank(),
    strip.text.y.left = element_text(size = 6.6, face = "bold", angle = 0, hjust = 0),
    panel.spacing.y = unit(1.1, "mm"),
    legend.position = "none",
    plot.margin = margin(1.5, 4.5, 1.5, 1.5, "mm")
  )

pdf_file <- file.path(out_dir, "fig_dataset_evidence_profiles.pdf")
png_file <- file.path(out_dir, "fig_dataset_evidence_profiles.png")

cairo_pdf(pdf_file, width = 3.5, height = 2.72, family = "Arial")
print(p)
dev.off()

ragg::agg_png(png_file, width = 3.5, height = 2.72, units = "in", res = 600, background = "white")
print(p)
dev.off()

cat("saved:", normalizePath(pdf_file, winslash = "/"), "\n")
cat("saved:", normalizePath(png_file, winslash = "/"), "\n")
