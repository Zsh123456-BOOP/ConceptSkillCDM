# CRG/LCRF 中文论文初稿包

本目录是给师兄或 GPT Pro 复审论文写法用的中文初稿包。仓库内对应路径为：

- 目录：`review_packets/crg_lcrf_cn_draft_review_20260523/`
- 压缩包：`review_packets/crg_lcrf_cn_draft_review_20260523.zip`
- docs 同步稿：`docs/paper_review_2025_2026/crg_lcrf_cn_paper_draft.md`
- docs 图提示词：`docs/paper_review_2025_2026/crg_lcrf_cn_figure_prompts.md`

本包包含：

- `paper_draft_cn.md`：完整中文论文初稿，含数据表和公式。
- `figure_prompts_cn.md`：后续使用 image2 / Figma / PPT 重画问题图和模型图的提示词。
- `figures_svg/`：可编辑 SVG 草图。
- `figures_pdf/`：对应 PDF 图，包括 SVG 生成图和部分已有实验图 PDF 版本。
- `figures_existing_png/`：从证据包复制的已有主图/附录图 PNG。
- `tables/`：关键数据 CSV，便于核对表格数据来源。

注意：

1. 本稿采用“概念证据缺口”作为主问题，避免把论文写成依赖题内多知识点共现。
2. CRG 是主模块，LCRF 是固定 support 内的个性化过滤模块。
3. 不要声称 sequence transition 是 prerequisite。
4. 不要声称 CRG 在所有数据集上证明强于任意图。
5. 不要声称当前实验完全排除了 student-ID shortcut。
