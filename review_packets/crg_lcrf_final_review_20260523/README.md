# CRG/LCRF Final Clean Review Packet 20260523

这是当前唯一推荐上传给师兄/GPT Pro 的干净材料包。旧的 core3 审图包和旧中文初稿包已清理，不再作为最新版使用。

## 目录说明

- `paper_draft_cn.md`：中文论文初稿，主线为“概念证据缺口”。
- `figure_prompts_cn.md`：后续用 image2/Figma/PPT 重画问题图、框架图、证据链图的提示词。
- `docs/top20_cd_paper_story_review.md`：20 篇 2025-2026 CD 顶会/顶刊论文复盘。
- `docs/crg_lcrf_paper_outline.md`：当前论文大纲、图表计划和 claim 边界。
- `docs/crg_lcrf_core3_review_packet.md`：core3 机制实验证据边界摘要。
- `papers_original_pdf/`：20 篇原始论文 PDF 和下载 manifest。
- `figures_main/`：正文候选图。
- `figures_appendix/`：附录候选图。
- `figures_editable_svg/`：GPT 生成的可编辑 SVG 草图。
- `figures_editable_pdf/`：SVG 草图对应 PDF。
- `tables/`：核心实验数据表。

## 写作边界

1. 主问题写“概念证据缺口”，不要窄化为题内多知识点共现。
2. CRG 是主模块：train-only concept reachability roadmap。
3. LCRF 是副模块：固定 CRG support 内的 learner-conditioned posterior filter。
4. sequence transition 只能写 empirical learning route，不能写 prerequisite。
5. Junyi 主讲 CRG 数据现象和 retrieval，不强讲 LCRF。
6. CRG necessity 是 dataset-dependent support dependence：assist_17 最强，assist_09 support-only，junyi weak。
7. state-source audit 只能作为 limitation，不能声称完全排除 student-ID shortcut。
