from pathlib import Path

import win32com.client


OUT = Path(r"C:\Users\zsh\Downloads\concept_evidence_gap_problem_first_editable.pptx")
PNG = Path(r"C:\Users\zsh\Downloads\concept_evidence_gap_problem_first_editable.png")

SLIDE_W = 960
SLIDE_H = 540

MSO_FALSE = 0
MSO_TRUE = -1
PP_BLANK = 12
PP_SAVE_AS_OPENXML = 24

SHAPE_RECT = 1
SHAPE_ROUND_RECT = 5
SHAPE_OVAL = 9
SHAPE_RIGHT_ARROW = 33
SHAPE_TRIANGLE = 7
SHAPE_FLOWCHART_MAGNETIC_DISK = 133

BLUE = "#0b63ce"
BLUE_LIGHT = "#f4f8ff"
BLUE_STROKE = "#94bdf2"
RED = "#e60000"
RED_LIGHT = "#fff3f3"
RED_STROKE = "#ffc4c4"
ORANGE = "#ff5a00"
GREEN = "#2fb85d"
GREEN_LIGHT = "#eef9f1"
INK = "#111111"
MUTED = "#555555"
LINE = "#c8c8c8"
PANEL = "#fbfbfb"
PANEL_STROKE = "#555555"
GRAY_FILL = "#f2f2f2"


def rgb(hex_color: str) -> int:
    aliases = {"white": "#ffffff", "black": "#000000"}
    hex_color = aliases.get(hex_color.lower(), hex_color)
    h = hex_color.lstrip("#")
    return int(h[0:2], 16) + int(h[2:4], 16) * 256 + int(h[4:6], 16) * 65536


def style_shape(shape, fill=None, line=None, weight=1.0, transparency=0.0, dash=False):
    if fill is None:
        shape.Fill.Visible = MSO_FALSE
    else:
        shape.Fill.Visible = MSO_TRUE
        shape.Fill.ForeColor.RGB = rgb(fill)
        shape.Fill.Transparency = transparency
    if line is None:
        shape.Line.Visible = MSO_FALSE
    else:
        shape.Line.Visible = MSO_TRUE
        shape.Line.ForeColor.RGB = rgb(line)
        shape.Line.Weight = weight
        if dash:
            shape.Line.DashStyle = 4


def add_text(slide, x, y, w, h, text, size=16, color=INK, bold=False, align=1, name=None):
    shape = slide.Shapes.AddTextbox(1, x, y, w, h)
    if name:
        shape.Name = name
    tf = shape.TextFrame
    tf.MarginLeft = 0
    tf.MarginRight = 0
    tf.MarginTop = 0
    tf.MarginBottom = 0
    tf.WordWrap = MSO_TRUE
    rng = tf.TextRange
    rng.Text = text
    rng.Font.Name = "Arial"
    rng.Font.Size = size
    rng.Font.Color.RGB = rgb(color)
    rng.Font.Bold = MSO_TRUE if bold else MSO_FALSE
    rng.ParagraphFormat.Alignment = align
    shape.Line.Visible = MSO_FALSE
    shape.Fill.Visible = MSO_FALSE
    return shape


def add_panel(slide, x, y, w, h, title, accent, idx, fill=PANEL, group_names=None):
    names = group_names if group_names is not None else []
    panel = slide.Shapes.AddShape(SHAPE_ROUND_RECT, x, y, w, h)
    panel.Name = f"{title}_panel"
    style_shape(panel, fill, PANEL_STROKE, 1.2)
    names.append(panel.Name)

    badge = slide.Shapes.AddShape(SHAPE_OVAL, x + 16, y + 16, 23, 23)
    badge.Name = f"{title}_badge"
    style_shape(badge, accent, "white", 1.4)
    names.append(badge.Name)
    names.append(add_text(slide, x + 16, y + 18, 23, 18, str(idx), 12, "white", True, 2, f"{title}_badge_text").Name)
    names.append(add_text(slide, x + 46, y + 15, w - 58, 26, title, 15.5, accent, True, 1, f"{title}_title").Name)
    return names


def add_circle_label(slide, cx, cy, r, text, fill, line, text_color=INK, bold=True, name_prefix="node", group_names=None):
    names = group_names if group_names is not None else []
    c = slide.Shapes.AddShape(SHAPE_OVAL, cx - r, cy - r, 2 * r, 2 * r)
    c.Name = f"{name_prefix}_circle"
    style_shape(c, fill, line, 1.3)
    names.append(c.Name)
    t = add_text(slide, cx - r, cy - 8, 2 * r, 16, text, 11, text_color, bold, 2, f"{name_prefix}_text")
    names.append(t.Name)
    return names


def add_line(slide, x1, y1, x2, y2, color=LINE, weight=1.5, dash=False, end_arrow=False, name=None):
    line = slide.Shapes.AddLine(x1, y1, x2, y2)
    if name:
        line.Name = name
    line.Line.ForeColor.RGB = rgb(color)
    line.Line.Weight = weight
    if dash:
        line.Line.DashStyle = 4
    if end_arrow:
        line.Line.EndArrowheadStyle = 3
    return line


def add_arrow(slide, x, y, w, h, name):
    arrow = slide.Shapes.AddShape(SHAPE_RIGHT_ARROW, x, y, w, h)
    arrow.Name = name
    style_shape(arrow, "#d9d9d9", None, 0)
    return arrow


def group(slide, names, group_name):
    try:
        g = slide.Shapes.Range(names).Group()
        g.Name = group_name
    except Exception:
        pass


def build():
    pp = win32com.client.Dispatch("PowerPoint.Application")
    pp.Visible = True
    pres = pp.Presentations.Add()
    try:
        pres.PageSetup.SlideWidth = SLIDE_W
        pres.PageSetup.SlideHeight = SLIDE_H
        slide = pres.Slides.Add(1, PP_BLANK)
        bg = slide.Shapes.AddShape(SHAPE_RECT, 0, 0, SLIDE_W, SLIDE_H)
        bg.Name = "background"
        style_shape(bg, "white", None)

        add_text(slide, 34, 26, 520, 32, "Figure 1. Concept Evidence Gap.", 22, INK, False, 1, "figure_title")
        add_text(
            slide,
            34,
            58,
            760,
            24,
            "The target concept may be absent from a learner's history; the model recovers usable evidence through route support and learner-state filtering.",
            10.8,
            MUTED,
            False,
            1,
            "figure_subtitle",
        )

        y, h = 105, 315
        x1, w1 = 36, 205
        x2, w2 = 276, 170
        x3, w3 = 482, 222
        x4, w4 = 740, 184

        # Panel 1: Learner history
        n1 = add_panel(slide, x1, y, w1, h, "Learner history", ORANGE, 1, group_names=[])
        n1.append(add_text(slide, x1 + 26, y + 65, w1 - 52, 24, "Observed concepts", 13.5, INK, True, 2, "history_label").Name)
        row_y = [205, 250, 295, 340]
        labels = ["c_1", "c_2", "c_3", "c_5"]
        for i, (yy, lab) in enumerate(zip(row_y, labels), start=1):
            n1.append(add_text(slide, x1 + 32, yy - 8, 22, 16, f"t{i}", 10.5, MUTED, True, 1, f"history_t{i}").Name)
            doc = slide.Shapes.AddShape(SHAPE_RECT, x1 + 62, yy - 18, 27, 34)
            doc.Name = f"history_doc_{i}"
            style_shape(doc, "white", "#333333", 1.0)
            n1.append(doc.Name)
            for off in (0, 7, 14):
                ln = add_line(slide, x1 + 68, yy - 9 + off, x1 + 84, yy - 9 + off, "#333333", 0.7, name=f"history_doc_line_{i}_{off}")
                n1.append(ln.Name)
            n1 = add_circle_label(slide, x1 + 128, yy, 16, lab, "#f6f1e8", "#8a8a8a", MUTED, True, f"history_{lab}", n1)
            ok = slide.Shapes.AddShape(SHAPE_OVAL, x1 + 169, yy - 10, 20, 20)
            ok.Name = f"history_seen_{i}"
            style_shape(ok, GREEN, None)
            n1.append(ok.Name)
            n1.append(add_text(slide, x1 + 169, yy - 11, 20, 18, "✓", 12, "white", True, 2, f"history_check_{i}").Name)
        n1.append(add_text(slide, x1 + 22, y + 275, w1 - 44, 26, "No direct record for c_q", 11.5, BLUE, True, 2, "history_no_cq").Name)
        group(slide, n1, "History_group")

        # Panel 2: Gap
        n2 = add_panel(slide, x2, y, w2, h, "Evidence gap", RED, 2, group_names=[])
        n2.append(add_text(slide, x2 + 22, y + 68, w2 - 44, 38, "Query target", 13.5, INK, True, 2, "gap_query_label").Name)
        n2 = add_circle_label(slide, x2 + w2 / 2, y + 145, 29, "c_q", "#eaf3ff", BLUE, BLUE, True, "gap_target", n2)
        dl = add_line(slide, x2 + 45, y + 210, x2 + w2 - 45, y + 210, RED, 2.3, dash=True, end_arrow=True, name="gap_direct_missing_line")
        n2.append(dl.Name)
        n2.append(add_text(slide, x2 + 26, y + 224, w2 - 52, 38, "direct evidence missing", 12.5, RED, True, 2, "gap_missing_text").Name)
        cross1 = add_line(slide, x2 + 76, y + 199, x2 + 96, y + 219, RED, 2.3, name="gap_cross_1")
        cross2 = add_line(slide, x2 + 96, y + 199, x2 + 76, y + 219, RED, 2.3, name="gap_cross_2")
        n2.extend([cross1.Name, cross2.Name])
        n2.append(add_text(slide, x2 + 20, y + 278, w2 - 40, 22, "Student-specific unseen concept", 10.5, MUTED, False, 2, "gap_scope").Name)
        group(slide, n2, "Gap_group")

        # Panel 3: CRG
        n3 = add_panel(slide, x3, y, w3, h, "CRG route support", BLUE, 3, BLUE_LIGHT, group_names=[])
        n3.append(add_text(slide, x3 + 22, y + 62, w3 - 44, 34, "Retrieve candidate routes from observed concepts to c_q", 12.5, BLUE, True, 2, "crg_subtitle").Name)
        left_nodes = [(x3 + 48, y + 150, "c_1"), (x3 + 48, y + 198, "c_2"), (x3 + 48, y + 246, "c_3")]
        mid_nodes = [(x3 + 125, y + 174, "b_1"), (x3 + 125, y + 226, "b_2")]
        tgt = (x3 + 186, y + 200, "c_q")
        for idx, (cx, cy, lab) in enumerate(left_nodes):
            n3 = add_circle_label(slide, cx, cy, 17, lab, "#f6f1e8", "#9a9a9a", MUTED, True, f"crg_hist_{idx}", n3)
        for idx, (cx, cy, lab) in enumerate(mid_nodes):
            n3 = add_circle_label(slide, cx, cy, 13, lab, GREEN_LIGHT, GREEN, GREEN, True, f"crg_bridge_{idx}", n3)
        n3 = add_circle_label(slide, tgt[0], tgt[1], 19, tgt[2], "#eaf3ff", BLUE, BLUE, True, "crg_target", n3)
        line_specs = [
            (*left_nodes[0][:2], *mid_nodes[0][:2]),
            (*left_nodes[1][:2], *mid_nodes[0][:2]),
            (*left_nodes[2][:2], *mid_nodes[1][:2]),
            (*mid_nodes[0][:2], *tgt[:2]),
            (*mid_nodes[1][:2], *tgt[:2]),
        ]
        for i, (xa, ya, xb, yb) in enumerate(line_specs):
            ln = add_line(slide, xa + 17, ya, xb - 13, yb, GREEN if i >= 3 else "#999999", 1.6, dash=i < 3, end_arrow=i >= 3, name=f"crg_route_{i}")
            n3.append(ln.Name)
        n3.append(add_text(slide, x3 + 36, y + 276, w3 - 72, 24, "bridgeable evidence gap", 11.8, GREEN, True, 2, "crg_bridge_label").Name)
        group(slide, n3, "CRG_group")

        # Panel 4: LCRF + output
        n4 = add_panel(slide, x4, y, w4, h, "LCRF filtering", RED, 4, RED_LIGHT, group_names=[])
        n4.append(add_text(slide, x4 + 18, y + 62, w4 - 36, 34, "Reweight routes by learner state", 12.5, RED, True, 2, "lcrf_subtitle").Name)
        # state bar
        for i, col in enumerate(["#ffdada", "#e95050", "#ff9999", "#ffffff", "#ffffff"]):
            box = slide.Shapes.AddShape(SHAPE_RECT, x4 + 32 + i * 22, y + 124, 22, 20)
            box.Name = f"lcrf_state_{i}"
            style_shape(box, col, "#e64c4c", 0.8)
            n4.append(box.Name)
        n4.append(add_text(slide, x4 + 143, y + 125, 20, 18, "...", 12, MUTED, False, 1, "lcrf_state_more").Name)
        funnel = slide.Shapes.AddShape(SHAPE_TRIANGLE, x4 + 63, y + 169, 64, 58)
        funnel.Name = "lcrf_funnel"
        funnel.Rotation = 180
        style_shape(funnel, "#ffe4e4", RED, 2.0)
        n4.append(funnel.Name)
        # weighted supports
        for i, (yy, ww, alpha) in enumerate([(218, 74, 0), (246, 34, 0)]):
            ln = add_line(slide, x4 + 28, y + yy, x4 + 82, y + yy, "#333333", 0.9, name=f"lcrf_mini_route_{i}")
            n4.append(ln.Name)
            for j, cx in enumerate([x4 + 28, x4 + 55, x4 + 82]):
                n4 = add_circle_label(slide, cx, y + yy, 5.5, "", GREEN if j == 1 else "white", "#333333", INK, False, f"lcrf_node_{i}_{j}", n4)
            bar = slide.Shapes.AddShape(SHAPE_RECT, x4 + 105, y + yy - 6, ww, 12)
            bar.Name = f"lcrf_weight_{i}"
            style_shape(bar, "#ef3b3b", None, transparency=0.15 + i * 0.12)
            n4.append(bar.Name)
        out = slide.Shapes.AddShape(SHAPE_ROUND_RECT, x4 + 24, y + 270, w4 - 48, 36)
        out.Name = "diagnosis_output_box"
        style_shape(out, "white", RED_STROKE, 1.0)
        n4.append(out.Name)
        n4.append(add_text(slide, x4 + 38, y + 279, w4 - 76, 18, "Diagnosis for c_q", 11.8, INK, True, 2, "diagnosis_output_text").Name)
        group(slide, n4, "LCRF_Diagnosis_group")

        # Arrows between panels
        a1 = add_arrow(slide, x1 + w1 + 11, y + 146, 29, 32, "arrow_history_gap")
        a2 = add_arrow(slide, x2 + w2 + 12, y + 146, 29, 32, "arrow_gap_crg")
        a3 = add_arrow(slide, x3 + w3 + 12, y + 146, 29, 32, "arrow_crg_lcrf")

        # Bottom takeaway
        box = slide.Shapes.AddShape(SHAPE_ROUND_RECT, 92, 455, 776, 52)
        box.Name = "takeaway_box"
        style_shape(box, "#fafafa", "#d0d0d0", 1.0)
        add_text(slide, 124, 468, 170, 22, "Problem:", 13, RED, True, 1, "takeaway_problem")
        add_text(slide, 198, 468, 280, 22, "c_q is absent from direct history.", 13, INK, True, 1, "takeaway_problem_text")
        add_text(slide, 493, 468, 155, 22, "Solution:", 13, BLUE, True, 1, "takeaway_solution")
        add_text(slide, 560, 468, 260, 22, "CRG supplies routes; LCRF filters them.", 13, INK, True, 1, "takeaway_solution_text")

        pres.SaveAs(str(OUT), PP_SAVE_AS_OPENXML)
        slide.Export(str(PNG), "PNG", 1920, 1080)
        print(f"pptx={OUT}")
        print(f"preview={PNG}")
        print(f"shapes={slide.Shapes.Count}")
    finally:
        pres.Close()
        pp.Quit()


if __name__ == "__main__":
    build()
