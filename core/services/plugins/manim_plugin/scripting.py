from __future__ import annotations
from textwrap import dedent

from .planning import fallback_plan
from .text_utils import clip_text, latex_to_text, wrap_text


def build_script_prompt(
    query: str,
    context_text: str,
    style_context: str,
    plan: dict[str, object],
    scene_name: str = "LessonScene",
) -> str:
    import json

    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    return dedent(
        f"""
        You are generating a Manim Community Edition script for a high-school tutoring app.
        Use this style guide context:
        {style_context}

        Student query:
        {query}

        Structured teaching blueprint:
        {plan_json}

        Source context:
        {context_text}

        Requirements:
        - Return only valid Python code in one ```python fenced block.
        - Code must include `from manim import *`.
        - Define class `{scene_name}(MovingCameraScene)`.
        - Keep all essential content inside safe margins; do not place important text at the bottom edge.
        - Use `self.camera.frame.animate...` to follow the active writing area instead of stacking one tall page of text.
        - Replace any placeholder "Visual Model" box with a small top-right badge that says `GyanDeep`.
        - Use at least two non-text visuals such as a diagram, number line, dots, arrows, shaded regions, or geometric shapes.
        - Break long explanations into short cards or beats; avoid giant text walls and avoid overlap between text and visuals.
        - The animation must teach the actual solution flow (formula + worked example), not just planning text.
        - Keep script robust for low-quality render (`-ql`) and avoid fragile APIs.
        - Use readable text sizes (title >= 40, body >= 26).
        - Include at least 5 explicit `self.wait(...)` pauses.
        - Keep total duration around 18 to 45 seconds.
        - Prefer MovingCameraScene, Text/MathTex, NumberLine, Dot, Line, Polygon, Circle, RoundedRectangle, VGroup, FadeIn, Write, Transform, Create, and `.animate`.
        - Always render latex and formulas under MathTex if possible
        """
    ).strip()


def _scene_lines(
    scene_name: str,
    title: str,
    learning_goal: str,
    formula_latex: str,
    answer_text: str,
    visual_focus: str,
    steps: list[str],
    worked: list[str],
) -> list[str]:
    lines = [
        "from manim import *",
        "",
        f"class {scene_name}(MovingCameraScene):",
        "    def construct(self):",
        "        self.camera.background_color = '#0D1326'",
        "        self.camera.frame.set(width=14.0)",
        "",
        "        def make_card(text, width=7.2, height=1.2, color=BLUE_B, font_size=26, text_color=WHITE):",
        "            label = Text(text, font_size=font_size, color=text_color, line_spacing=0.9).scale_to_fit_width(width - 0.55)",
        "            box = RoundedRectangle(width=width, height=height, corner_radius=0.18, color=color)",
        "            box.set_fill('#111A33', opacity=0.48)",
        "            label.move_to(box)",
        "            return VGroup(box, label)",
        "",
        "        def focus_on(mobject, width=12.0, offset=ORIGIN):",
        "            return self.camera.frame.animate.move_to(mobject.get_center() + offset).set(width=width)",
        "",
        f"        title = Text({repr(title)}, font_size=40, color=BLUE_B).scale_to_fit_width(7.6)",
        f"        goal = Text({repr(learning_goal)}, font_size=28, color=WHITE, line_spacing=0.9).scale_to_fit_width(7.2)",
        "        hero = VGroup(title, goal).arrange(DOWN, aligned_edge=LEFT, buff=0.24)",
        "        hero.to_edge(UL, buff=0.45).shift(DOWN * 0.1)",
        "",
        "        brand_box = RoundedRectangle(width=2.45, height=0.74, corner_radius=0.22, color=YELLOW)",
        "        brand_box.set_fill('#111A33', opacity=0.88)",
        "        brand_text = Text('GyanDeep', font_size=24, color=YELLOW)",
        "        brand = VGroup(brand_box, brand_text)",
        "        brand.move_to(self.camera.frame.get_corner(UR) + LEFT * 1.55 + DOWN * 0.56)",
        "",
        "        visual_panel = RoundedRectangle(width=4.3, height=2.9, corner_radius=0.22, color=BLUE_B)",
        "        visual_panel.set_fill('#111A33', opacity=0.35)",
        "        visual_panel.to_edge(RIGHT, buff=0.45).shift(UP * 0.12)",
    ]

    if visual_focus == "triangle":
        lines.extend(
            [
                "        panel_title = Text('Geometry view', font_size=20, color=TEAL_B).move_to(visual_panel.get_top() + DOWN * 0.28)",
                "        center = visual_panel.get_center() + DOWN * 0.12",
                "        p1 = center + LEFT * 1.2 + DOWN * 0.65",
                "        p2 = center + RIGHT * 1.2 + DOWN * 0.65",
                "        p3 = center + UP * 0.95",
                "        height_foot = p3[0] * RIGHT + p1[1] * UP",
                "        fill_tri = Polygon(p1, p2, p3).set_fill(BLUE_E, opacity=0.22).set_stroke(width=0)",
                "        tri = Polygon(p1, p2, p3, color=YELLOW, stroke_width=4)",
                "        base = Line(p1, p2, color=GREEN_B)",
                "        height = DashedLine(p3, height_foot, color=BLUE_B)",
                "        base_label = Text('b', font_size=18, color=GREEN_B).next_to(base, DOWN, buff=0.08)",
                "        height_label = Text('h', font_size=18, color=BLUE_B).next_to(height, RIGHT, buff=0.08)",
            ]
        )
    elif visual_focus == "circle":
        lines.extend(
            [
                "        panel_title = Text('Circle view', font_size=20, color=TEAL_B).move_to(visual_panel.get_top() + DOWN * 0.28)",
                "        center = visual_panel.get_center() + DOWN * 0.08",
                "        fill_circle = Circle(radius=0.96).move_to(center).set_fill(BLUE_E, opacity=0.18).set_stroke(width=0)",
                "        circle = Circle(radius=0.96, color=YELLOW, stroke_width=4).move_to(center)",
                "        radius = Line(center, center + RIGHT * 0.96, color=GREEN_B)",
                "        radius_label = Text('r', font_size=18, color=GREEN_B).next_to(radius, UP, buff=0.08)",
                "        arc = Arc(radius=0.96, start_angle=0, angle=PI / 2, color=TEAL_B).move_arc_center_to(center)",
            ]
        )
    elif visual_focus == "algebra":
        lines.extend(
            [
                "        panel_title = Text('Equation view', font_size=20, color=TEAL_B).move_to(visual_panel.get_top() + DOWN * 0.28)",
                "        x_box = RoundedRectangle(width=0.9, height=0.7, corner_radius=0.12, color=YELLOW).move_to(visual_panel.get_center() + LEFT * 1.2)",
                "        plus = Text('+', font_size=24, color=WHITE).next_to(x_box, RIGHT, buff=0.16)",
                "        known_box = RoundedRectangle(width=0.9, height=0.7, corner_radius=0.12, color=GREEN_B).next_to(plus, RIGHT, buff=0.16)",
                "        eq_sign = Text('=', font_size=24, color=WHITE).next_to(known_box, RIGHT, buff=0.16)",
                "        result_box = RoundedRectangle(width=1.0, height=0.7, corner_radius=0.12, color=BLUE_B).next_to(eq_sign, RIGHT, buff=0.16)",
                "        x_text = Text('x', font_size=26, color=YELLOW).move_to(x_box)",
                "        known_text = Text('5', font_size=24, color=GREEN_B).move_to(known_box)",
                "        result_text = Text('9', font_size=24, color=BLUE_B).move_to(result_box)",
                "        undo_arrow = Arrow(result_box.get_bottom() + DOWN * 0.06, known_box.get_bottom() + DOWN * 0.06, buff=0.12, color=TEAL_B, stroke_width=5, max_tip_length_to_length_ratio=0.14)",
            ]
        )
    elif visual_focus == "numberline":
        lines.extend(
            [
                "        panel_title = Text('Number line view', font_size=20, color=TEAL_B).move_to(visual_panel.get_top() + DOWN * 0.28)",
                "        line = NumberLine(x_range=[-3, 3, 1], length=3.4, include_tip=True, color=BLUE_B)",
                "        line.move_to(visual_panel.get_center() + DOWN * 0.22)",
                "        real_span = Line(line.n2p(-2.8), line.n2p(2.8), color=TEAL_B, stroke_width=10).set_opacity(0.18)",
                "        integer_dot = Dot(line.n2p(-2), color=ORANGE)",
                "        rational_dot = Dot(line.n2p(0.5), color=GREEN_B)",
                "        irrational_dot = Dot(line.n2p(1.4), color=YELLOW)",
                "        integer_label = Text('integer', font_size=16, color=ORANGE).next_to(integer_dot, UP, buff=0.08)",
                "        rational_label = Text('1/2', font_size=16, color=GREEN_B).next_to(rational_dot, UP, buff=0.08)",
                "        irrational_label = Text('sqrt2', font_size=16, color=YELLOW).next_to(irrational_dot, DOWN, buff=0.08)",
            ]
        )
    else:
        lines.extend(
            [
                "        panel_title = Text('Concept view', font_size=20, color=TEAL_B).move_to(visual_panel.get_top() + DOWN * 0.28)",
                "        center_point = visual_panel.get_center() + DOWN * 0.08",
                "        core_dot = Dot(center_point, color=BLUE_B, radius=0.1)",
                "        left_dot = Dot(center_point + LEFT * 1.1 + UP * 0.35, color=YELLOW, radius=0.09)",
                "        right_dot = Dot(center_point + RIGHT * 1.0 + UP * 0.15, color=GREEN_B, radius=0.09)",
                "        lower_dot = Dot(center_point + DOWN * 0.72, color=ORANGE, radius=0.09)",
                "        link_1 = Line(core_dot.get_center(), left_dot.get_center(), color=BLUE_B)",
                "        link_2 = Line(core_dot.get_center(), right_dot.get_center(), color=BLUE_B)",
                "        link_3 = Line(core_dot.get_center(), lower_dot.get_center(), color=BLUE_B)",
                "        left_label = Text('fact', font_size=16, color=YELLOW).next_to(left_dot, UP, buff=0.08)",
                "        right_label = Text('rule', font_size=16, color=GREEN_B).next_to(right_dot, UP, buff=0.08)",
                "        lower_label = Text('answer', font_size=16, color=ORANGE).next_to(lower_dot, DOWN, buff=0.08)",
            ]
        )

    lines.extend(
        [
            "",
            "        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.9)",
            "        self.play(FadeIn(goal, shift=UP * 0.15), run_time=1.0)",
            "        self.play(FadeIn(brand, shift=LEFT * 0.12), run_time=0.6)",
            "        brand.add_updater(lambda mob: mob.move_to(self.camera.frame.get_corner(UR) + LEFT * 1.55 + DOWN * 0.56))",
            "        self.wait(0.6)",
            "",
            "        self.play(Create(visual_panel), FadeIn(panel_title, shift=UP * 0.1), run_time=0.8)",
        ]
    )

    if visual_focus == "triangle":
        lines.extend(
            [
                "        self.play(FadeIn(fill_tri), Create(tri), Create(base), run_time=1.0)",
                "        self.play(Create(height), FadeIn(base_label), FadeIn(height_label), run_time=0.8)",
                "        self.play(fill_tri.animate.set_fill(BLUE_B, opacity=0.28), run_time=0.5)",
                "        self.play(fill_tri.animate.set_fill(BLUE_E, opacity=0.22), run_time=0.5)",
            ]
        )
    elif visual_focus == "circle":
        lines.extend(
            [
                "        self.play(FadeIn(fill_circle), Create(circle), run_time=1.0)",
                "        self.play(Create(radius), Create(arc), FadeIn(radius_label), run_time=0.8)",
                "        self.play(arc.animate.set_color(YELLOW), run_time=0.5)",
                "        self.play(arc.animate.set_color(TEAL_B), run_time=0.5)",
            ]
        )
    elif visual_focus == "algebra":
        lines.extend(
            [
                "        self.play(Create(x_box), FadeIn(x_text), run_time=0.7)",
                "        self.play(FadeIn(plus), Create(known_box), FadeIn(known_text), FadeIn(eq_sign), Create(result_box), FadeIn(result_text), run_time=0.9)",
                "        self.play(Create(undo_arrow), run_time=0.7)",
                "        self.play(x_box.animate.set_stroke(TEAL_B, width=5), run_time=0.5)",
                "        self.play(x_box.animate.set_stroke(YELLOW, width=4), run_time=0.5)",
            ]
        )
    elif visual_focus == "numberline":
        lines.extend(
            [
                "        self.play(FadeIn(real_span), Create(line), run_time=0.9)",
                "        self.play(FadeIn(integer_dot), FadeIn(rational_dot), FadeIn(irrational_dot), run_time=0.8)",
                "        self.play(FadeIn(integer_label), FadeIn(rational_label), FadeIn(irrational_label), run_time=0.8)",
                "        self.play(integer_dot.animate.scale(1.15), rational_dot.animate.scale(1.15), irrational_dot.animate.scale(1.15), run_time=0.5)",
                "        self.play(integer_dot.animate.scale(0.87), rational_dot.animate.scale(0.87), irrational_dot.animate.scale(0.87), run_time=0.5)",
            ]
        )
    else:
        lines.extend(
            [
                "        self.play(Create(link_1), Create(link_2), Create(link_3), run_time=0.8)",
                "        self.play(FadeIn(core_dot), FadeIn(left_dot), FadeIn(right_dot), FadeIn(lower_dot), run_time=0.8)",
                "        self.play(FadeIn(left_label), FadeIn(right_label), FadeIn(lower_label), run_time=0.8)",
                "        self.play(core_dot.animate.scale(1.25), run_time=0.4)",
                "        self.play(core_dot.animate.scale(0.8), run_time=0.4)",
            ]
        )

    lines.extend(
        [
            "        self.wait(0.8)",
            "",
            "        formula_header = Text('Key Idea', font_size=26, color=YELLOW)",
            "        try:",
            f"            formula = MathTex({repr(formula_latex)}, font_size=24, color=GREEN_B).scale_to_fit_width(6.4)",
            "        except Exception:",
            f"            formula = Text({repr(latex_to_text(formula_latex))}, font_size=24, color=GREEN_B, line_spacing=0.9).scale_to_fit_width(6.4)",
            "        formula_text_group = VGroup(formula_header, formula).arrange(DOWN, aligned_edge=LEFT, buff=0.18)",
            "        formula_box = SurroundingRectangle(formula_text_group, color=BLUE_B, buff=0.28)",
            "        formula_box.set_fill('#111A33', opacity=0.45)",
            "        formula_panel = VGroup(formula_box, formula_text_group)",
            "        formula_panel.next_to(hero, DOWN, buff=0.58, aligned_edge=LEFT)",
            "        concept_arrow = Arrow(formula_panel.get_right() + RIGHT * 0.04, visual_panel.get_left() + LEFT * 0.04, buff=0.12, color=TEAL_B, stroke_width=5, max_tip_length_to_length_ratio=0.12)",
            "",
            "        self.play(focus_on(VGroup(formula_panel, visual_panel), width=12.8, offset=DOWN * 0.08), run_time=1.0)",
            "        self.play(Create(formula_box), run_time=0.7)",
            "        self.play(Write(formula_header), FadeIn(formula, shift=UP * 0.1), run_time=0.9)",
            "        self.play(Create(concept_arrow), run_time=0.7)",
            "        self.wait(1.0)",
            "",
            "        steps_header = Text('Solution Flow', font_size=26, color=TEAL_B)",
            "        steps_header.next_to(formula_panel, DOWN, buff=0.48, aligned_edge=LEFT)",
            "        progress_dots = VGroup(*[Dot(radius=0.07, color=BLUE_D) for _ in range(4)]).arrange(RIGHT, buff=0.12)",
            "        progress_dots.next_to(steps_header, RIGHT, buff=0.3)",
            f"        step_card = make_card({repr(wrap_text(clip_text(steps[0], 95), 40, 2))}, width=7.4, height=1.4, color=TEAL_B, font_size=25)",
            "        step_card.next_to(steps_header, DOWN, buff=0.24, aligned_edge=LEFT)",
            "",
            "        self.play(focus_on(step_card, width=11.8, offset=DOWN * 0.08), run_time=0.9)",
            "        self.play(Write(steps_header), FadeIn(progress_dots), run_time=0.7)",
            "        self.play(Create(step_card[0]), FadeIn(step_card[1], shift=UP * 0.08), progress_dots[0].animate.set_color(YELLOW), run_time=0.9)",
            "        self.wait(1.0)",
        ]
    )

    for idx, step in enumerate(steps[1:], start=2):
        prev_index = idx - 2
        current_index = idx - 1
        lines.extend(
            [
                f"        step_{idx} = make_card({repr(wrap_text(clip_text(step, 95), 40, 2))}, width=7.4, height=1.4, color=TEAL_B, font_size=25)",
                f"        step_{idx}.move_to(step_card)",
                f"        self.play(Transform(step_card, step_{idx}), progress_dots[{prev_index}].animate.set_color(BLUE_D), progress_dots[{current_index}].animate.set_color(YELLOW), run_time=0.95)",
                "        self.wait(1.0)",
            ]
        )

    lines.extend(
        [
            "",
            "        worked_header = Text('Worked Example', font_size=26, color=ORANGE)",
            f"        example_1 = make_card({repr(wrap_text(clip_text(worked[0], 95), 40, 2))}, width=7.4, height=1.15, color=ORANGE, font_size=24)",
            f"        example_2 = make_card({repr(wrap_text(clip_text(worked[1], 95), 40, 2))}, width=7.4, height=1.15, color=ORANGE, font_size=24)",
            f"        example_3 = make_card({repr(wrap_text(clip_text(worked[2], 95), 40, 2))}, width=7.4, height=1.15, color=GREEN_B, font_size=24, text_color=GREEN_B)",
            f"        takeaway = make_card({repr(wrap_text(answer_text, 40, 2))}, width=7.4, height=1.2, color=GREEN_B, font_size=24, text_color=GREEN_B)",
            "        worked_group = VGroup(worked_header, example_1, example_2, example_3, takeaway).arrange(DOWN, aligned_edge=LEFT, buff=0.22)",
            "        worked_group.next_to(step_card, DOWN, buff=0.9, aligned_edge=LEFT)",
            "",
            "        self.play(focus_on(worked_group, width=12.2, offset=DOWN * 0.12), run_time=1.1)",
            "        self.play(Write(worked_header), run_time=0.7)",
            "        self.play(FadeIn(example_1, shift=UP * 0.1), run_time=0.75)",
            "        self.wait(0.7)",
            "        self.play(FadeIn(example_2, shift=UP * 0.1), run_time=0.75)",
            "        self.wait(0.7)",
            "        self.play(FadeIn(example_3, shift=UP * 0.1), run_time=0.8)",
            "        self.wait(0.8)",
            "        self.play(FadeIn(takeaway, shift=UP * 0.1), run_time=0.8)",
            "        self.wait(1.4)",
            "        self.play(focus_on(VGroup(formula_panel, worked_group), width=13.2, offset=DOWN * 0.1), run_time=1.0)",
            "        self.wait(1.0)",
        ]
    )
    return lines


def template_script_from_plan(
    query: str,
    plan: dict[str, object],
    scene_name: str = "LessonScene",
) -> str:
    title = clip_text(str(plan.get("title", "") or "DeepGyan Animation"), 70)
    learning_goal = wrap_text(clip_text(str(plan.get("learning_goal", "") or query), 95), 40, 3)


    formula_latex = str(plan.get("formula_latex", "") or "")
    answer_text = wrap_text(
        clip_text(
            str(plan.get("answer_line", "") or "The answer follows from the core idea in the lesson."),
            120,
        ),
        width=40,
        max_lines=2,
    )

    steps = [clip_text(str(item), 95) for item in (plan.get("steps") or []) if str(item).strip()]
    if len(steps) < 4:
        steps = [clip_text(str(item), 95) for item in fallback_plan(query, "").get("steps", [])]
    steps = steps[:4]

    worked = [clip_text(str(item), 95) for item in (plan.get("worked_example") or []) if str(item).strip()]
    if len(worked) < 3:
        worked = [clip_text(str(item), 95) for item in fallback_plan(query, "").get("worked_example", [])]
    worked = worked[:3]

    visual_focus = str(plan.get("visual_focus", "generic")).strip().lower()
    lines = _scene_lines(scene_name, title, learning_goal, formula_latex, answer_text, visual_focus, steps, worked)
    return "\n".join(lines).strip()


def script_looks_valid(script: str, scene_name: str) -> bool:
    import ast

    if "from manim import" not in script:
        return False
    if "visual model" in script.lower():
        return False
    if "GyanDeep" not in script:
        return False
    if "camera.frame.animate" not in script:
        return False
    if script.count("self.wait(") < 5:
        return False
    try:
        tree = ast.parse(script)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == scene_name:
            base_names: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_names.append(base.attr)
            return any(name in {"Scene", "MovingCameraScene"} for name in base_names)
    return False
