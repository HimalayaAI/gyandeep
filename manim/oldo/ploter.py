# plotter.py
from manim import *

class Plotter:

#*parameter haru le ko
    @staticmethod
    def plot_data(
        scene,
        x,
        y,
        connection=False,
        name=None,
        point_color=BLUE,
        line_color=YELLOW,
        label_color=WHITE,
        show_axes=True,
        axis_padding=1,
    ):

#*graph ko, x ra y ko range calculate garne
        x_min, x_max = min(x) - axis_padding, max(x) + axis_padding
        y_min, y_max = min(y) - axis_padding, max(y) + axis_padding

        axes = None
        if show_axes:
            axes = Axes(
                x_range=[x_min, x_max, 1],
                y_range=[y_min, y_max, 1],
                axis_config={"include_numbers": True},
            )


#* plot scene create gareko -->
        def to_scene_coords(xi, yi):
            return axes.c2p(xi, yi) if axes else [xi, yi, 0]


        plot_objects = VGroup()
        if axes:
            plot_objects.add(axes)

        points = [to_scene_coords(xi, yi) for xi, yi in zip(x, y)]
        dots = VGroup(*[Dot(point=p, color=point_color) for p in points])
        plot_objects.add(dots)


        if not connection and name:
            if isinstance(name, str):
                name = [name] * len(points)

            if len(name) == len(points):
                labels = VGroup(
                    *[
                        Text(str(label), font_size=24, color=label_color)
                        .next_to(dots[i], UR, buff=0.1)
                        for i, label in enumerate(name)
                    ]
                )
                plot_objects.add(labels)


        if connection:
            line = VMobject(color=line_color)
            line.set_points_as_corners(points)
            plot_objects.add(line)

            if isinstance(name, str) and name:
                connection_label = Text(
                    name, font_size=28, color=label_color
                ).move_to(line.get_center() + UP * 0.3)
                plot_objects.add(connection_label)

        return plot_objects