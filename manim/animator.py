from pathlib import Path
import pandas as pd
import manim
from oldo.ploter import Plotter

csv_path = Path("solution.csv")
if not csv_path.exists():
    csv_path = Path("manim/files/solution.csv")

df = pd.read_csv(csv_path, skipinitialspace=True)
df.rename(columns=lambda x: str(x).strip(), inplace=True)

from manim import *

class EquationScene(Scene):
    def construct(self):
        equations_group = VGroup()
        active_graph = None

        #* Solution, line wise animate gareko 


        for index, row in df.iterrows():
            eq_val = row.get("equation")
            if pd.isna(eq_val):
                continue
            equation_str = str(eq_val).strip()
            if not equation_str or equation_str == "nan":
                continue
                
            #! CSV bata reasoning, plot type, x ra y values read gareko

            reasoning = str(row.get("reasoning")).strip()
            plot_type = str(row.get("plot")).strip().lower()


            x_str = "" if pd.isna(row.get("x")) else str(row.get("x")).strip()
            y_str = "" if pd.isna(row.get("y")) else str(row.get("y")).strip()

            connection_val = str(row.get("connection", "False")).strip().lower()
            connection = connection_val == "true"
            name = "" if pd.isna(row.get("name")) else str(row.get("name")).strip()


#--------------------------------------------------------------------------------------------------------------------
            #* Solution ko equation animate gareko

            eq = MathTex(equation_str, color=WHITE).scale(0.8)
            
            if len(equations_group) == 0:
                eq.to_edge(LEFT + UP, buff=1.0)
            else:
                eq.next_to(equations_group, DOWN, buff=0.5, aligned_edge=LEFT)
            
            self.play(Write(eq), run_time=1)
            equations_group.add(eq)

            #* Equation explain gareko
            if reasoning and reasoning.lower() != "nah":
                reasoning_text = Text(
                    reasoning,
                    font_size=28,
                    color=YELLOW,
                ).to_edge(UP, buff=0.5)
                self.play(FadeIn(reasoning_text, shift=DOWN * 0.2))
                self.wait(1)
                self.play(FadeOut(reasoning_text, shift=UP * 0.2))

#---------------------------------------------------------------------------------------------------------------------
            #* Plot type check gareko ra graph plot gareko

            if plot_type in ["old", "new"] and x_str and y_str:
                try:
                    x_values = list(map(float, x_str.split(";")))
                    y_values = list(map(float, y_str.split(";")))


                    #! params haru deko
                    new_graph = Plotter.plot_data(
                        self,
                        x_values,
                        y_values,
                        connection=connection,
                        name=name,
                        point_color=BLUE if plot_type == "old" else RED,
                        line_color=GOLD if plot_type == "old" else GREEN,
                        label_color=WHITE,
                        show_axes=True,
                        axis_padding=1,
                    )


                    #! graph ko position ra size
                    new_graph.scale(0.6).to_edge(RIGHT, buff=0.5)
                    

                    
                    #* graph animate gareko
                    if active_graph:
                        self.play(ReplacementTransform(active_graph, new_graph))
                    else:
                        self.play(Create(new_graph), run_time=2)
                    
                    active_graph = new_graph
                    self.wait(1)
                except Exception as e:
                    print(f"Error plotting row {index}: {e}")



            #! eqution haru 5 line ko pugda scene clear gareko
            if len(equations_group) >= 5:
                self.wait(2)
                self.play(
                    FadeOut(equations_group), 
                    FadeOut(active_graph) if active_graph else Wait(run_time=0)
                )
                equations_group = VGroup()
                active_graph = None
            
            self.wait(0.5)

        self.wait(2)
