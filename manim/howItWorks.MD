User_input --> API --> data --> scrape_CSV --> CSV --> create_animation

Go inside the the manim dir and run
``` manim -pqh animator.py EquationScene```


When user gives the input it is send to sarvam API then it passed through response scraper which format the text for the csv file so that it can be converted to manim animation



CSV format:
equation: the main equation
reasoning: it's the explantaion of the equation and overall step
plot: say's if we want a old graph new graph or no graph
x: the value of x
y: The value of y
connection: If we want to connect points or not
name: the point name

