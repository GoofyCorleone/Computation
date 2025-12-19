from manim import *

class prueba(Scene):
    def construct(self):
        circulo=Circle()
        self.play(Write(circulo))
        self.wait(3)