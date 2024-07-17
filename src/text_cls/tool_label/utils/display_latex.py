import numpy as np
from latexifier import latexify
from IPython.display import Latex
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')

# Create a sample "Matrix" (NumPy 2-D array)
MM = np.arange(1,13).reshape(3,4)

# Specify that we want newlines and we want curly braces ("Bmatrix") as the matrix display type.
converted = latexify(MM, newline=True, arraytype="Bmatrix")

print("Showing converted LaTeX string:\n")
print(converted)

print("\n\nDisplaying it using the Latex method of IPython.display:\n")
Latex(converted)