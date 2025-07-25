from pathlib import Path
from nbformat import read

# Load the uploaded notebook
notebook_path = Path("C:\\Users\\Sanskar Gupta\\OneDrive\\Desktop\\ML Projects\\Credit Card Fraud Detection Model\\CREDIT CARD FRAUD.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = read(f, as_version=4)

# Extract code cells
code_cells = [cell["source"] for cell in notebook.cells if cell.cell_type == "code"]
code_combined = "\n\n".join(code_cells)
code_combined[:2000]  # Show preview of the code for analysis
