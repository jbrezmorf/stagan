import matplotlib.pyplot as plt
from io import BytesIO

# Create the plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Sample Plot")
ax.set_xlabel("X-axis Label")
ax.set_ylabel("Y-axis Label")

# Save the plot to a BytesIO buffer in SVG format
plot_buffer = BytesIO()
fig.savefig(plot_buffer, format='svg')
plot_buffer.seek(0)
plt.close(fig)


from svglib.svglib import svg2rlg

# Convert SVG to ReportLab Drawing object
drawing = svg2rlg(plot_buffer)


from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create the PDF document
pdf_path = "final_report_with_vector_plot.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=letter)

# Prepare the document elements
elements = []

# Add the drawing (plot) to the elements
elements.append(drawing)
elements.append(Spacer(1, 12))  # Add space after the plot

# Define the styles for the text
styles = getSampleStyleSheet()
style_normal = styles["BodyText"]

# Markdown-like content
markdown_text = """
<h1>Explanation of the Plot</h1>
<p>This plot demonstrates a simple line graph with customized labels and title.</p>
<ul>
    <li><b>X-Axis</b>: Represents the sample X values used in the plot.</li>
    <li><b>Y-Axis</b>: Represents the corresponding Y values.</li>
    <li><b>Purpose</b>: This plot is intended as an example of embedding Matplotlib graphics into a PDF.</li>
</ul>
Trying some equations:
$$
x = \frac{y+z}{y-z}
$$
<p>You can add more details here using <i>italicized</i> or <b>bold</b> text, as well as bullet points for structured information.</p>
"""

# Add the formatted text
elements.append(Paragraph(markdown_text, style_normal))

# Build the PDF
doc.build(elements)

print(f"PDF report saved to {pdf_path}")
