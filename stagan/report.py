from svglib.svglib import svg2rlg
import markdown
from weasyprint import HTML
import xml.etree.ElementTree as ET

#
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.pagesizes import A4
#
#
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet

md_text = """
# Výpočet příjmů z předmětů

- **příjem na hodinu** : příjem předmětu / odhad počtu hodin
- **odhad počtu hodin** : podle rozpočtu FM v daném roce; = 1.2 (počet přednášek) + 1.0 (počet cvičení pro kruhy 1 a 2) + 0.8 (počet cvičení pro kruhu >3) 
- **příjem předmětu** : odpovídá metodice rektorátu pro dělení peněz na programy a katedry; suma přes programy ( počet studentů programu na předmětu * váha předmětu v programu * příjem programu)
- **váha_předmětu v programu** : počet kreditů předmětu * počet studentů programu na předmětu * KEN fakulty předmětu / suma vah přes program
- **příjem programu** : koeficient ekonomické náročnosti (KEN) programu * počet studentů * normativ
- **KEN programu** : stanoven MŠMT pro tématické zaměření programu, dáno akreditací
- **KEN fakulty** : průměr přes programy na fakultě vážený počtem studentů v daném roce
- **počet studentů** : počet studentů k 31.10. (s drobnými opravami)
- **normativ** : příjem na jednoho normativního studenta pro rok 2024 (=44 000 Kč)
- předměty jsou seřazeny sestupně podle průměru přes roky; Inf a NaN hodnoty viz níže.

# Poznámky

- zdrojem je především webové rozhraní STAGu
- *odhad počtu hodin* je převzatý z rozpočtů FM, ale tam občas chybí předměty, které byly vyučované (podle STAGu)
- chybí kontrola počtu hodin v rozpočtu FM vůči STAGu, zároveň rozpočet FM obsahuje dílčí korekce oproti údajům ve STAGu
- některé předměty rozvrhových akcí nejsou STAGem reportovány u žádného oboru; tyto předměty ve výpisu chybí
- pokud není v daném roce předmět nalezen v rozpočtu FM j emu přiřazen nulový počet hodin
- *příjem na hodinu* je nastaven na maximum rozsahu grafu pokud je předmět vyučován podle STAGu, ale má nulový počet odučených hodin podle rozpočtu
- *příjem na hodinu* je nastaven na NaN a ignorován při výpočtu průměru přes roky pokud je příjem i počet hodin nulový
"""


def make_report(plot_buffer, pdf_path):
    # Markdown-like content
    html_text = markdown.markdown(md_text)
    svg_plot = plot_buffer.getvalue().decode('utf-8')
    root = ET.fromstring(svg_plot)
    height_attr = root.get('height')

    height_in_inches = 0
    if height_attr:
        if 'pt' in height_attr:
            height_in_points = float(height_attr.replace('pt', ''))
            height_in_inches = height_in_points / 72  # Convert points to inches
        elif 'px' in height_attr:
            height_in_pixels = float(height_attr.replace('px', ''))
            height_in_inches = height_in_pixels / 96  # Assuming 96 dpi
    height_in_inches += 2

    # Convert SVG to ReportLab Drawing object
    #drawing = svg2rlg(plot_buffer)
    #
    #
    # # Prepare the document elements
    # elements = []
    #
    # # Add the drawing (plot) to the elements
    # elements.append(drawing)
    # elements.append(Spacer(1, 12))  # Add space after the plot
    #
    #
    # print(html_text)
    # # Add the formatted text
    # styles = getSampleStyleSheet()
    # style_normal = styles["BodyText"]
    # elements.append(Paragraph(html_text, style_normal))
    #
    # # Build the PDF
    # content_height = sum([f.wrap(A4[0], A4[1])[1] if hasattr(f, 'wrap') else f.height for f in elements])
    # doc = SimpleDocTemplate(str(pdf_path),  pagesize=(A4[0], content_height + 100))
    # doc.build(elements)

    # Combine the HTML content and SVG into a single HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PDF with Long Plot</title>
    <style>
        /* Set custom page size with long height */
        @page {{
            size: 8.5in {height_in_inches}in;  /* Set height to fit your content */
            margin: 1in;
        }}
        body {{
            width: 6.5in; /* Content width to fit page margins */
        }}
        /* Style for shifting SVG content */
        .shifted-svg {{
            margin-left: -1in;  /* Adjust the value as needed for the left shift */
        }}
    </style>
</head>
    <body>
        <div class="shifted-svg">
            {svg_plot}  <!-- Insert the SVG content directly here -->
        </div>    
        {html_text}
    </body>
    </html>
    """

    # Render the HTML to a PDF file
    HTML(string=full_html).write_pdf(pdf_path)
