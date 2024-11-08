import attrs
import pandas as pd
from pathlib import Path
import re

script_dir = Path(__file__).parent

@attrs.define
class RozpocetCols:
    predmet: int = 0
    pr_hodin: int = 1
    cv_hodin: int = 2
    n_kruhu: int = 3
    hodino_body: int = 4

def process_excel_file(filename, cols: RozpocetCols):
    # Read the Excel file, sheet '5_Výuka'
    df = pd.read_excel(filename, sheet_name='5_Výuka', header=None)
    # Initialize variables
    katedra = None
    data_rows = []
    column_positions = {}

    # Define the set of header names to look for
    header_names = set([
        'Počet studentů na předmětu',
        'Počet hodin přednášek za celý semestr',
        'Počet hodin cvičení za celý semestr',
        'Počet kroužků ke cvičení ze STAGu',
        'Body celkem'
    ])

    # Loop over each row in the DataFrame
    for index, row in df.iterrows():
        is_header = False
        # Check if this row is a header row by checking if any cell (except the first) matches any of the header names
        for i, cell in enumerate(row[1:], start=1):
            cell_value = ' '.join(str(cell).strip().split())  # Replace all whitespace characters with a single space
            if cell_value in header_names:
                is_header = True
                break
        if is_header:
            # This is a header row
            # Read the 'katedra' value from the first column
            katedra = str(row[0]).strip()
            # Create a mapping from header names to column positions
            column_positions = {}  # Reset column positions for the new header
            for i, cell in enumerate(row):
                cell_value = ' '.join(str(cell).strip().split())  # Replace all whitespace characters with a single space
                if cell_value in header_names:
                    column_positions[cell_value] = i
            continue  # Skip to the next row


        # Skip rows like 'I. ročník'
        first_cell = str(row[cols.predmet]).strip()
        if re.match(r'^[IVX]+\.\s*ročník', first_cell):
            continue  # Skip this row

        # Skip empty or irrelevant rows
        if pd.isnull(row[1]) or pd.isnull(row[0]):
            continue

        # Process regular data rows
        # Extract 'predmet' from the first column (subject description)
        subject_desc = first_cell
        match = re.search(r' \b([A-Z][-A-Z0-9/*]*)\b', subject_desc)
        if match:
            predmet = match.group(1)
        else:
            predmet = None  # Or handle as you see fit

        # Extract the required columns based on the positions in 'column_positions'
        try:
            # Append the processed data to the list
            data_rows.append({
                'katedra': katedra,
                'predmet': predmet,
                'pr_hodin': row[cols.pr_hodin],
                'cv_hodin': row[cols.cv_hodin],
                'n_kruhu': row[cols.n_kruhu],
                'hodino_body': row[cols.hodino_body]
            })
        except (IndexError, TypeError, KeyError):
            # Handle rows that don't have enough columns or missing columns
            continue

    # Create a DataFrame from the processed data
    result_df = pd.DataFrame(data_rows)
    return result_df


