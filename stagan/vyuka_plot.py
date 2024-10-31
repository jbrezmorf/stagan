
# Re-importing necessary libraries due to state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

faculty_colors = {
    'FS': '#888B95',  # Faculty of Mechanical Engineering
    'FT': '#924C14',  # Faculty of Textile Engineering
    'FP': '#0076D5',  # Faculty of Science, Humanities, and Education
    'FM': '#EA7603',  # Faculty of Mechatronics, Informatics, and Interdisciplinary Studies
    'FZS': '#00B0BE', # Faculty of Health Studies
    'FA': '#006443',  # Faculty of Arts and Architecture
    'FE': '#65A812'   # Faculty of Economics
}

def df_mock():
    # Re-generating the mock data with 8 'predmet' values over 5 years and randomly assigned 'fakulta_program'
    np.random.seed(0)
    data_mock_v2 = {
        'katedra': np.repeat(['K1', 'K1', 'K1', 'K1', 'K2', 'K2', 'K2', 'K2'], 5),
        'predmet': np.repeat(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 5),
        'rel_naklady': np.random.uniform(50, 150, 40),
        'rel_prijmy': np.random.uniform(100, 200, 40),
        'fakulta_program': np.random.choice(['Program1', 'Program2', 'Program3', 'Program4',
                                             'Program5', 'Program6', 'Program7', 'Program8'], 40),
        'rok': np.tile([2018, 2019, 2020, 2021, 2022], 8)
    }
    df_mock_v2 = pd.DataFrame(data_mock_v2)

    # Adding small variance in fraction per year
    df_mock_v2['rel_naklady'] += np.random.normal(0, 5, size=len(df_mock_v2))
    df_mock_v2['rel_prijmy'] = df_mock_v2['rel_naklady'] * np.random.uniform(1.1, 1.3, size=len(df_mock_v2))

    return df_mock_v2


def plot_vyuka_df(df: pd.DataFrame):
    # Calculate fraction and sort by last year's fraction values
    df['fraction'] = df['rel_prijmy'] / df['rel_naklady']
    df['label'] = df['katedra'] + '/' + df['predmet']
    last_year = max(df['rok'])
    last_year_fraction = df[df['rok'] == last_year].set_index('label')['fraction']
    sorted_predmet_by_last_year = last_year_fraction.sort_values(ascending=False).index
    df['label'] = pd.Categorical(df['label'], categories=sorted_predmet_by_last_year, ordered=True)

    # Plotting as horizontal bars grouped by 'predmet', with separate bars per year within each 'predmet' group
    plt.figure(figsize=(8, 12))
    # Plot each 'predmet' group with separate horizontal bars for each year
    for i, predmet in enumerate(sorted_predmet_by_last_year):
        subset_v3 = df[df['predmet'] == predmet]

        # Offset each year's bar within the same 'predmet' group for clarity
        for j, (_, row) in enumerate(subset_v3.iterrows()):
            plt.barh(i - j * 0.15, row['fraction'], height=0.15, color=faculty_colors[row['fakulta_program']],
                     edgecolor='black', alpha=0.8)

    # Setting labels and ticks for better readability
    plt.xlabel("Fraction of rel_prijmy / rel_naklady")
    plt.ylabel("Predmet (sorted by 2022 fraction)")
    plt.title("Fraction of rel_prijmy / rel_naklady by 'predmet' over Years (Grouped, Colored by fakulta_program)")
    plt.yticks(range(len(sorted_predmet_by_last_year)), sorted_predmet_by_last_year)

    # Adding legend
    handles = [plt.Line2D([0], [0], color=colors_v3[prog], lw=4) for prog in colors_v3.keys()]
    plt.legend(handles, colors_v3.keys(), title='Fakulta Program', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
