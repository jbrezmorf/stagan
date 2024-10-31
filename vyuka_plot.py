import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

facoulties = ['FS', 'FT', 'FP', 'FM', 'FZS', 'FA', 'FE']

def plot_predmety(df):
    """
    plot data:
    subject tag, re_prijmy, rel_naklady, rok, fakulta_program
    :param predmety:
    :return:
    """
    # Re-importing necessary libraries due to state reset


    # Calculate fraction and sort by last year's fraction values
    df['fraction'] = df['rel_prijmy'] / df['rel_naklady']
    last_year_fraction_v2 = df[df['rok'] == 2022].set_index('predmet')['fraction']
    sorted_predmet_by_last_year_v2 = last_year_fraction_v2.sort_values(ascending=False).index
    df['predmet'] = pd.Categorical(df['predmet'], categories=sorted_predmet_by_last_year_v2,
                                   ordered=True)

    # Plotting as horizontal bars grouped by 'predmet', with separate bars per year within each 'predmet' group
    plt.figure(figsize=(12, 8))
    colors_v3 = {program: plt.cm.tab10(i % 10) for i, program in enumerate(df['fakulta_program'].unique())}

    # Plot each 'predmet' group with separate horizontal bars for each year
    for i, predmet in enumerate(sorted_predmet_by_last_year_v2):
        subset_v3 = df[df['predmet'] == predmet]

        # Offset each year's bar within the same 'predmet' group for clarity
        for j, (_, row) in enumerate(subset_v3.iterrows()):
            plt.barh(i - j * 0.15, row['fraction'], height=0.15, color=colors_v3[row['fakulta_program']],
                     edgecolor='black', alpha=0.8)

    # Setting labels and ticks for better readability
    plt.xlabel("Fraction of rel_prijmy / rel_naklady")
    plt.ylabel("Predmet (sorted by 2022 fraction)")
    plt.title("Fraction of rel_prijmy / rel_naklady by 'predmet' over Years (Grouped, Colored by fakulta_program)")
    plt.yticks(range(len(sorted_predmet_by_last_year_v2)), sorted_predmet_by_last_year_v2)

    # Adding legend
    handles = [plt.Line2D([0], [0], color=colors_v3[prog], lw=4) for prog in colors_v3.keys()]
    plt.legend(handles, colors_v3.keys(), title='Fakulta Program', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def mock_df():
    # Re-generating the mock data with 8 'predmet' values over 5 years and randomly assigned 'fakulta_program'
    np.random.seed(0)
    data_mock_v2 = {
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


# Example usage
if __name__ == "__main__":
    df_mock_v2 = mock_df()
    plot_predmety(df_mock_v2)