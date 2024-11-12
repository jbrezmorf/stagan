from itertools import product

# Re-importing necessary libraries due to state reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.transforms import blended_transform_factory
from io import BytesIO


from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import seaborn as sns

faculty_colors = {
    'FS': '#888B95',  # Faculty of Mechanical Engineering
    'FT': '#924C14',  # Faculty of Textile Engineering
    'FP': '#0076D5',  # Faculty of Science, Humanities, and Education
    'FM': '#EA7603',  # Faculty of Mechatronics, Informatics, and Interdisciplinary Studies
    'FZS': '#00B0BE', # Faculty of Health Studies
    'FA': '#006443',  # Faculty of Arts and Architecture
    'FE': '#65A812',   # Faculty of Economics
    'none':  '#FF0000'
}

def df_mock():
    # Re-generating the mock data with 8 'predmet' values over 5 years and randomly assigned 'fakulta_program'
    np.random.seed(0)
    data_mock_v2 = {
        'katedra': np.repeat(['K1', 'K1', 'K1', 'K1', 'K2', 'K2', 'K2', 'K2'], 5),
        'predmet': np.repeat(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 5),
        'rel_naklady': np.random.uniform(50, 150, 40),
        'rel_prijmy': np.random.uniform(100, 200, 40),
        'fakulta_program': np.random.choice(list(faculty_colors.keys()), 40),
        'rok': np.tile([2018, 2019, 2020, 2021, 2022], 8)
    }
    df_mock_v2 = pd.DataFrame(data_mock_v2)

    # Adding small variance in fraction per year
    df_mock_v2['rel_naklady'] += np.random.normal(0, 5, size=len(df_mock_v2))
    df_mock_v2['rel_prijmy'] = df_mock_v2['rel_naklady'] * np.random.uniform(1.1, 1.3, size=len(df_mock_v2))
    df_mock_v2['naklady_estimate'] = np.random.choice([True, False], len(df_mock_v2))
    df_mock_v2['label'] = df_mock_v2['katedra'] + '/' + df_mock_v2['predmet']

    podily_predmetu = {
        'K1/A': {'FS': 0.5, 'FE': 0.5},
        'K1/B': {'FS': 0.5, 'FE': 0.5},
        'K1/C': {'FS': 0.5, 'FE': 0.5},
        'K1/D': {'FS': 0.5, 'FE': 0.5},
        'K2/E': {'FS': 0.5, 'FE': 0.5},
        'K2/F': {'FS': 0.5, 'FE': 0.5},
        'K2/G': {'FS': 0.5, 'FE': 0.5},
        'K2/H': {'FS': 0.5, 'FE': 0.5}}

    print(df_mock_v2.head())
    return df_mock_v2,podily_predmetu


def plot_vyuka_df(df: pd.DataFrame, podily_predmetu, pdf_path):
    fm_per_hod = 1500
    normativ = 44000
    # Calculate fraction and sort by last year's fraction values
    df['fraction'] = df['rel_prijmy'] / df['rel_naklady'] * normativ / 1000
    max_fraction = df[np.isfinite(df['fraction'])]['fraction'].max()


    # Set 'label' as a categorical variable with the specified order
    max_fakulta = lambda d : max(zip(d.values(), d.keys()))[1]
    fakulta_max_programu = { l:max_fakulta(dc) for l, dc in podily_predmetu.items()}
    fakulty = [fakulta_max_programu[l] for l in df['label']]
    df['label_f'] = df['label'] + " |   " + np.array(fakulty)

    # Automatically skip NaNs in group mean.
    sort_by_df =  df.groupby('label_f')['fraction'].mean()

    labels_sorted = sort_by_df.sort_values(ascending=False).index.tolist()
    # Set 'label' as an ordered categorical
    df['label_f'] = pd.Categorical(df['label_f'], categories=labels_sorted, ordered=True)


    # Sort dataframe by 'label' and 'rok'
    df = df.sort_values(['label_f', 'rok'])
    group_labels = df.groupby('label_f')['label'].first().reindex(labels_sorted).reset_index()['label']

    df.loc[df['fraction'] > 1000 * normativ, 'fraction'] = max_fraction + 0.1
    null_over_null = np.logical_and(df['rel_prijmy'] < 1e-6, df['fraction'] > max_fraction)
    df.loc[null_over_null, 'fraction'] = np.nan
    plotting_df = df.dropna()
    # Ensure years are sorted ascending
    hue_order = sorted(df['rok'].unique())


    # Create a DataFrame with all possible combinations of 'label_f' and 'rok'
    #all_combinations = pd.DataFrame(list(product(labels_sorted, hue_order)), columns=['label_f', 'rok'])
    # Merge the complete combinations with your data
    # plotting_df = all_combinations.merge(
    #     df[['label_f', 'rok', 'fraction', 'naklady_estimate']],
    #     on=['label_f', 'rok'],
    #     how='left'
    # )
    # Reset index to ensure proper alignment
    # plotting_df = plotting_df.reset_index(drop=True)
    plotting_df = df

    # Plotting using seaborn with explicit figure and axis
    sns.set(style="whitegrid")

    mpl.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=(8, len(plotting_df) / 8))
    # Plot the barplot
    sns.barplot(
        data=plotting_df,
        y='label_f',
        x='fraction',
        hue='rok',
        order=labels_sorted,
        orient='h',
        width=0.6,
        dodge=True,
        fill=True,
        edgecolor='none',
        hue_order=hue_order,
        ax=ax  # Explicitly specify the axis
    )

    # Mark the null values with red edge
    children = ax.get_children()

    bars = [child for child in children
            if isinstance(child, Rectangle) and (child.get_height() > 0)]
    height = bars[0].get_height()
    bars = [bar for bar in bars if np.isclose(bar.get_height(), height)]
    # bars = ax.patches
    # # Check that the number of bars matches the number of data points
    # assert len(bars) == len(plotting_df), f"Mismatch between bars ({len(bars)}) data points ({len(df)})."
    #
    # # Define a function to map 'naklady_estimate' to an edge color
    # # def edgecolor_mapper(naklady_estimate):
    # #     return 'red' if naklady_estimate else 'none'
    #
    # # Loop through the bars and set edge colors
    # for bar, (_, row) in zip(sorted(bars, key=lambda x:x.get_y()), plotting_df.iterrows()):
    #     if row['naklady_estimate']:
    #         bar.set_edgecolor('red')
    #         bar.set_linewidth(1.0)

    # Get positions and labels
    positions = ax.get_yticks()
    ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    #label_to_position = dict(zip(ticklabels, positions))

    # Create a blended transform for mixed coordinate systems
    transform = blended_transform_factory(ax.transAxes, ax.transData)

    # Define x positions in axes coordinates
    x0 = -0.1  # Left end (adjust as needed)
    x1 = -0.005  # Right end (before the plot area)

    for i, (label, y_i) in enumerate(zip(group_labels, positions)):
        if i == 0:
            y_min = y_i - (positions[i + 1] - y_i) / 2
        else:
            y_min = y_i - (y_i - positions[i - 1]) / 2

        if i == len(positions) - 1:
            y_max = y_i + (y_i - positions[i - 1]) / 2
        else:
            y_max = y_i + (positions[i + 1] - y_i) / 2

        # Make a gap between rectangles
        gap = 0.05
        y_min, y_max = y_min + gap, y_max - gap
        # Get the dictionary for faculty composition of the label
        faculty_counts = podily_predmetu.get(label, {'none':1.0})

        # Calculate total and fractions
        total = sum(faculty_counts.values())
        if total == 0:
            continue
        fractions = [(count / total,  faculty)  for faculty, count in faculty_counts.items()]
        print(label, ticklabels[i], list(sorted(fractions)))

        # Draw fractional rectangles within each label's rectangle
        y0 = y_min  # Start position for the first fraction
        for fraction, faculty in sorted(fractions):
            # Width of the segment based on fraction
            y1 = y0 + fraction * (y_max - y_min)

            # Get color for the faculty
            color = faculty_colors.get(faculty, faculty_colors['none'])  # Default to gray if color not found

            # Draw rectangle segment
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                             transform=transform, facecolor=color, edgecolor='none',
                             clip_on=False, zorder=0)
            ax.add_patch(rect)

            # Move start position to end of this segment
            y0 = y1

    ax.set_xlabel('příjem na hodinu výuky (předmětu). [tis. Kč]')
    ax.set_ylabel('katedra / predmet / fakulta programu ')
    ax.set_xlim(0, normativ * 0.5 / 1000)
    ax.set_title('Příjmy na hodinu výuky předmětů na FM')

    # average fraction on FM
    total_frac = lambda  df : df['rel_prijmy'].sum() / df['rel_naklady'].sum() * normativ / 1000
    fm_fraction = total_frac(df)
    fm_target = fm_per_hod / 1000
    vline = lambda x, color, label, **kwargs: (
        ax.vlines(x, ymin=min(positions), ymax=max(positions), colors=color, label=label, **kwargs))
    vline(fm_fraction, faculty_colors['FM'], 'FM avg.', linewidth=1.5)
    vline(fm_target, 'red', 'min: 1500 Kč/h', linewidth=1.5)

    ustavy = ['ITE', 'MTI', 'NTI']
    for ustav, c in zip(ustavy, ['magenta', 'cyan', 'blue']):
        ustav_x = total_frac(df[df['katedra'] == ustav])
        vline(ustav_x, c, ustav+ " avg.", linestyle='dashed', linewidth=1.0)
    # Customize the plot using the axis object
    ax.legend(loc='upper right')  #bbox_to_anchor=(1.05, 1),

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches='tight')

    # Save the plot to a BytesIO buffer in SVG format
    plot_buffer = BytesIO()
    fig.savefig(plot_buffer, bbox_inches='tight',  format='svg')
    plot_buffer.seek(0)
    plt.close(fig)
    return plot_buffer


if __name__ == '__main__':
    df, podily_predmetu = df_mock()
    plot_vyuka_df(df, podily_predmetu, 'vyuka_plot.pdf')
