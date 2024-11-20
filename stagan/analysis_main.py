from typing import Dict, List, Tuple, Set

import numpy as np
from tools import pretty_print_yaml
from stagan.stag import read_stag, VyukaPr, StudijniProgram, PredmetIdx, Predmet, PredmetAkce
from tables import facoulty_abr, katedra_faculty_dict, KENs
import attrs

import pathlib
script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
#MemoizeCfg.instance(cache_dir=script_dir / "funpy_cache")
#import cloudpickle
import joblib
mem = joblib.Memory(location = script_dir / "joblib_cache", verbose=0)
import pandas as pd





#@mem.cache
def read_rozpocet(year):
    from read_rozpocet_fm import process_excel_file, RozpocetCols
    cols = {
        2019: RozpocetCols(0, 1, 2, 3, 7),
        2020: RozpocetCols(0, 1, 2, 3, 7),
        2021: RozpocetCols(0, 1, 2, 3, 7),
        2022: RozpocetCols(0, 1, 2, 3, 7),
        2023: RozpocetCols(0, 1, 2, 3, 7),
        2024: RozpocetCols(0, 2, 3, 4, 8)
    }
    filename = script_dir.parent / 'dokumenty' / f'RozpFM_{year}.xlsx'
    rozpocet_df = process_excel_file(filename, cols[year])
    rozpocet_df.set_index(['katedra', 'predmet'], inplace=True)
    rozpocet_df.sort_index(level=['katedra', 'predmet'], inplace=True)
    return rozpocet_df


#@mem.cache
def students_on_programs(predmety_akce, programy) -> Dict[Tuple[int, str], Set[str]]:
    """
    Dict of year -> Dict of program_id -> number of student in program for the year
    """
    # mismatch prgs: B0111A190021
    # year, program -> students set
    rok_program_students = {}
    for pa in predmety_akce.values():
        for prg_id, students in pa.students.items():
            idx = (pa.rok, programy[prg_id].kod)
            prg_st_set = rok_program_students.setdefault(idx, set())
            rok_program_students[idx] = prg_st_set.union(students)
    return rok_program_students

    # refactor to year -> (program -> n_students)
    # rps={}
    # for (rok, prg_kod), students in rok_program_students.items():
    #     prg_n_students = rps.setdefault(rok, {})
    #     prg_n_students[prg_kod] = len(students)
    # return rps


def years_range(df):
    """
    Get string denoting the range of years in the dataframe.
    """
    min_year = min(df['rok'])
    max_year = max(df['rok'])
    if min_year == max_year:
        years = str(min_year)
    else:
        years = str(min_year)[2:] + "-" + str(max_year)[2:]
    return years

# def stag_derived_data(workdir, years, predmety_akce, programy):
#     rok_program_students = {}
#     sum_program_kredity = {}
#     normovanany_studenti = {}
#     predmety_podily_fakult = {}
#     fakulty_KEN = {}
#
#     for year in years:
#         n_prog_students_year = rok_program_students.setdefault(year, {})
#         sum_kredity_year = sum_program_kredity.setdefault(year, {})
#         norm_stud_year = normovanany_studenti.setdefault(year, {})
#         prog_n_students = rok_program_students[year]
#         # Compute average KEN of programs of each facoulty, weighted by students on a program
#         year_f_KEN = fakulty_KEN.setdefault(year, {})
#         for f in facoulty_abr.values():
#             KEN_students = [(p.koefEkonomickeNarocnosti, ns)
#                             for p, ns in zip(programy.values(), prog_n_students)
#                             if p.fakulta == f]
#             KEN, N = zip(*KEN_students)
#             year_f_KEN[f] = float(np.average(KEN, weights=N))
#         print(f"[{year}] fakulty KEN:", year_f_KEN)
#
#
#         for pa in predmety_akce.values():
#             #p = p_akce.predmet
#             if not pa.rok == year:
#                 continue
#             #if p.fakulta_programu == 'CDV':
#             #    continue
#             for prg_id, students_set in pa.students.items():
#                 prg_kod = programy[prg_id].kod
#                 prg_set = n_prog_students_year.setdefault(prg_kod, set())
#                 n_prog_students_year[prg_kod] = prg_set.union(students_set)
#
#                 katedro_program = (pa.katedra, prg_kod)
#                 sum_kredity_year.setdefault(katedro_program, 1e-6)
#                 vazeny_st_kredit = pa.vazeny_studento_kredit(prg_id, year_f_KEN)
#                 sum_kredity_year[katedro_program] += float(vazeny_st_kredit)    # just to be sure
#                 predmet_tag = pa.label
#
#                 podily_predmetu = predmety_podily_fakult.setdefault(predmet_tag, {})
#                 cilova_fakulta = programy[prg_id].fakulta
#                 podily_predmetu.setdefault(cilova_fakulta, 1e-6)
#                 podily_predmetu[cilova_fakulta] += float(vazeny_st_kredit)
#
#         # students per program, normalized students per program
#         for prg_kod in n_prog_students_year.keys():
#             n_stud = n_prog_students_year[prg_kod] = len(n_prog_students_year[prg_kod])
#             norm_stud_year[prg_kod] = n_stud * KENs[prg_kod]
#
#         # print programm codes and n_students
#         #skp = {(k, programy[i_pr].kod): float(sk) for (k , i_pr), sk in sum_studento_kredity.items()}
#         pretty_print_yaml(sum_kredity_year, fname=workdir / f"studento_kredity_{year}.yaml")
#         #programy_n_students = {programy[prg_id].kod: ns for prg_id, ns in prog_n_students.items()}
#         pretty_print_yaml(prog_n_students, fname=workdir / f"programy_n_students_{year}.yaml")
#         #norm_prog_students = {programy[prg_id].kod: float(ns) for prg_id, ns in norm_stud.items()}
#         pretty_print_yaml(norm_stud_year, fname=workdir / f"norm_students_{year}.yaml")
#
#     pretty_print_yaml(predmety_podily_fakult, fname=workdir / "predmety_podily_fakult.yaml")
#     return sum_program_kredity, normovanany_studenti, predmety_podily_fakult, fakulty_KEN


def predmet_body_rozpocet(predmet: PredmetAkce, rozpocet_df:Dict[int, pd.DataFrame]):
    try:
        row = rozpocet_df[predmet.rok].loc[(predmet.katedra, predmet.zkratka)]
        pr_hodin = row['pr_hodin'].astype(int).sum()
        cv_hodin = row['cv_hodin'].astype(int).sum()
        n_kruhu = row['n_kruhu'].astype(int).sum()
        hodino_body = row['hodino_body'].astype(float).sum()
        return VyukaPr(pr_hodin, cv_hodin, 0, n_kruhu, hodino_body)
    except KeyError as e:
        return VyukaPr(0, 0, 0, 0, 0)

def vyuka_merge(pa: PredmetAkce, vyuka_rozpocet: VyukaPr):
    """
    Report missing or inconsistent rozpocet_df entry.
    """
    vyuka_estimate = pa.vyuka_stag
    error = None
    if (not np.allclose(vyuka_estimate.array, vyuka_rozpocet.array)):
        kw_estimate = vyuka_estimate.to_prefixed_dict("est_")
        kw_rozpocet = vyuka_rozpocet.to_prefixed_dict("rozp_")
        error = dict(rok=pa.rok, katedra=pa.katedra, zkratka=pa.zkratka,
                     **kw_estimate, **kw_rozpocet)
    body = vyuka_rozpocet.body()
    estimate = False
    if body == 0.0:
        body = vyuka_estimate.body()
        estimate = True
    return body, estimate, error

def common(x_col):
    x_vals = x_col.unique()
    assert len(x_vals) == 1, x_vals
    return x_vals[0]

def pairs_to_dict(pairs, agg_fn='mean'):
    # Convert the list of pairs to a DataFrame
    df = pd.DataFrame(pairs, columns=['first', 'second'])
    # Group by 'first' and calculate the mean of 'second'
    average_dict = df.groupby('first')['second'].agg(func=agg_fn).to_dict()
    return average_dict

def facoulty_KENs(year_prog_students, programy):
    fakulty_KEN = {}

    new_fac_names = {'UZS': 'FZS', 'HF': 'FE'}
    fzs = lambda x: new_fac_names[x] if x in new_fac_names else x
    kod_to_KEN = pairs_to_dict([(p.kod, p.koefEkonomickeNarocnosti) for p in programy.values()], agg_fn='mean')
    kod_to_fakulta = pairs_to_dict([(p.kod, (p.platnyOd, fzs(p.fakulta))) for p in programy.values()], agg_fn='max')
    kod_to_fakulta = {k: f for k, (d, f) in kod_to_fakulta.items()}
    df = pd.DataFrame()
    df['year'], df['prog_kod'], df['n_students'] = zip(*[(year, prog_kod, len(students)) for (year, prog_kod), students in year_prog_students.items()])
    df['KEN'] = df['prog_kod'].map(kod_to_KEN)
    df['fakulta'] = df['prog_kod'].map(lambda kod: kod_to_fakulta[kod])
    grouped = df.groupby(['year', 'fakulta']).apply(
        lambda x: np.average(x['KEN'], weights=x['n_students'])
    )
    df_year_facoulty_to_ken = grouped.reset_index(name='fakulta_KEN')
    year_facoulty_to_ken = grouped.to_dict()
    prg_n_students = df.groupby(['year', 'prog_kod'])['n_students'].sum().to_dict()

    return year_facoulty_to_ken, kod_to_KEN, kod_to_fakulta, prg_n_students

PredmetPrgId = Tuple[int, str, str, int]
def union_students(data: List[Tuple[PredmetPrgId, Set[str]]]):
    """
    Create student sets on (rok, katedra, predmet, program_id) indices
    :param data:
    :return:
    """
    result_dict = {}
    for idx, students in data:
        # Use (rok, katedra, zkratka) as the key and add the students set to the existing set
        result_dict.setdefault(idx, set())
        result_dict[idx].update(students)

    return result_dict

def group_sets(pair_list):
    set_dict = {}
    for idx, val in pair_list:
        idx_set = set_dict.setdefault(idx, set())
        idx_set.update([val])
    return set_dict

def label_for_pa_set(pa_set):
    katedra_sets = group_sets([(katedra, zkratka) for n, katedra, zkratka in sorted(pa_set)])
    label_lines = [katedra + '/' + '+'.join(zkratka_set) for katedra, zkratka_set in katedra_sets.items()]
    return ' | '.join(label_lines)


#@mem.cache
def make_plot_df(years, plot_katedry):
    predmet_akce, rozvrhove_akce, programy = read_stag(years, katedry=None)
    # predmet_akce = {i: p for i, p in predmet_akce.items()
    #                 if not (p.fakulta_programu == 'CDV') }

    # Prepare for grouping of RA in same space-time
    space_time_pa_list = [(ra.space_time, (ra.obsazeni, ra.katedra, ra.predmet))   for ra in rozvrhove_akce.values()]
    space_time = group_sets(space_time_pa_list)
    label_dict_rev = {label_for_pa_set(pa_set): pa_set for pa_set in space_time.values()}
    label_dict = { (katedra, zkratka): label
        for label, pa_set in label_dict_rev.items()
        for n_students, katedra, zkratka in pa_set}

    year_prog_students = students_on_programs(predmet_akce, programy)
    year_facoulty_KEN, kod_to_KEN, kod_to_fakulta, prg_n_students = facoulty_KENs(year_prog_students, programy)

    # (rok, katedra, zkratka) -> number of unique students
    n_students_dict = union_students([
            ((pa.rok, pa.katedra, pa.zkratka,  prg_id), students)
            for pa in predmet_akce.values()
                for prg_id, students in pa.students.items()
        ])
    n_students_dict = {k: len(v) for k, v in n_students_dict.items()}

    prg_kod_for_id = {prg_id: prg.kod for prg_id, prg in programy.items()}
    df = pd.DataFrame()
    index = lambda pa, kod: (pa.rok, pa.katedra, pa.zkratka, kod)
    row = lambda pa, prg_id, prg_kod: (*index(pa, prg_kod), n_students_dict[index(pa, prg_id)],
          pa.kreditu, kod_to_KEN[prg_kod])
    df['rok'], df['katedra'], df['zkratka'], df['prg_kod'], df['n_students'], df['kredits'], df['prg_KEN'] = zip(*[
        row(pa, prg_id, prg_kod_for_id[prg_id])
            for pa in predmet_akce.values()
                for prg_id in pa.students.keys()
    ])
    df = df[df['katedra'] != 'CDV']
    df['prg_n_students'] = df.apply(lambda row: prg_n_students[(row['rok'], row['prg_kod'])], axis=1)
    df['fac_of_katedra'] = df['katedra'].map(lambda k: facoulty_abr[katedra_faculty_dict[k]])
    df['fac_of_prg'] = df['prg_kod'].map(lambda kod: kod_to_fakulta[kod])
    df['katedra_KEN'] = df.apply(lambda row: year_facoulty_KEN[(row['rok'], row['fac_of_katedra'])], axis=1)
    df['studento_kredit'] = df['n_students'] * df['kredits'] * df['katedra_KEN']
    df['prg_norm_students'] = df['prg_KEN'] * df['prg_n_students']
    df = df[df['studento_kredit'] > 0]
    df['label'] = df.apply(lambda row: label_dict.get((row['katedra'], row['zkratka']), None), axis=1)

    sk_kzf = df.groupby(['label', 'fac_of_prg'])['studento_kredit'].sum().rename('sk_kzf').reset_index()
    sk_kz = df.groupby(['label'])['studento_kredit'].sum().rename('sk_kz').reset_index()
    merged = sk_kzf.merge(sk_kz, on=['label'])
    merged['weight'] = merged['sk_kzf'] / np.maximum(merged['sk_kz'], 1e-5)
    podily_fakult = merged.groupby(['label']).apply(
        lambda x: {fac: wt for fac, wt in zip(x['fac_of_prg'], x['weight']) if wt != 0}
    ).to_dict()
    pretty_print_yaml(podily_fakult, fname=workdir / "podily_fakult.yaml")

    sk_kzp = df.groupby(['rok', 'katedra', 'zkratka', 'prg_kod'])['studento_kredit'].sum().rename('sk_kzp').reset_index()
    sk_p = df.groupby(['rok', 'prg_kod'])['studento_kredit'].sum().rename('sk_p').reset_index()
    merged = sk_kzp.merge(sk_p, on=['rok', 'prg_kod'])
    merged['weight'] = merged['sk_kzp'] / merged['sk_p']
    df = df.merge(merged, on=['rok', 'katedra', 'zkratka', 'prg_kod'])
    df['rel_prijmy'] = df['weight'] * df['prg_norm_students']
    df_prijem = df
    pretty_print_yaml(df_prijem, fname=workdir / "rel_prijmy_full.csv")

    rozpocet_df = { year: read_rozpocet(year) for year in years}
    for year, df in rozpocet_df.items():
        pretty_print_yaml(df.copy().reset_index().to_dict(orient='records'), fname=f'rozpocet_{year}.yaml')

    rozpocet_errors = []

    def make_item(pa: PredmetAkce):
        naklady_rozpocet = predmet_body_rozpocet(pa, rozpocet_df)
        naklady_merge, naklady_estimate, error = vyuka_merge(pa, naklady_rozpocet)
        if error is not None:
            rozpocet_errors.append(error)
        return (pa.rok, pa.katedra, pa.zkratka, naklady_merge, naklady_estimate)

    df_naklady = pd.DataFrame()
    df_naklady['rok'], df_naklady['katedra'], df_naklady['zkratka'], df_naklady['rel_naklady'], df_naklady['naklady_estimate'] = zip(*[
        make_item(p_akce) for p_akce in predmet_akce.values()
    ])

    df_errors = pd.DataFrame(rozpocet_errors)
    df_errors.set_index(["katedra", "zkratka", "rok"], inplace=True)
    df_errors.sort_index(inplace=True)
    pretty_print_yaml(df_errors, fname='rozpocet_errors.yaml')

    df = df_prijem
    df.reset_index(inplace=True)
    df = df.merge(df_naklady, on=['rok', 'katedra', 'zkratka'])
    df = df[df['rel_naklady'] > 0]

    df.set_index(["katedra", "zkratka", "rok"], inplace=True)
    df.sort_values('label', inplace=True)
    pretty_print_yaml(df, fname=workdir / "vyuka_wide_all.csv")

    df.reset_index(inplace=True)
    df = df[df['katedra'].isin(plot_katedry)]
    df.set_index(["katedra", "zkratka", "rok"], inplace=True)
    pretty_print_yaml(df, fname=workdir / "vyuka_wide_plot.csv")

    # aggregate all programs of predmet to single item
    # def check_common_value(series):
    #     if series.nunique() == 1:
    #         return series.iloc[0]
    #     else:
    #         return '+'.join((str(it) for it in series))  # Indicator for differing values

    # def has_multiple_values(columns):
    #     def _has_multiple_values(group):
    #         return (group[columns].nunique(dropna=False) > 1).any()
    #     return _has_multiple_values
    # # Check for common values
    # df_grouped.filter(has_multiple_values(['rel_naklady', 'naklady_estimate', 'label']))

    agg_functions = {'n_students': 'sum',
                     'rel_prijmy': 'sum',
                     'rel_naklady': 'max',
                     'naklady_estimate': 'max',
                     'katedra': 'first',
                     'zkratka': 'first'
                     }
    df.reset_index(inplace=True)
    df.set_index(["label", "rok"], inplace=True)
    df_grouped = df.groupby(level=df.index.names)
    aggregated_df = df_grouped.agg(agg_functions)
    aggregated_df.reset_index(inplace=True)
    pretty_print_yaml(aggregated_df, fname=workdir / "vyuka_eff_plot.csv")

    return aggregated_df, podily_fakult

# Example usage
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance")

    #fakulta_katedry = {k: 'FM' for k in ['ITE', 'MTI', 'NTI']}
    years = [2021, 2022, 2023, 2024]
    #years = [2023]
    plot_katedry = ['NTI', 'MTI', 'ITE']
    aggregated_df, predmety_podily_fakult = make_plot_df(years, plot_katedry)


    y_range = years_range(aggregated_df)
    from vyuka_plot import plot_vyuka_df
    svg_plot = plot_vyuka_df(aggregated_df, predmety_podily_fakult, pdf_path=workdir / f"vyuka_plot_{y_range}.pdf")

    from report import make_report
    make_report(svg_plot, workdir / f"report_{y_range}.pdf")