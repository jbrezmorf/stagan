from typing import Dict, List, Tuple
import pathlib

import numpy as np
import yaml

from stagan.stag import read_stag, VyukaPr, StudijniProgram, PredmetIdx, Predmet, PredmetAkce
from tables import facoulty_abr


script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
#MemoizeCfg.instance(cache_dir=script_dir / "funpy_cache")
#import cloudpickle
import joblib
mem = joblib.Memory(location = script_dir / "joblib_cache", verbose=0)
import pandas as pd



def pretty_print_yaml(data, fname=None):
    """
    Pretty prints a hierarchy of lists and dicts using YAML formatting.

    Args:
        data (dict or list): The hierarchical data to print.
    """
    # Dump the data to YAML format with indentation and default_flow_style off (for multiline format)

    if fname is None:
        print(yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False))
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        with open(workdir / fname, 'w') as f:
            if isinstance(data, pd.DataFrame):
                f.write(data.to_string(index=True))
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


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
def students_on_programs(predmety_akce) -> Dict[int, Dict[int, int]]:
    """
    Dict of year -> Dict of program_id -> number of student in program for the year
    """
    # year, program -> students set
    rok_program_students = {}
    for pa in predmety_akce.values():
        for prg_id, students in pa.students.items():
            idx = (pa.rok, prg_id)
            prg_st_set = rok_program_students.setdefault(idx, set())
            rok_program_students[idx] = prg_st_set.union(students)

    # refactor to year -> (program -> n_students)
    rps={}
    for (rok, prg_id), students in rok_program_students.items():
        prg_n_students = rps.setdefault(rok, {})
        prg_n_students[prg_id] = len(students)
    return rps


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

def stag_derived_data(workdir, years, predmety_akce, programy):
    rok_program_students = students_on_programs(predmety_akce)

    sum_program_kredity = {}
    normovanany_studenti = {}
    predmety_podily_fakult = {}
    fakulty_KEN = {}
    for year in years:
        prog_n_students = rok_program_students[year]
        # Compute average KEN of programs of each facoulty, weighted by students on a program
        year_f_KEN = fakulty_KEN.setdefault(year, {})
        for f in facoulty_abr.values():
            KEN_students = [(p.koefEkonomickeNarocnosti, ns)
                            for p, ns in zip(programy.values(), prog_n_students)
                            if p.fakulta == f]
            KEN, N = zip(*KEN_students)
            year_f_KEN[f] = float(np.average(KEN, weights=N))
        print(f"[{year}] fakulty KEN:", year_f_KEN)

        sum_studento_kredity = {}
        for pa in predmety_akce.values():
            #p = p_akce.predmet
            if not pa.rok == year:
                continue
            #if p.fakulta_programu == 'CDV':
            #    continue
            for prg_id in pa.students.keys():
                katedro_program = (pa.katedra, prg_id)
                sum_studento_kredity.setdefault(katedro_program, 0)
                vazeny_st_kredit = pa.vazeny_studento_kredit(prg_id, year_f_KEN)
                sum_studento_kredity[katedro_program] += vazeny_st_kredit
                predmet_tag = pa.label

                podily_predmetu = predmety_podily_fakult.setdefault(predmet_tag, {})
                cilova_fakulta = programy[prg_id].fakulta
                podily_predmetu.setdefault(cilova_fakulta, 0.0)
                podily_predmetu[cilova_fakulta] += float(vazeny_st_kredit)

        # Sum studento_kredity for each program, that is for katedro_program[1]
        spk = sum_program_kredity.setdefault(year, {})
        for (k, p), v in sum_studento_kredity.items():
            spk.setdefault(p, 1.0e-6)
            spk[p] += v

        norm_stud = normovanany_studenti.setdefault(year, {})
        for prg_id, ns in prog_n_students.items():
            norm_stud[prg_id] = ns * year_f_KEN[programy[prg_id].fakulta]

        # print programm codes and n_students
        skp = {(k, programy[i_pr].kod): float(sk) for (k , i_pr), sk in sum_studento_kredity.items()}
        pretty_print_yaml(skp, fname=workdir / f"studento_kredity_{year}.yaml")
        programy_n_students = {programy[prg_id].kod: ns for prg_id, ns in prog_n_students.items()}
        pretty_print_yaml(programy_n_students, fname=workdir / f"programy_n_students_{year}.yaml")
        norm_prog_students = {programy[prg_id].kod: float(ns) for prg_id, ns in norm_stud.items()}
        pretty_print_yaml(norm_prog_students, fname=workdir / f"norm_students_{year}.yaml")

    pretty_print_yaml(predmety_podily_fakult, fname=workdir / "predmety_podily_fakult.yaml")
    return sum_program_kredity, normovanany_studenti, predmety_podily_fakult, fakulty_KEN


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

#@mem.cache
def make_plot_df(year, plot_katedry):
    predmet_akce, programy = read_stag(years, katedry=None)
    # predmet_akce = {i: p for i, p in predmet_akce.items()
    #                 if not (p.fakulta_programu == 'CDV') }


    rozpocet_df = { year: read_rozpocet(year) for year in years}
    for year, df in rozpocet_df.items():
        pretty_print_yaml(df.copy().reset_index().to_dict(orient='records'), fname=f'rozpocet_{year}.yaml')

    sum_program_kredity, normovanany_studenti, predmety_podily_fakult, fakulty_KEN = stag_derived_data(workdir, years, predmet_akce, programy)
    pretty_print_yaml(programy, fname='programy.yaml')
    #pretty_print_yaml(predmet_akce, fname='predmety.yaml')
    rozpocet_errors = []


    df_columns = {}
    df_add = lambda col, item: df_columns.setdefault(col, []).append(item)
    for p_akce in predmet_akce.values():
        if p_akce.katedra not in plot_katedry:
            continue
        naklady_rozpocet = predmet_body_rozpocet(p_akce, rozpocet_df)
        naklady_merge, naklady_estimate, error = vyuka_merge(p_akce, naklady_rozpocet)
        if naklady_merge < 1e-6:
            continue
        if error is not None:
            rozpocet_errors.append(error)
        df_add('label', p_akce.label)
        df_add('katedra', p_akce.katedra)
        df_add('zkratka', p_akce.zkratka)
        df_add('rok', p_akce.rok)
        df_add('rel_naklady', naklady_merge)
        df_add('naklady_estimate', naklady_estimate)
        df_add('n_students', p_akce.n_students)
        stud_kredit = sum((p_akce.vazeny_studento_kredit(prg_id, fakulty_KEN[p_akce.rok])
                           for prg_id in p_akce.students.keys()))

        rel_prijmy = sum((
                stud_kredit/sum_program_kredity[p_akce.rok][prg_id]
                * normovanany_studenti[p_akce.rok][prg_id] for prg_id in p_akce.students.keys()))
        df_add('rel_prijmy', rel_prijmy)

    df_errors = pd.DataFrame(rozpocet_errors)
    df_errors.set_index(["katedra", "zkratka", "rok"], inplace=True)
    df_errors.sort_index(inplace=True)
    pretty_print_yaml(df_errors, fname='rozpocet_errors.yaml')

    # dataframe for programo_predmet
    df = pd.DataFrame(df_columns)
    df.set_index(["katedra", "zkratka", "rok"], inplace=True)
    df.sort_values('label', inplace=True)
    pretty_print_yaml(df, fname=workdir / "vyuka_eff_split.csv")

    # aggregate all programs of predmet to single item
    def check_common_value(series):
        if series.nunique() == 1:
            return series.iloc[0]
        else:

            return '+'.join((str(it) for it in series))  # Indicator for differing values
    df.reset_index()
    df_grouped = df.groupby(level=df.index.names)

    def has_multiple_values(columns):
        def _has_multiple_values(group):
            return (group[columns].nunique(dropna=False) > 1).any()
        return _has_multiple_values
    # Check for common values
    df_grouped.filter(has_multiple_values(['rel_naklady', 'naklady_estimate', 'label']))

    agg_functions = {'n_students': 'sum',
                     'rel_prijmy': 'sum',
                     'rel_naklady': 'max',
                     'naklady_estimate': 'max',
                     'label': 'first',
                     }
    aggregated_df = df_grouped.agg(agg_functions)


    pretty_print_yaml(aggregated_df, fname=workdir / "vyuka_eff.csv")
    aggregated_df.reset_index(inplace=True)
    return aggregated_df, predmety_podily_fakult

# Example usage
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="indexing past lexsort depth may impact performance")

    #fakulta_katedry = {k: 'FM' for k in ['ITE', 'MTI', 'NTI']}
    years = [2021, 2022, 2023, 2024]
    plot_katedry = ['NTI', 'MTI', 'ITE']
    aggregated_df, predmety_podily_fakult = make_plot_df(years, plot_katedry)


    y_range = years_range(aggregated_df)
    from vyuka_plot import plot_vyuka_df
    svg_plot = plot_vyuka_df(aggregated_df, predmety_podily_fakult, pdf_path=workdir / f"vyuka_plot_{y_range}.pdf")

    from report import make_report
    make_report(svg_plot, workdir / f"report_{y_range}.pdf")