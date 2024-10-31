import json
from typing import *
import webbrowser
from functools import cached_property
import urllib.parse as urlparse
import requests
import attrs
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from funpy import memoize, MemoizeCfg
import pathlib
script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
#MemoizeCfg.instance(cache_dir=script_dir / "funpy_cache")
#import cloudpickle
import joblib
mem = joblib.Memory(location = script_dir / "joblib_cache")

"""
TODO:
- read students on RA -> save them to predmet
- sum them to program -> add to programs
- studento_kredity[subject.katedra, subject.program] += subject.N * subject.kredity
- fakulta_KEN[program.fakulta] += program.koefEkonomickeNarocnosti
- fakulta[katedra] ?? getPredmetyByFakulta fakulta -> katedry předmětů ... read Clari24 podily kateder
  or just mak Chat GPT do conversion to a YAML






- make parsistent memoization work  

- reduce df columns to necessary
- agregate 'rozvrhova akce' to 'predmet'
- pro 3 predmety zkusit získat data z tabulky rucne:
  ITE/MTLB
  NTI/PJP
  NTI/UDP
  MTI/DBS  

  index (rok, katedra, predmet, obor, semestr)    
  # Jsou předměty (Pedagogické praktikum), které se vyučují v obou semestrech
  info columns: kredity, 
  concat agregate info columns ( not in df, but in separate dict indexed by the tuple): vsichniUciteleJmenaSPodily, krouzky
  sum agregate columns: hod_prednaska, hod_cviko, (hod_seminar) přednášky, N hod cvika, kredity 
- rozvrhove akce ... add to particular rows, use index (rok, katedra, predmet)
  agreagate: kredits, cvika, prednasky, vyucujici
  
- getPredmetyByObor
"""

@attrs.define
class STAG:
    base_url: str
    ticket: str = None
    user: str = None
    filename: pathlib.Path = pathlib.Path("stag_ticket_user.json")

    def dump_auth(self):
        """Dump 'ticket' and 'user' attributes to a JSON file."""
        auth_data = {
            'ticket': self.ticket,
            'user': self.user
        }
        with open(self.filename, 'w') as f:
            json.dump(auth_data, f)

    def load_auth(self):
        """Load 'ticket' and 'user' attributes from a JSON file."""
        try:
            with open(self.filename, 'r') as f:
                auth_data = json.load(f)
                self.ticket = auth_data.get('ticket')
                self.user = auth_data.get('user')
        except (FileNotFoundError, json.JSONDecodeError):
            self.get_login_ticket()

    def call_api(self, request) -> Any:
        """
        Call the STAG API with the given request object.

        Parameters:
        - request: An instance of a request data class.

        Returns:
        - The parsed JSON response from the API.

        Raises:
        - HTTPError: If the API call was not successful.
        """
        base_url = self.base_url.rstrip('/')
        # Construct the URL
        url = f"{base_url}/{request.request_class}/{request.call_name}"

        self.load_auth()

        # Get the parameters from the request object, excluding special attributes
        params = {
            key: value
            for key, value in asdict(request).items()
            if key not in {'request_class', 'call_name'}
            and value is not None
        }

        # Set headers (you can adjust Accept header based on expected response format)
        headers = {'Accept': 'application/json'}
        #print(f"{request.request_class}/{request.call_name}{params}")
        request_args = dict(url=url, params=params, headers=headers)
        # Make the API request
        try:
            response = requests.get(auth=(self.ticket, ""), **request_args)
            # Raise an exception for HTTP errors
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("Renew access ticket.")
                self.get_login_ticket()
            elif e.response.status_code == 500:
                print(request_args)
                raise e
        response = requests.get(auth=(self.ticket, ""), **request_args)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        return response.json()


    def get_login_ticket(self):
        # Set the redirect URL (just a placeholder for this case)
        original_url = "http://www.stag-client.cz"

        # Construct the login URL
        login_url = f"https://stag-ws.tul.cz/ws/login?originalURL={urlparse.quote(original_url)}"

        # Step 1: Open the login URL in the user's default browser
        print("Opening the STAG login page in your browser. Please log in.")
        webbrowser.open(login_url)

        redirected_url = input("After logging in, please copy and paste the full URL you were redirected to here:\n")

        # Step 3: Parse the stagUserTicket from the URL
        parsed_url = urlparse.urlparse(redirected_url)
        query_params = urlparse.parse_qs(parsed_url.query)
        self.ticket = query_params.get("stagUserTicket", [None])[0]
        self.user = query_params.get("stagUserInfo", [None])[0]
        self.dump_auth()


def stag_call(stag, req):
    return stag.call_api(req)

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
        with open(workdir / fname, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def akreditace_year(date: Dict[str, str]):
    if date is None:
        return None
    dne, mesic, rok = date['value'].split('.')
    return int(rok)


def get_dict(obj):
    if attrs.has(obj):
        return attrs.asdict(obj)
    return obj


def initialize_from_dict(cls, *dicts):
    merged_dict = get_dict(dicts[-1])
    for d in reversed(dicts[:-1]):
        merged_dict.update(get_dict(d))

    # Get the set of attribute names defined in the class
    cls_fields = {field.name for field in attrs.fields(cls)}

    # Filter the dictionary to include only keys that match the class attributes
    filtered_data = {key: value for key, value in merged_dict.items() if key in cls_fields}

    # Initialize the class with filtered dictionary data
    try:
        instance = cls(**filtered_data)
    except TypeError as e:
        print(e)
        print("Input Dict:")
        pretty_print_yaml(list(dicts))
        return None
    return (instance.idx(), instance)

def is_pair(value):
    return isinstance(value, tuple) and len(value) == 2

def extend_valid(lst, add_lst):
    filtered = [item for item in add_lst if is_pair(item) and item[0] is not None]
    lst.extend(filtered)



@attrs.define
class StudijniProgram:
    stprIdno: int
    nazev: str
    kod: str
    fakulta: str
    kredity: int
    koefEkonomickeNarocnosti: float
    typ: str
    forma: str
    platnyOd: int
    akreditaceDoDate: str = attrs.field(converter=akreditace_year)
    n_students: int = 0

    def idx(self):
        return self.stprIdno

@dataclass
class GetStudijniProgramyRequest:
    """
    Request to retrieve study programs.

    Parameters:
    - fakulta (str): Faculty code (optional).
    - typStProgramu (str): Type of study program (optional).
    - plusNove (bool): Include new programs (optional).
    - plusNeplatne (bool): Include invalid programs (optional).
    - vSemestru (str): In semester (optional).
    - vAkademRoc (str): In academic year (optional).
    """
    request_class: str = 'programy'
    call_name: str = 'getStudijniProgramy'
    fakulta: Optional[str] = None
    typStProgramu: Optional[str] = None
    plusNove: Optional[bool] = None
    plusNeplatne: Optional[bool] = None
    vSemestru: Optional[str] = None
    vAkademRoc: Optional[str] = None

@mem.cache
def studijni_programy(stag_client):
    # Example 2: Get study programs
    programy_request = GetStudijniProgramyRequest(
        #fakulta='FAV',  # Optional faculty code
        #typStProgramu='B'  # Bachelor's programs
    )

    programy_response = stag_client.call_api(programy_request)['programInfo']
    #pretty_print_yaml(programy_response)
    programs = []
    programs_add = [ initialize_from_dict(StudijniProgram, prg) for prg in programy_response ]
    extend_valid(programs, programs_add)
    return dict(programs)


@attrs.define
class StudijniObor:
    oborIdno: int
    nazev: str
    cisloOboru: str
    fakulta: str
    typ: str
    forma: str
    limitKreditu: int
    # passed from program.
    stprIdno: int
    kredity: int
    koefEkonomickeNarocnosti: float
    akreditaceDoDate: int
    platnyOd:str  = attrs.field(converter=int)     # year

    def idx(self):
        return self.oborIdno


@dataclass
class GetOboryStudijnihoProgramuRequest:
    """
    Request to retrieve branches of a study program.

    Parameters:
    - fakulta (str): Faculty code (e.g., 'FAV').
    - spKey: Key of the study program (required).
    """
    stprIdno: int
    rok: int = None     # None == actual year
    request_class: str = 'programy'
    call_name: str = 'getOboryStudijnihoProgramu'

@mem.cache
def studijni_obory(stag_client, programy:List[StudijniProgram], roky: List[int]):
    obory = []
    for prg in programy:
        for rok in roky:
            request = GetOboryStudijnihoProgramuRequest(prg.stprIdno, rok)
            obor_list = stag_client.call_api(request)['oborInfo']
            obory_add = [initialize_from_dict(StudijniObor, obor, prg) for obor in obor_list]
            extend_valid(obory, obory_add)
    return dict(obory)


@attrs.define
class VyukaPr:
    n_hodin_pr: int
    n_hodin_cv: int
    n_hodin_sem: int
    n_kruh: int         # myšleno počet paralelních cvik, fakticky může být více kruhů
                        # z různých programů na jedné RA
    _body: float = None

    def body(self):
        koef_pr = 1.2
        koef_cv12 = 1.0
        koef_cv3 = 0.8
        b = (self.n_hodin_pr * koef_pr
             + min(2, self.n_hodin_cv) * koef_cv12
             + max(0, self.n_hodin_cv - 2) * koef_cv3)
        if self._body is None:
            return b
        else:
            assert b == self._body, repr(self)
            return self._body


@attrs.define
class Predmet:
    katedra: str
    zkratka: str
    nazev: str
    kreditu: int
    rok: int
    vyukaLS: bool
    vyukaZS: bool
    rozsah: str
    # from Obor and Program
    oborIdno: int
    cisloOboru: str
    stprIdno: int
    fakulta_programu: str
    koefEkonomickeNarocnosti: float

    # agregated
    vyuka_rozpocet: VyukaPr

    students: Set[str] = attrs.field(factory=set)   # osobni cisla studentu
    cvika: Set[Any] = attrs.field(factory=set)         # rozvrhove akce cvik
    prednasky: Set[Any] = attrs.field(factory=set)         # rozvrhove akce cvik

    @property
    def vyuka_stag(self) -> VyukaPr:
        return VyukaPr(self.n_hodin_prednaska(), self.n_hodin_cvika(), self.n_hodin_sem(),
                       self.n_paralel_cv(), None)

    def vazeny_studento_kredit(self, fakulty_KEN):
        return self.n_students * self.kreditu * fakulty_KEN[self.fakulta_programu]

    def n_hodin(self):
        n_hod_cv: int = 0
        n_hod_pr: int = 0

    def idx(self):
        # Účtování předmětů je po jednotlivých programech
        return (self.rok, self.katedra, self.zkratka, self.stprIdno)

    @cached_property
    def n_students(self):
        return len(self.students)

    @cached_property
    def rozsah_tuple(self):
        return tuple([i for i in self.rozsah.split('+')])

    def _n_hodin(self, set_akci, dotace):
        n_per_tyd = {'K': 2, 'L':1, 'S':1, 'J':0}
        tydny_sum = {akce[1:] for akce in set_akci}
        assert len(tydny_common) == 1
        od, do, tydenZkr = list(tydny_common)[0]
        n_tydnu = do - od + 1

        return [akce[0] for akce in set_akci], tydny_common, dotace

    def n_hodin_cvika(self):
        return self._n_hodin(self.cvika, self.rozsah_tuple[1])

    def n_hodin_prednaska(self):
        return self._n_hodin(self.prednasky, self.rozsah_tuple[0])

PredmetIdx = Tuple[int, str, str, int]

@dataclass
class GetPredmetyByOborRequest:
    """
    Request to retrieve subjects by branch.

    Parameters:
    - fakulta (str): Faculty code (e.g., 'FAV').
    - obor: Branch code or key (required).
    """
    oborIdno: int
    rok: int
    request_class: str = 'predmety'
    call_name: str = 'getPredmetyByObor'

def semestr_dict(predmet_dict):
    sem_decode = dict(AN='ZS', NA='LS', NN=None, AA=None)
    sem_str = predmet_dict.get('vyukaZS', 'N') + predmet_dict.get('vyukaLS', 'N')
    return dict(semestr=sem_decode[sem_str])

@mem.cache
def get_predmety(stag_client, obory:List[StudijniObor], roky: List[int]):
    predmety = []
    for obor in obory:
        for rok in roky:
            od, do = obor.platnyOd, obor.akreditaceDoDate
            if od is None:
                od = -1
            if do is None:
                do = 9999
            if od <= rok <= do:
                request = GetPredmetyByOborRequest(obor.oborIdno, rok)
                predmety_res = stag_client.call_api(request)['predmetOboru']
                predmet_add = [initialize_from_dict(Predmet, predmet, obor, {'fakulta_programu': obor.fakulta}) for predmet in predmety_res]
                pretty_print_yaml(predmet_add, fname=f'predmety_{rok}_{obor.oborIdno}.yaml')
                extend_valid(predmety, predmet_add)
    return dict(predmety)


@attrs.define
class RozvrhovaAkce:
    roakIdno: int
    katedra: str
    predmet: str    # zkratka
    rok: int
    obsazeni: int   # pocet studentu na akci
    typAkceZkr: str     # CV, Pr,
    semestr: str        # ZS, LS
    tydenZkr: str       # K (Každý), S (sudý), L (lichý), J (Jiný, kombinované)
    tydny: Tuple[int, int]      # Od, Do
    krouzky: str

    def idx(self):
        return self.roakIdno

    def krouzky_list(self):
        if self.krouzky is None:
            return []
        return self.krouzky.split(',')


@dataclass
class GetRozvrhByKatedraRequest:
    """
    Request to retrieve the schedule by department.

    Parameters:
    - katedra (str): Department code (e.g., 'KMA').
    - rok (int): Academic year (e.g., 2023).
    - semestr (str): Semester code ('ZS' for winter, 'LS' for summer).
    """
    katedra: str = ''
    rok: int = 0
    semestr: str = ''
    request_class: str = 'rozvrhy'
    call_name: str = 'getRozvrhByKatedra'


def tydny_guess(ra):
    pair = (ra['tydenOd'], ra['tydenDo'])
    return {'tydny': pair}

@mem.cache
def get_rozvrh(stag_client, katedry: List[str], years: List[int]) -> Dict[int, RozvrhovaAkce]:
    rozvrh = []

    for katedra in katedry:
        for rok in years:
            for semestr in ['ZS', 'LS']:
                # Example 1: Get schedule by department
                rozvrh_request = GetRozvrhByKatedraRequest(katedra, rok, semestr)
                ra_res = stag_client.call_api(rozvrh_request)['rozvrhovaAkce']
                add = [initialize_from_dict(RozvrhovaAkce, ra, tydny_guess(ra)) for ra in ra_res]
                extend_valid(rozvrh, add)
    return dict(rozvrh)


@attrs.define
class Student:
    osCislo: str
    stprIdno: int
    oborIdnos: int
    kodSp: str


    def idx(self):
        return self.osCislo



@dataclass
class GetStudentiByRoakceRequest:
    """
    Request to retrieve the students of 'krouzek'.
    """
    roakIdno: int
    request_class: str = 'student'
    call_name: str = 'getStudentiByRoakce'


def check_one(set_in):
    assert len(set_in) == 1, f"Non unique set: {set_in}"
    return list(set_in)[0]

def get_studenti(akce: Dict[int, RozvrhovaAkce], predmety: Dict[PredmetIdx, Predmet], programy: Dict[int, StudijniProgram]):
    for ra in akce.values():
        studenti_req = GetStudentiByRoakceRequest(roakIdno=str(ra.roakIdno))
        studenti = stag_client.call_api(studenti_req)['studentPredmetu']
        studenti:List[Tuple[int, Student]] = [initialize_from_dict(Student, s) for s in studenti]
        for _, s in studenti:
            predmet_idx = (ra.rok, ra.katedra, ra.predmet, int(s.stprIdno))
            try:
                predmety[predmet_idx].students.add(s.osCislo)
            except KeyError:
                print(f"Missing Predmet: {predmet_idx}")
                continue

            if ra.typAkceZkr == 'Cv':
                predmety[predmet_idx].cvika.add((ra.roakIdno, *ra.tydny, ra.tydenZkr))
            elif ra.typAkceZkr == 'Př':
                predmety[predmet_idx].prednasky.add((ra.roakIdno, *ra.tydny, ra.tydenZkr))
            else:
                print(f"Unsupported typAkceZkr: {ra.typAkceZkr}, {ra.roakIdno} ")

    # sum n_students to StudijniProgram
    for p in predmety.values():
        programy[p.stprIdno].n_students += p.n_students


# Example usage
if __name__ == "__main__":
    filename = script_dir.parent / 'dokumenty' / 'RozpFM24_verze 02.5._FINAL_VEDOUCI.XLSX'
    from read_rozpocet_fm import process_excel_file
    rozpocet_df = process_excel_file(filename)
    rozpocet_df.set_index(['katedra', 'predmet'], inplace=True)

    # TODO: celá TUL z Clari24
    fakulta_katedry = {k:'FM' for k in ['ITE', 'MTI', 'NTI']}

    years = [2023]


    # Initialize the STAG client with the base URL
    base_url = "https://stag-ws.tul.cz/ws/services/rest2"
    # base_url_ = str(base_url)
    # print(hashlib.sha256(cloudpickle.dumps(base_url)).hexdigest())
    # print(hashlib.sha256(cloudpickle.dumps(base_url_)).hexdigest())
    # print(hex(hash(base_url)))
    # print(hex(hash(base_url_)))

    stag_client = STAG(base_url)
    #print(hashlib.sha256(cloudpickle.dumps(stag_client)).hexdigest())
    #print(hashlib.sha256(cloudpickle.dumps(attrs.evolve(stag_client))).hexdigest())

    programy  = studijni_programy(stag_client)
    pretty_print_yaml(programy, fname='programy.yaml')
    obory = studijni_obory(stag_client, list(programy.values()), years)
    pretty_print_yaml(obory, fname='obory.yaml')
    predmety = get_predmety(stag_client, list(obory.values()), years)
    pretty_print_yaml(predmety, fname='predmety.yaml')
    print(len(predmety))

    #katedry = {p.katedra for p in predmety.values()}
    katedry = ['NTI', 'MTI', 'ITE']
    rozvrhove_akce = get_rozvrh(stag_client, katedry, years)
    #krouzky_kod = {k for ra in rozvrhove_akce.values() for k in ra.krouzky_list()}
    #print(krouzky_kod)

    # update 'n_students, n_cv, n_pr' in Predmet and Program instances
    get_studenti(rozvrhove_akce, predmety, programy)

    for p in predmety.values():
        if p.katedra in katedry:
            row = rozpocet_df.loc[(p.katedra, p.zkratka)]
            p.vyuka_rozpocet = VyukaPr(row['pr_hodin'],
                                       row['cv_hodin'],
                                       0,
                                       row['n_kruhu'],
                                       _body=row['hodino_body'])

    katedra_faculty_dict = dict([
        ("KMP", 2), ("KSP", 2), ("KMT", 2), ("KEZ", 2), ("KKY", 2), ("KST", 2), ("KOM", 2), ("KVM", 2), ("KSR", 2),
        ("KTS", 2), ("KVS", 2), ("DFS", 2), ("KPE", 3), ("KIN", 3), ("KSY", 3), ("KFÚ", 3), ("KMG", 3), ("KCJ", 3),
        ("KEK", 3), ("KPO", 3), ("DFE", 3), ("KTT", 4), ("KMI", 4), ("KHT", 4), ("KOD", 4), ("KNT", 4), ("KDE", 4),
        ("DFT", 4), ("KFL", 5), ("KPP", 5), ("KMD", 5), ("KSS", 5), ("KCH", 5), ("KFY", 5), ("KRO", 5), ("KAP", 5),
        ("KGE", 5), ("KPV", 5), ("KRO", 5), ("KHI", 5), ("KAJ", 5), ("KTV", 5), ("KNJ", 5), ("KCL", 5), ("DFP", 5),
        ("CPP", 5), ("KVU", 6), ("KPS", 6), ("KNK", 6), ("KUR", 6), ("KAR", 6), ("KDA", 6), ("KED", 6), ("DFA", 6),
        ("ITE", 7), ("MTI", 7), ("NTI", 7), ("DFM", 7), ("UZS", 8)
    ])
    facoulty_abr = {
        2: "FS",   # Fakulta strojní
        3: "FE",   # Ekonomická fakulta
        4: "FT",   # Fakulta textilní
        5: "FP",   # Fakulta přírodovědně-humanitní a pedagogická
        6: "FA",   # Fakulta umění a architektury
        7: "FM",   # Fakulta mechatroniky, informatiky a mezioborových studií
        8: "FZS"   # Fakulta zdravotnických studií
    }

    # Compute average KEN of programs of each facoulty, weighted by students on a program
    fakulty_KEN = {}
    for f in facoulty_abr.values():
        KEN_students = [(p.koefEkonomickeNarocnosti, p.n_students) for p in programy.values() if p.fakulta == f]
        KEN, N = zip(*KEN_students)
        fakulty_KEN[f] = np.average(KEN, weights=N)

    sum_studento_kredity = {}
    for p in predmety.values():
        katedro_program = (p.katedra, p.stprIdno)
        sum_studento_kredity.setdefault(katedro_program, 0)
        sum_studento_kredity[katedro_program] += p.vazeny_studento_kredit(fakulty_KEN)

    # Sum studento_kredity for each program, that is for katedro_program[1]
    sum_program_kredity = {}
    for (k, p), v in sum_studento_kredity.items():
        sum_program_kredity.setdefault(p, 0)
        sum_program_kredity[p] += v

    # print programm codes and n_students
    for p in programy.values():
        print(f"{p.kod}: {p.n_students}")
    normovanany_studenti = sum([p.n_students * p.koefEkonomickeNarocnosti for p in programy.values()])

    # Fill a dataframe for predmety having columns:
    # f"{p.katedra}/{p.zkratka}",
    # p.rok,
    # rel_prijmy = p.vazeny_studento_kredit(fakulty_KEN) / sum_program_kredity[p.stprIdno] * normovany_studenti
    # rel_naklady= p.vyuka_rozpocet.body()

    df_data = []
    for p in predmety.values():
        rel_prijmy = p.vazeny_studento_kredit(fakulty_KEN) / sum_program_kredity[p.stprIdno] * normovanany_studenti
        rel_naklady = p.vyuka_rozpocet.body()
        program = programy[p.stprIdno]
        df_data.append([p.katedra, p.zkratka, p.rok, p.n_students, rel_prijmy, rel_naklady, program.fakulta_programu])

    # dataframe for programo_predmet
    df = pd.DataFrame(df_data, columns=["katedra", "zkratka", "rok", "n_students", "rel_prijmy", "rel_naklady", 'fakulta_programu'])
    df.set_index(["katedra", "zkratka", "rok"], inplace=True)
    # aggregate all programs of predmet to single item
    def check_common_value(series):
        if series.nunique() == 1:
            return series.iloc[0]
        else:
            return '+'.join((str(it) for it in series))  # Indicator for differing values

    agg_functions = {'n_students': 'sum',
                     'rel_prijmy': 'sum',
                     'rel_naklady': check_common_value,
                     'fakulta_programu': check_common_value
                     }
    aggregated_df = df.groupby(df.index).agg(agg_functions)