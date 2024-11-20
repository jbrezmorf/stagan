import json
from typing import *
import webbrowser
from functools import cached_property
import urllib.parse as urlparse
import requests
import attrs
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from tools import pretty_print_yaml
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from tables import KENs, katedra_faculty_dict, facoulty_abr

#from funpy import memoize, MemoizeCfg
import pathlib


script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
#MemoizeCfg.instance(cache_dir=script_dir / "funpy_cache")
#import cloudpickle
import joblib
mem = joblib.Memory(location = script_dir / "joblib_cache", verbose=0)

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

@mem.cache
def stag_call(stag, req):
    return stag.call_api(req)


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
        print("Input Dict:")
        pretty_print_yaml(list(dicts))
        raise e
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

    def idx(self):
        return self.stprIdno

@dataclass
class GetStudijniProgramyRequest:
    """nazev
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

    programy_response = stag_call(stag_client, programy_request)['programInfo']
    #pretty_print_yaml(programy_response)
    programs = []
    programs_add = [ initialize_from_dict(StudijniProgram, prg) for prg in programy_response ]
    filtered = []
    for i, p in programs_add:
        if p.kod in KENs:
            p.koefEkonomickeNarocnosti = KENs[p.kod]
            filtered.append((i, p))
        else:
            print(f"Missing KEN for {p.kod}, {p.fakulta}, {p.nazev}, {p.typ}, {p.forma}, ({p.platnyOd} - {p.akreditaceDoDate})")
    filtered = programs_add
    extend_valid(programs, filtered)
    return dict(programs)


@attrs.define
class StudijniObor:
    oborIdno: int
    nazev: str
    cisloOboru: str
    typ: str
    forma: str
    limitKreditu: int
    # passed from program.
    stprIdno: int
    fakulta: str
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
            obor_list = stag_call(stag_client, request)['oborInfo']
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

    def _body_calc(self):
        koef_pr = 1.2
        koef_cv12 = 1.0
        koef_cv3 = 0.8
        b = (self.n_hodin_pr * koef_pr
             + self.n_hodin_cv * min(2, self.n_kruh) * koef_cv12
             + self.n_hodin_cv * max(0, self.n_kruh - 2) * koef_cv3)
        return b

    def body(self):

        if self._body is None:
            b = self._body_calc()
            return b
        else:
            #assert b == self._body, repr(self)
            return self._body

    @property
    def array(self):
        return np.array([self.n_hodin_pr, self.n_hodin_cv, self.n_hodin_sem, self.n_kruh, self.body()])

    def to_prefixed_dict(self, prefix: str) -> dict:
        """
        Returns a dictionary of the class attributes with keys prefixed by the given prefix string.

        :param prefix: The prefix string to add to each attribute name.
        :return: A dictionary with prefixed attribute names and their corresponding values.
        """
        d = {
            f"{prefix}{field.name}": getattr(self, field.name)
            for field in attrs.fields(self.__class__)
        }
        d[f"{prefix}_body"] = self.body()
        return d

i_ra_prednaska = 0
i_ra_cviceni = 1
i_ra_seminar = 2

@attrs.define
class Predmet:
    katedra: str
    zkratka: str
    nazev: str
    kreditu: int = attrs.field(converter=int)
    rok: int = attrs.field(converter=int)
    vyukaLS: bool
    vyukaZS: bool
    rozsah: str
    # from Obor and Program
    oborIdno: int
    cisloOboru: str
    stprIdno: int
    fakulta_programu: str
    koefEkonomickeNarocnosti: float = attrs.field(converter=float)

    # agregated
    #vyuka_rozpocet: VyukaPr = None


    def idx(self):
        # Účtování předmětů je po jednotlivých programech
        return (self.rok, self.katedra, self.zkratka, self.stprIdno)


    @property
    def label(self):
        return f"{self.katedra}/{self.zkratka}"

    @cached_property
    def rozsah_tuple(self):
        return tuple([i for i in self.rozsah.split('+')])




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
    for obor in tqdm(obory, desc='predmety'):
        for rok in roky:
            od, do = obor.platnyOd, obor.akreditaceDoDate
            if od is None:
                od = -1
            if do is None:
                do = 9999
            if od <= rok <= do:
                request = GetPredmetyByOborRequest(obor.oborIdno, rok)
                predmety_res = stag_call(stag_client, request)['predmetOboru']
                predmet_add = [initialize_from_dict(Predmet, predmet, obor, {'fakulta_programu': obor.fakulta}) for predmet in predmety_res]
                #pretty_print_yaml(predmet_add, fname=f'predmety_{rok}_{obor.oborIdno}.yaml')
                extend_valid(predmety, predmet_add)
    return dict(predmety)


@attrs.define
class RozvrhovaAkce:
    roakIdno: int
    katedra: str
    predmet: str    # zkratka
    rok: int = attrs.field(converter=int)
    semestr: str        # ZS, LS
    platnost: str       # A, B (A = platná, B = blokovaná)
    budova: str
    mistnost: str
    denZkr: str
    hodiny: Tuple[int, int]                         # hodinaOd, hodinaDo
    obsazeni: int  = attrs.field(converter=int)     # pocet studentu na akci
    typAkceZkr: str     # CV, Pr,
    tydenZkr: str       # K (Každý), S (sudý), L (lichý), J (Jiný, chaos!)
    tydny: Tuple[int, int]      # Od, Do
    datum: Tuple[str, str]      # Od, Do
    krouzky: str

    def __hash__(self):
        return self.roakIdno


    def idx(self):
        return self.roakIdno

    def krouzky_list(self):
        if self.krouzky is None:
            return []
        return self.krouzky.split(',')

    def is_fake(self):
        return (self.obsazeni == 0) or (self.mistnost is None) or (self.budova is None)

    @property
    def space_time(self):
        return (self.rok, self.semestr, self.budova, self.mistnost, self.denZkr, self.hodiny, self.tydenZkr, self.tydny)

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


def int_pair(a, b):
    if a == None or b == None:
        return  None
    else:
        return (int(a), int(b))

def tydny_guess(ra):
    pair = (ra['tydenOd'], ra['tydenDo'])
    return {'tydny': int_pair(*pair)}

def hodiny_guess(ra):
    pair = (ra['hodinaOd'], ra['hodinaDo'])
    return {'hodiny': int_pair(*pair)}

def datum_guess(ra):
    pair = (ra['datumOd'], ra['datumDo'])
    return {'datum': pair}

@attrs.define
class PredmetAkce:
    """
    Combine Predmet and RozvrhovaAkce and student ids.
    TODO:
    - remove _predmety from dict representation, used in yaml,
      not sure what is called
    """

    rok: int
    katedra: str
    zkratka: str
    kreditu: float
    rozsah_tuple: Tuple[str, str, str]
    #_predmety: Dict[PredmetIdx, Predmet] = attrs.field(repr=False, metadata={'serialize': False})
    students: Dict[int, Set[str]]   # program_id -> set of student_ids
    akce: List[Dict[str, Any]] = attrs.field(factory=lambda : [{}, {}, {}])  # rozvrhove akce

    n_per_tyd = {'K': 2, 'L': 1, 'S': 1, 'J': 2}
    ra_types = ['Př', 'Cv','Sem']
    ra_types_resolve = {ra_type: i for i, ra_type in enumerate(ra_types)}

    @staticmethod
    def common(set_vals):
        assert len(set_vals) == 1
        return next(iter(set_vals))


    @classmethod
    def from_predmet(cls, predmety_all: Dict[PredmetIdx, Predmet], idx, prog_ids):
        r, k, z = idx
        predmety = [predmety_all[(r, k, z, prog_id)] for prog_id in prog_ids]
        rozsah_tuple = cls.common({p.rozsah_tuple for p in predmety})
        kreditu = cls.common({p.kreditu for p in predmety})
        students = {prog_id: set() for prog_id in prog_ids}
        return cls(r, k, z, kreditu, rozsah_tuple, students)

    @property
    def label(self):
        return f"{self.katedra}/{self.zkratka}"

    def vazeny_studento_kredit(self, prg_id, fakulty_KEN):
        n_students = len(self.students[prg_id])
        fakulta_katedry = facoulty_abr[katedra_faculty_dict[self.katedra]]
        return n_students * self.kreditu * fakulty_KEN[fakulta_katedry]

    def union_students(self, prog_id, students):
        self.students.setdefault(prog_id, set())
        self.students[prog_id] = self.students[prog_id].union(students)

    def add_ra(self, ra: 'RozvrhovaAkce'):
        i_ra_type = self.ra_types_resolve[ra.typAkceZkr]
        ra_dict = self.akce[i_ra_type]
        ra_dict.setdefault(ra.krouzky, set())
        ra_dict[ra.krouzky].add(ra)

    def _n_hodin(self, i_ra_type):
        """
        Check consistency, return:
        number of parallel actions of sam type in week,
        number of unique hours of action par week,
        number of weeks
        """
        rozsah = self.rozsah_tuple[i_ra_type]
        try:
            rozsah = int(rozsah)
        except ValueError:
            print(f"{self.label}, akce: {self.ra_types[i_ra_type]}, rozsah: {rozsah}")
            rozsah = 0

        akce = self.akce[i_ra_type]
        n_parallel = len(akce)
        if n_parallel == 0:
            return 0, 0, 0

        count_ra = lambda ra_set: sum(self.n_per_tyd[ra.tydenZkr] for ra in ra_set)
        akce_rozsah = {krouzky: count_ra(ra_set) for krouzky, ra_set in akce.items()}
        if akce_rozsah.values():
            rozsah_min = min(akce_rozsah.values())
            rozsah_max = max(akce_rozsah.values())
            if (rozsah_min != rozsah_max) or (rozsah_min != rozsah):
                print(f"{self.label}, akce: {self.ra_types[i_ra_type]}," +
                 f" inconsistent rozsah {rozsah} != (min: {rozsah_min}, max: {rozsah_max})")
        else:
            assert rozsah == 0, \
                (f"{self.label}({self.rok}), akce: {self.ra_types[i_ra_type]}," +
                 f" inconsistent rozsah {rozsah} != 0")
            rozsah = 0

        def count_tydny(ra):
            count = ra.tydny[1] - ra.tydny[0]
            if count < 0:
                count += 52
            return count + 1
        all_akce = [ra  for kruhy_akce in akce.values() for ra in kruhy_akce]
        n_tydny = [count_tydny(ra) for ra in all_akce]
        if len(n_tydny) > 0:
            min_n_tydny = min(n_tydny)
            max_n_tydny = max(n_tydny)
            if  min_n_tydny != max_n_tydny:
                print(f"{self.label}({self.rok}), akce: {self.ra_types[i_ra_type]}," + \
                        f" inconsitent tydny: (min: {min_n_tydny}, max: {max_n_tydny})")
            n_tydnu = max_n_tydny
        else:
            n_tydnu = 0
        return n_parallel, rozsah, n_tydnu


    @property
    def n_students(self):
        return sum((len(s) for s in self.students.values()))

    @property
    def vyuka_stag(self) -> VyukaPr:
        par_pred, rozsah_pred, weeks_pred = self._n_hodin(0)
        par_cv, rozsah_cv, weeks_cv = self._n_hodin(1)
        par_sem, rozsah_sem, weeks_sem = self._n_hodin(2)
        if par_sem > 0:
            assert par_sem == 1
            assert weeks_pred == 0 and weeks_cv == 0
            assert rozsah_sem >0 and weeks_sem > 0
        else:
            if (par_pred > 0) and (par_cv > 0):
                if weeks_pred != weeks_cv:
                    print(f"{self.label}({self.rok}), weeks_pr {weeks_pred} != weeks_cv {weeks_cv}")
        return VyukaPr(rozsah_pred * weeks_pred, rozsah_cv * weeks_cv, rozsah_sem * weeks_sem,
                       par_cv, None)





@mem.cache
def get_rozvrh(stag_client, katedry: List[str], years: List[int]) -> Dict[int, RozvrhovaAkce]:
    rozvrh = []

    for katedra in tqdm(katedry, desc='rozvrh'):
        for rok in years:
            for semestr in ['ZS', 'LS']:
                # Example 1: Get schedule by department
                rozvrh_request = GetRozvrhByKatedraRequest(katedra, rok, semestr)
                ra_res = stag_call(stag_client, rozvrh_request)['rozvrhovaAkce']
                pretty_print_yaml(ra_res, fname=f'rozvrh_{rok}_{katedra}_{semestr}.yaml')
                add = [initialize_from_dict(RozvrhovaAkce, ra, tydny_guess(ra), datum_guess(ra), hodiny_guess(ra)) for ra in ra_res]
                filter_fake_actions = [(i, ra) for i, ra in add if not ra.is_fake()]
                extend_valid(rozvrh, filter_fake_actions)
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


def get_studenti(stag_client, akce: Dict[int, RozvrhovaAkce],
                 predmety: Dict[PredmetIdx, Predmet]):

    predmety_ = {}
    for r,k,z,p in predmety.keys():
        predmety_.setdefault( (r,k,z), set()).add(p)
    predmety_akce = {rkz: PredmetAkce.from_predmet(predmety, rkz, prg_set)
                     for rkz, prg_set in predmety_.items()}

    missing_predmety = set()
    predmety_lock = Lock()
    missing_predmety_lock = Lock()

    def process_ra(ra):
        local_missing_predmety = set()
        studenti_req = GetStudentiByRoakceRequest(roakIdno=str(ra.roakIdno))
        try:
            response = stag_call(stag_client, studenti_req)
            studenti_data = response.get('studentPredmetu', [])
        except Exception as e:
            print(f"Error fetching studenti for roakIdno {ra.roakIdno}: {e}")
            return  # Skip this ra if there's an error

        studenti = [initialize_from_dict(Student, s) for s in studenti_data]
        loc_prg_predmet = {}
        for _, s in studenti:
            loc_prg_students = loc_prg_predmet.setdefault(s.stprIdno, set())
            loc_prg_students.add(s.osCislo)

        for stprIdno, students in loc_prg_predmet.items():
            predmet_akce_idx = (ra.rok, ra.katedra, ra.predmet)
            prg_id = int(stprIdno)
            with predmety_lock:
                try:
                    predmet_akce = predmety_akce[predmet_akce_idx]
                except KeyError:
                    local_missing_predmety.add(predmet_akce_idx)
                    continue
                predmet_akce.union_students(prg_id, students)
                predmet_akce.add_ra(ra)

                # prg: 1277, 1267
                # KMAGSW students from 1277  missing: P22000433 (Kombinovaná forma), P21000333 ('J')
                # new ver. has 7, SATG: 9, [ new_ prg has 1, orig. prg. : has 27 ???]
                # 2 students from 1267 lost, students dict finally with 1278 other prg
                # except KeyError:
                #     print(f"Unsupported typAkceZkr: {ra.typAkceZkr}, {ra.roakIdno}")

        with missing_predmety_lock:
            missing_predmety.update(local_missing_predmety)

    # Use ThreadPoolExecutor to process requests concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_ra, ra): ra for ra in akce.values()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing akce"):
            pass  # We're handling everything within process_ra

    #for ra in tqdm(akce.values(), desc="Processing akce"):
    #    process_ra(ra)
    # After all threads complete, print any missing predmety
    pd.DataFrame(missing_predmety).to_csv(workdir / 'missing_predmety_for_RA.csv')
    for p in missing_predmety:
        print(f"RA for missing Predmet oboru: {p}")

    return predmety_akce



@mem.cache
def read_stag(years, katedry=None):
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
    obory = studijni_obory(stag_client, list(programy.values()), years)
    #pretty_print_yaml(obory, fname='obory.yaml')
    predmety = get_predmety(stag_client, list(obory.values()), years)
    # remove CDV (univerzita tretiho veku)
    predmety = {k: v for k, v in predmety.items() if v.fakulta_programu != 'CDV'}
    if katedry is None:
        katedry = {p.katedra for p in predmety.values()}
    print(len(predmety))

    rozvrhove_akce = get_rozvrh(stag_client, katedry, years)
    pretty_print_yaml(rozvrhove_akce, "rozvrhove_akce.yaml")
    #krouzky_kod = {k for ra in rozvrhove_akce.values() for k in ra.krouzky_list()}
    #print(krouzky_kod)

    # update 'n_students, n_cv, n_pr' in Predmety
    predmet_akce = get_studenti(stag_client, rozvrhove_akce, predmety)
    return predmet_akce, rozvrhove_akce, programy
