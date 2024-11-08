import yaml
from KENs import KENs
from stag import StudijniProgram
import pathlib
script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"

class MyLoad(yaml.SafeLoader):
    def through(self, node):
        return self.construct_mapping(node)

MyLoad.add_constructor(None, MyLoad.through)

with open(workdir / "programy.yaml") as f:
    programy = yaml.load(f, Loader=MyLoad)
for p in programy.values():
    p['KEN'] = None

name_ken = {}
for kod, ken in KENs.items():
    for p in programy.values():
        if p['kod'] == kod:
            if p['nazev'] in name_ken:
                orig_ken, orig_kod = name_ken[p['nazev']]
                if orig_ken != ken:
                    name_ken[p['nazev']] = (None, None)
                    #print(f"KEN mismatch {orig_ken}, {orig_kod} != {ken}, {kod} for {p['nazev']}")
                    continue
            name_ken[p['nazev']] = (ken, kod)
for p in programy.values():
    if p['akreditaceDoDate'] is None\
       or p['akreditaceDoDate'] < 2024 - 4\
       or p['typ'] == 'Doktorský'\
       or p['forma'] == 'Kombinovaná':
        continue
    if p['KEN'] is None:
        p['KEN'] = name_ken.get(p['nazev'], None)
    if p['KEN'] is None:
            print(f"Missing KEN for {p}")

