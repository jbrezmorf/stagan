from typing import Dict, List, Tuple
import pathlib
import yaml

script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"
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
                pd.options.display.float_format = '{:.3f}'.format
                # Create a string for the header with commas
                header_str = ','.join(data.columns.astype(str)) + ',\n'
                data_str = data.astype(str).applymap(lambda x: x + ',')
                f.write(header_str)
                f.write(data_str.to_string(index=False, header=False))

            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=True)
