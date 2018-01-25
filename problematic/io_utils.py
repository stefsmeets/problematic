import yaml
import pandas as pd
import io
import numpy as np
import hyperspy.api as hs


def get_files(file_pat):
    """Grab files from globbing pattern or stream file"""
    if os.path.exists(file_pat):
        root, ext = os.path.splitext(file_pat)
        if ext.lower() == ".ycsv":
            df, d = read_ycsv(file_pat)
            fns = df.index.tolist()
        else:
            f = open(file_pat, "r")
            fns = [line.split("#")[0].strip() for line in f if not line.startswith("#")]
    else:
        fns = glob.glob(file_pat)

    if len(fns) == 0:
        raise IOError("No files matching '{}' were found.".format(file_pat))

    return fns


def save_orientations(orientations, out="orientations.npy"):
    np.save(out, orientations.data)


def load_orientations(fin="orientations.npy"):
    orientations = hs.signals.Signal1D(np.load(fin))
    orientations.axes_manager.set_signal_dimension(0)
    return orientations


def read_ycsv(f):
    """
    read file in ycsv format:
    https://blog.datacite.org/using-yaml-frontmatter-with-csv/
    
    format:
        ---
        $YAML_BLOCK
        ---
        $CSV_BLOCK
    """
    
    if isinstance(f, str):
        f = open(f, "r")
    
    first_line = f.tell()
    
    in_yaml_block = False
    
    yaml_block = []
    
    for line in f:
        if line.strip() == "---":
            if not in_yaml_block:
                in_yaml_block = True
            else:
                in_yaml_block = False
                break
            continue
                
        if in_yaml_block:
            yaml_block.append(line)
    
    # white space is important when reading yaml
    d = yaml.load(io.StringIO("".join(yaml_block)))
    
    # workaround to fix pandas crash when it is not at the first line for some reason
    f.seek(first_line)
    header = len(yaml_block) + 2
    try:
        df = pd.DataFrame.from_csv(f, header=header)
    except pd.io.common.EmptyDataError:
        df = None
        
    # print "".join(yaml_block)
    
    return df, d


def write_ycsv(f, data, metadata):
    """
    write file in ycsv format:
    https://blog.datacite.org/using-yaml-frontmatter-with-csv/
    
    format:
        ---
        $YAML_BLOCK
        ---
        $CSV_BLOCK
    """
        
    if isinstance(f, str):
        f = open(f, "w")
    f.write("---\n")
    yaml.dump(metadata, f, default_flow_style=False)
    f.write("---\n")
    data.to_csv(f)
    f.close()

