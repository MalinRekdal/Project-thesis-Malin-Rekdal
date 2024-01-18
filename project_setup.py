# Setup file to run things that is common for to run for multiple files in the beginning. 
# Run it in another file in the beginning by using %run project_setup.py

import os
import sys
import warnings
from constants import personal_path_to_kaldi

sys.path.append("../")

# Set kaldi environment 
os.environ['KALDI_ROOT']=personal_path_to_kaldi


# To remove warning about potentially not getting reliable results. Done to get a cleaner write out.
# Warning that creates NaN values (mostly for prosody). 
warnings.filterwarnings("ignore", category=RuntimeWarning)  









