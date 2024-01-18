# Setup file to run things that is common for many files in the beginning. 
# Run it in another file in the beginning by using %run project_setup.py


import os
import sys
from constants import personal_path_to_kaldi

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) # To remove warnings for prosody about potentially not getting reliable results. 

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # To avoide tensorflow warning 
# import tensorflow as tf

sys.path.append("../")

# Set kaldi environment 
os.environ['KALDI_ROOT']=personal_path_to_kaldi

print("HEI")






