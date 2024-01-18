"""
Script to run to extract features from all wav files in a folder structure.
Copies the structure from an existing folder to a new one and extract Articulation, Prosody and Phonation features. 
Might need to change variables about what folder to extract features from, where to put the new folder and the new folder name before you run the script. 
If you want to change between static and dynamic features or change settings for how the features are saved you need to change that as well.
"""

import os
import sys
import warnings

from constants import path_PC_GITA_16k, personal_path_to_code, personal_path_to_kaldi

os.environ['KALDI_ROOT'] = personal_path_to_kaldi
warnings.filterwarnings("ignore",
                        category=RuntimeWarning)  # To remove warnings for prosody about potentially not getting reliable results.
sys.path.append("../")

from disvoice.articulation.articulation import Articulation
from disvoice.phonation.phonation import Phonation
from disvoice.prosody.prosody import Prosody

############### VARIABLES TO CHANGE IF WANTED ###############

path_to_waveforms = path_PC_GITA_16k  # Defines the path to the folder with all the waveforms you want features computed from
path_to_feature_saving = personal_path_to_code  # Defines the path to where the resulting features will be saved.
new_folder_name = sys.argv[1] # Defines the name of the folder that will be created and all the features will be saved in. Takes 1. argument given when running the script from the terminal. 

features_same_file = True  # True if you want all features for the same word saved so the same file, False if not. If true then features will be static no mather the def of "static".
static = True  # False if you want dynamic features. Only matters to set if features_same_file = False.


# For dynamic features: / To be fixed before dynamic features are used
#   I get a warning for prosody --> avoid prosody for warning for now, but should be fixed.
#   I also get nothing out for Articulation --> should also be fixed. 

###################################################

def create_new_folder(path, name):
    """
    Creates a new folder with "name" in at "path" location if there is not already a folder with that name. 

    Args:
        path (str): path to where we want the new folder
        name (str): name of the new folder

    Returns:
        str: path to the new folder created. Or False if such a folder already exists. 
    """
    new_folder_path = os.path.join(path, name)
    if os.path.exists(new_folder_path):
        print(f"Already have a {name} folder in this location. Remove it if you want a new created. ")
        new_folder_path = False
    else:
        os.makedirs(new_folder_path)
    return new_folder_path


def copy_folder_structure(source_folder, new_folder):
    """
    Creates corresponding directories in the new_folder as we have in the source_folder without
    copying the files. 
    Args:
        source_folder (str): the path to the folder we want to copy from. 
        new_folder (str): the path to the folder we want to copy to. 
    """
    for dirpath, dirnames, filenames in os.walk(source_folder):
        for dirname in dirnames:
            source_path = os.path.join(dirpath, dirname)
            new_path = os.path.join(new_folder, os.path.relpath(source_path, source_folder))
            os.makedirs(new_path, exist_ok=True)


def add_feature_content_in_same_file(waveform_folder, feature_folder):
    """
    Uses DisVoice to create Articulation, Prosody and Phonation features from all waveforms in 
    waveform_folder, and saving them with the same structure in feature_folder, but one feature file for each of the 3
    3 feature types per folder of wav files.. 

    Args:
        waveform_folder (str): path to folder where we have the waveforms we want to extract features from
        feature_folder (str): path to folder where we want the features located. Needs to have same structure as waveform_folder. 
    """
    for dirpath, dirnames, filenames in os.walk(waveform_folder):
        if filenames != [] and filenames != [
            '.DS_Store']:  # Avoids folder without filenames and folder with ['.DS_Store'] files.
            if "bodega" not in filenames[0] and "clavo" not in filenames[
                0]:  # Avoids bodega and clavo folders due to problems there.
                rel_path = os.path.relpath(dirpath, waveform_folder)
                waveform_path = os.path.join(waveform_folder, rel_path) + "/"
                print("Working on files in ", waveform_path, " ...")
                feature_path = os.path.join(feature_folder, rel_path)

                phonation_file = os.path.join(feature_path, "Phonation.csv")
                phonation_features = phonationf.extract_features_path(waveform_path, static=True, plots=False, fmt="csv")
                phonation_features.to_csv(phonation_file, index=False)

                articulation_file = os.path.join(feature_path, "Articulation.csv")
                articulation_features = articulationf.extract_features_path(waveform_path, static=True, plots=False,
                                                                            fmt="csv")
                articulation_features.to_csv(articulation_file, index=False)

                prosody_file = os.path.join(feature_path, "Prosody.csv")
                prosody_features = prosodyf.extract_features_path(waveform_path, static=True, plots=False, fmt="csv")
                prosody_features.to_csv(prosody_file, index=False)


def add_feature_content_in_seperate_files(waveform_folder, feature_folder, static):
    """
    Uses DisVoice to create Articulation, Prosody and Phonation features from all waveforms in 
    waveform_folder, and saving them with the same structure in feature_folder where each feature type for each wav file has one new file. 
    Used if features_same_file = False --> if you want seperate files for the features of each wav file, or if you want dynamic features. 
        # Need to do it like this for dynamic features since they give more than one row of features for each wav file. 

    Args:
        waveform_folder (str): path to folder where we have the waveforms we want to extract features from
        feature_folder (str): path to folder where we want the features located. Needs to have same structure as waveform_folder. 
        static (bool): boolean value that defines if the features to be extracted are static or dynamic. 
    """
    for dirpath, dirnames, filenames in os.walk(waveform_folder):
        if filenames != [] and filenames != [
            '.DS_Store']:  # Avoids folder without filenames and folder with ['.DS_Store'] files.
            if "bodega" not in filenames[0] and "clavo" not in filenames[
                0]:  # Avoids bodega and clavo folders due to problems there.
                rel_path = os.path.relpath(dirpath, waveform_folder)
                waveform_path = os.path.join(waveform_folder, rel_path) + "/"
                print("Working on files in ", waveform_path, " ...")
                feature_path = os.path.join(feature_folder, rel_path)

                for dirpath, dirnames, filenames in os.walk(waveform_path):
                    for filename in filenames:
                        filepath = os.path.join(waveform_path, filename)

                        phonation_file = os.path.join(feature_path, filename).split('.wav')[0] + "_phonation.csv"
                        features1 = phonationf.extract_features_file(filepath, static=static, plots=False, fmt="csv")
                        features1.to_csv(phonation_file, index=False)

                        articulation_file = os.path.join(feature_path, filename).split('.wav')[0] + "_articulation.csv"
                        features1 = articulationf.extract_features_file(filepath, static=static, plots=False, fmt="csv")
                        features1.to_csv(articulation_file, index=False)

                        if static:  # Done so to avoide prosody if extracting dynamic features due to error with that.
                            prosody_file = os.path.join(feature_path, filename).split('.wav')[0] + "_prosody.csv"
                            features1 = prosodyf.extract_features_file(filepath, static=static, plots=False, fmt="csv")
                            features1.to_csv(prosody_file, index=False)


phonationf = Phonation()
articulationf = Articulation()
prosodyf = Prosody()

PC_GITA_features = create_new_folder(path_to_feature_saving, new_folder_name)
if PC_GITA_features:
    copy_folder_structure(path_to_waveforms, PC_GITA_features)
    if features_same_file:
        print(
            "Adding features of all wav files from the same folder in PC-GITA to the same file. Using static features. ")
        add_feature_content_in_same_file(path_to_waveforms, PC_GITA_features)
    else:
        print("Adding features of all wav files to different file.")
        add_feature_content_in_seperate_files(path_to_waveforms, PC_GITA_features, static)
