# Project-thesis for Malin Rekdal
 - Title: Automatic Speech Analysis for Detection of Parkinson’s Disease
 - Sub title: Feature analysis for optimizing Parkinson’s disease detection

Supervisor: Torbjørn Karl Svendsen

Co-supervisors: Maria Francesca Torco and Marco Sabato Siniscalchi 

 This code is created for a project in the subject "TFE4580 1 Elektronisk systemdesign og innovasjon, fordypningsprosjekt", which is the project thesis for the study Electronical System Design and Innovation in NTNU. This is the pre-work before a master thesis on the same subject. The main focus for this code in this project has been on setting up an environment that is easly adapeble to the different experiments that is wanted to do. Therefore a lot of the code in this repository is made generic, like the create_features_from_PC_GITA can also be used for dynamic features in the future by only doing small adjustments, and the validations methods are designed to work with other model types like CNN as well. 



 The code in this repo is also saved in localhome/studenter/malinre/project-thesis in Aulus3. 
 

## 1 -  What this repository contains: 
- #### Code: 
  - readMe: this file with an introduction to the project code. 
  - constants: constant definitions that is used throughout the other code
  - functions: function definitions that is used throughout the other code
  - project_setup: Setup file to run things that is common for many files in the beginning. 
  - possible_data_to_extract: python file that contains all utterance types we can get the features from through PC-GITA-static-features 
  - features_to_choose_from: md file with an overview of the Phonation, Articulation and Prosody features we can choose between. 
  - analyse_metadata: jupyter notebook that analyses the metadata from PC-GITA.
  - analyse_signal: jupyter notebook that analyses the waveform signals from PC-GITA and extracts some features 
  - create_features_from_PC_GITA: python script that creates a folder like the PC-GITA-static-features folder 
  - model_and_features_eval: Plotting of features, training model and evaluating models. The main file for the project. 
  - simpler_model_to_run: Essentially the same as "model_and_features_eval" but stripped down to be faster to re run and get exactly what is needed for each test of the model. 

- #### PC-GITA-static-features: Folder that contains all features for all utterences from the PC-GITA dataset. 
- #### Modifications needed for DisVoice. Is the needed files to make DisVoice work. Created by Torbjørn Karl Svendsen. 
- #### Images: Folder with images of used in the report for the project. 
- #### Excell litterature review: Litterature review summary for state of the art. 


## 2 - Getting started


### 2 choises for what to do: 
1) #### It is only the create_features_from_PC_GITA and analyse_signals files that needs the setup with DisVoice, Praat and Kaldi. Therefore you can choose to skip the steps underneath and just use the features in PC-GITA-features file that is attached and run the rest of the code as normal. Then the packages you need must be installed separately as well. 

2) #### If you do do the steps underneath and set up then you can run create_features_from_PC-GITA to create features for all data. This file needs to be run with one system argument that is a string, and corresponds to the name of a folder you want the features saved to. A folder will then be created with that name and phonation, articulation and prosody features will be saved to that folder. Then you use this folder of features from the PC-GITA-data file and run the rest of the code.

### The steps underneath will explain how to set up an environment on NTNU's Aulus computers or any other Linux device and install everything needed for this project. This is based on my experience and the problems I acountered.  


### 2.1 - Setup environment

- Install conda or Miniconda

- Set up virtual env 
  - Create environment 

    ```
    conda create -n disenv python=3.11.5 ipython ipykernel
    ```

  - Activate and deactivate environment
    ```
    conda activate disvoice
    ```
    ```
    conda deactivate
    ```
- Create a kernal using: 
  ```
  python -m ipykernel install --user --name disenv --display-name "disenv"
  ```



### 2.2 - Make sure pip is linked to the virtual environmentet 
- Use
  ```
  which pip
  ```
- If this is not the pip related to the env you have created you need to add an alias to the .bashrc file by adding this line at the end: 
  ```
  alias disenvPipp=/localhome/studenter/user_name/miniconda3/envs/disenv/bin/pip
  ```
  Then use disenvPipp instead of pip when installing things. 

### 2.3 - Install requirements to conda environment 
- Install praat: 
  ```
		wget https://www.fon.hum.uva.nl/praat/praat6318_linux64nogui.tar.gz 
		gunzip praat6318_linux64nogui.tar.gz
		tar xvf praat6318_linux64nogui.tar
		rm praat6318_linux64nogui.tar
		mv praat_nogui praat 
  ```

- Install DisVoice to same location as Praat

  ```
  pip install disvoice
  ```
  Note that pip install disvoice installs both pythorch and tensorflow and the other packages that is needed to use DisVoice. 
  Can also clone Disvoice repo if nootebook examples are wanted: 
  ```
  git clone https://github.com/jcvasquezc/DisVoice.git
  ```


- Install kaldi: 
  ```
		git clone https://github.com/kaldi-asr/kaldi
  ```
  Follow the instructions in the INSTALL file in order to install the toolkit. This might need an earlier version of python, and in that case you can create such an environment like this: 
  ```
		conda create -name disvoice27 python=2.7
  ```
  
  Do the installation and then go back to the disenv environment created earlier with a newer python version. 


### 2.4 - Changes needed to DisVoice package: 
  - Take the files "FormantsPraat.praat, praat_functions and vuv_praat.praat" from Modifications needed for DisVoice and change the files in the DisVoice installation. The articulation_features_fixed is also an improved version of the example notebook for Articulation that comes with DisVoice. 

  - In miniconda/env/disvoice you need to change 2 files: 

    - ./miniconda3/envs/disvoice/lib/python3.11/sitepackages/disvoice/articulation/articulation_functions.py
        - line 68: change np.int to np.int64

    - ./miniconda3/envs/disvoice/lib/python3.11/site-packages/disvoice/praat/praat_functions.py
      -	change the beginning of lines 44 and 68
        -	from command='praat '+PATH… 
        -	to command='praat --run '+PATH... 
        - Note: important to have " " after "--run". 



## 3 - Update constant.py according to your setup: 
- personal_path_to_disvoice = the path to where DisVoice repo is cloned to. 
- personal_path_to_kaldi = the path to where kaldi repo is cloned to.
- personal_path_to_code =  the path to where this repo is cloned to.
- personal_path_to_PC_GITA = path to the PC-GITA dataset
  - This has multiple choises. 
    1) can be a link in Aulus to the data that is located in: /talebase/data/speech_raw
    2) Can copy the data to a location of your choise in Aulus like this: 
      ```
      mkdir PC-GITA
      ln -s /talebase/data/speech_raw/PC-GITA/* path/to/new/location 
      ```
    3) Can copy the data to a local computer (windows) to work on it offline: 
        - Do step 2. 
        - Download putty and add putty to path. 
        - Copy the data using putty: 
          ```
          pscp -r username@aulusX.ies.ntnu.no:\path/to/new/location/for/PC-GITA  C:\Users\username\path\to\local\location
          ```
      