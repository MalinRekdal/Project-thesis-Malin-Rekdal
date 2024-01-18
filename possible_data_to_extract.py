"""
File that has all paths to the different types of analysis.
This file is created manually, so have subject to errors (specially with the words).

This file can be changed to include the different ddk, vowels, sentences or words you want to use.
"""

############### DDK_analysis ##########################
ddk = ["ka-ka-ka", "pa-pa-pa", "pakata", "pataka", "petaka", "ta-ta-ta"]
hc_paths_ddk = ["/DDK_analysis/" + elem + "/sin_normalizar/" + ("HC" if elem=="pataka" else "hc") for elem in ddk]
pd_paths_ddk = ["/DDK_analysis/" + elem + "/sin_normalizar/" + ("PD" if elem=="pataka" else "pd") for elem in ddk]

############### modulated_vowels ##########################
mod_vowels = ["A", "E", "I", "O", "U"]
hc_paths_mod_vowels = ["/modulated_vowels/hc/" + elem for elem in mod_vowels]
pd_paths_mod_vowels = ["/modulated_vowels/pd/" + elem for elem in mod_vowels]

############### monologue ##########################
hc_paths_monologue = ["/monologue/sin_normalizar/hc"]
pd_paths_monologue = ["/monologue/sin_normalizar/pd"]

############### read text ##########################
hc_paths_read_text = ["/read_text/ayerfuialmedico/sin_normalizar/hc"]
pd_paths_read_text = ["/read_text/ayerfuialmedico/sin_normalizar/pd"]

############### senteces ##########################
sentences = ["laura", "loslibros", "luisa", "micasa", "omar", "rosita"]
hc_paths_sentences = ["/sentences/" + elem + "/sin_normalizar/HC" for elem in sentences]
pd_paths_sentences = ["/sentences/" + elem + "/sin_normalizar/PD" for elem in sentences]

############### sentences2 ##########################
sentences2 = ["1_viste", "2_juan", "3_triste", "4_preocupado"]
hc_paths_sentences2 = ["/sentences2/" + elem + "/non-normalized/hc" for elem in sentences2]
pd_paths_sentences2 = ["/sentences2/" + elem + "/non-normalized/pd" for elem in sentences2]

############### Vowels ##########################
vowels = ["A", "E", "I", "O", "U"]
hc_paths_vowels = ["/Vowels/" + "Control/" + elem for elem in vowels]
pd_paths_vowels = ["/Vowels/" + "Patologicas/" + elem for elem in vowels]

############### Words ##########################
# All words:
""" 
words = ["apto", "atleta", "blusa", "bodega", "braso", "campana", "caucho", "clavo", "coco", "crema", "drama", "flecha",
        "fruta", "gato", "globo", "grito", "llueve", "nÌƒame", "pato", "petaka", "plato", "presa", "reina", "trato",
        "viaje"]
"""
# bodega and clavo will be removed because they will not work either way (because of the feature extraction method)
# nÌƒame will be scipped because it is not able to read the name.
# Fruta also has 49 instead of 50 repetitions for PD (missing person nr. 16) so remove this as well.

# Adjusted words: 
words = ["apto", "atleta", "blusa", "braso", "campana", "caucho", "coco", "crema", "drama", "flecha",
        "gato", "globo", "grito", "llueve", "pato", "petaka", "plato", "presa", "reina", "trato",
        "viaje"]
hc_paths_words = ["/Words/Sin_normalizar/Control/" + elem for elem in words]
pd_paths_words = ["/Words/Sin_normalizar/Patologica/" + elem for elem in words]


############### All utterances ######################
hc_paths_all = hc_paths_ddk + hc_paths_monologue + hc_paths_read_text + hc_paths_sentences + hc_paths_sentences2 + hc_paths_words + hc_paths_mod_vowels + hc_paths_vowels
pd_paths_all = pd_paths_ddk + pd_paths_monologue + pd_paths_read_text + pd_paths_sentences + pd_paths_sentences2 + pd_paths_words + pd_paths_mod_vowels + pd_paths_vowels 

############### Continuous speech ######################
hc_paths_continuous_speech = hc_paths_ddk + hc_paths_monologue + hc_paths_read_text + hc_paths_sentences + hc_paths_sentences2
pd_paths_continuous_speech = pd_paths_ddk + pd_paths_monologue + pd_paths_read_text + pd_paths_sentences + pd_paths_sentences2 
