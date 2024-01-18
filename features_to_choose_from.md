# Possible features to extract
In addition to the features, each of them also has an id column, that is the same for all features types for one waveform. 

## Phonation

- Static matrix is formed with 29 features formed with (7 descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced
(Or we actually dont get degree of unvoiced --> 28 features instead. )
    1. First derivative of the fundamental Frequency
    2. Second derivative of the fundamental Frequency
    3. Jitter
    4. Shimmer
    5. Amplitude perturbation quotient
    6. Pitch perturbation quotient
    7. Logaritmic Energy

```['avg DF0', 'avg DDF0', 'avg Jitter', 'avg Shimmer', 'avg apq', 'avg ppq', 'avg logE', 'std DF0', 'std DDF0', 'std Jitter', 'std Shimmer', 'std apq', 'std ppq', 'std logE', 'skewness DF0', 'skewness DDF0', 'skewness Jitter', 'skewness Shimmer', 'skewness apq', 'skewness ppq', 'skewness logE', 'kurtosis DF0', 'kurtosis DDF0', 'kurtosis Jitter', 'kurtosis Shimmer', 'kurtosis apq', 'kurtosis ppq', 'kurtosis logE']```


## Articulation

- 1 - 22. Bark band energies in onset transitions (22 BBE).
- 23 - 34. Mel frequency cepstral coefficients in onset transitions (12 MFCC onset)
- 35 - 46. First derivative of the MFCCs in onset transitions (12 DMFCC onset)
- 47 - 58. Second derivative of the MFCCs in onset transitions (12 DDMFCC onset)
- 59 - 80. Bark band energies in offset transitions (22 BBE).
- 81 - 92. MFCCC in offset transitions (12 MFCC offset)
- 93 - 104. First derivative of the MFCCs in offset transitions (12 DMFCC offset)
- 105 - 116. Second derivative of the MFCCs in offset transitions (12 DMFCC offset)
- 117 First formant Frequency
- 118 First Derivative of the first formant frequency
- 119 Second Derivative of the first formant frequency
- 120 Second formant Frequency
- 121 First derivative of the Second formant Frequency
- 122 Second derivative of the Second formant Frequency

```['avg BBEon_1', 'avg BBEon_2', 'avg BBEon_3', 'avg BBEon_4', 'avg BBEon_5', 'avg BBEon_6', 'avg BBEon_7', 'avg BBEon_8', 'avg BBEon_9', 'avg BBEon_10', 'avg BBEon_11', 'avg BBEon_12', 'avg BBEon_13', 'avg BBEon_14', 'avg BBEon_15', 'avg BBEon_16', 'avg BBEon_17', 'avg BBEon_18', 'avg BBEon_19', 'avg BBEon_20', 'avg BBEon_21', 'avg BBEon_22', 'avg MFCCon_1', 'avg MFCCon_2', 'avg MFCCon_3', 'avg MFCCon_4', 'avg MFCCon_5', 'avg MFCCon_6', 'avg MFCCon_7', 'avg MFCCon_8', 'avg MFCCon_9', 'avg MFCCon_10', 'avg MFCCon_11', 'avg MFCCon_12', 'avg DMFCCon_1', 'avg DMFCCon_2', 'avg DMFCCon_3', 'avg DMFCCon_4', 'avg DMFCCon_5', 'avg DMFCCon_6', 'avg DMFCCon_7', 'avg DMFCCon_8', 'avg DMFCCon_9', 'avg DMFCCon_10', 'avg DMFCCon_11', 'avg DMFCCon_12', 'avg DDMFCCon_1', 'avg DDMFCCon_2', 'avg DDMFCCon_3', 'avg DDMFCCon_4', 'avg DDMFCCon_5', 'avg DDMFCCon_6', 'avg DDMFCCon_7', 'avg DDMFCCon_8', 'avg DDMFCCon_9', 'avg DDMFCCon_10', 'avg DDMFCCon_11', 'avg DDMFCCon_12', 'avg BBEoff_1', 'avg BBEoff_2', 'avg BBEoff_3', 'avg BBEoff_4', 'avg BBEoff_5', 'avg BBEoff_6', 'avg BBEoff_7', 'avg BBEoff_8', 'avg BBEoff_9', 'avg BBEoff_10', 'avg BBEoff_11', 'avg BBEoff_12', 'avg BBEoff_13', 'avg BBEoff_14', 'avg BBEoff_15', 'avg BBEoff_16', 'avg BBEoff_17', 'avg BBEoff_18', 'avg BBEoff_19', 'avg BBEoff_20', 'avg BBEoff_21', 'avg BBEoff_22', 'avg MFCCoff_1', 'avg MFCCoff_2', 'avg MFCCoff_3', 'avg MFCCoff_4', 'avg MFCCoff_5', 'avg MFCCoff_6', 'avg MFCCoff_7', 'avg MFCCoff_8', 'avg MFCCoff_9', 'avg MFCCoff_10', 'avg MFCCoff_11', 'avg MFCCoff_12', 'avg DMFCCoff_1', 'avg DMFCCoff_2', 'avg DMFCCoff_3', 'avg DMFCCoff_4', 'avg DMFCCoff_5', 'avg DMFCCoff_6', 'avg DMFCCoff_7', 'avg DMFCCoff_8', 'avg DMFCCoff_9', 'avg DMFCCoff_10', 'avg DMFCCoff_11', 'avg DMFCCoff_12', 'avg DDMFCCoff_1', 'avg DDMFCCoff_2', 'avg DDMFCCoff_3', 'avg DDMFCCoff_4', 'avg DDMFCCoff_5', 'avg DDMFCCoff_6', 'avg DDMFCCoff_7', 'avg DDMFCCoff_8', 'avg DDMFCCoff_9', 'avg DDMFCCoff_10', 'avg DDMFCCoff_11', 'avg DDMFCCoff_12', 'avg F1', 'avg DF1', 'avg DDF1', 'avg F2', 'avg DF2', 'avg DDF2', 'std BBEon_1', 'std BBEon_2', 'std BBEon_3', 'std BBEon_4', 'std BBEon_5', 'std BBEon_6', 'std BBEon_7', 'std BBEon_8', 'std BBEon_9', 'std BBEon_10', 'std BBEon_11', 'std BBEon_12', 'std BBEon_13', 'std BBEon_14', 'std BBEon_15', 'std BBEon_16', 'std BBEon_17', 'std BBEon_18', 'std BBEon_19', 'std BBEon_20', 'std BBEon_21', 'std BBEon_22', 'std MFCCon_1', 'std MFCCon_2', 'std MFCCon_3', 'std MFCCon_4', 'std MFCCon_5', 'std MFCCon_6', 'std MFCCon_7', 'std MFCCon_8', 'std MFCCon_9', 'std MFCCon_10', 'std MFCCon_11', 'std MFCCon_12', 'std DMFCCon_1', 'std DMFCCon_2', 'std DMFCCon_3', 'std DMFCCon_4', 'std DMFCCon_5', 'std DMFCCon_6', 'std DMFCCon_7', 'std DMFCCon_8', 'std DMFCCon_9', 'std DMFCCon_10', 'std DMFCCon_11', 'std DMFCCon_12', 'std DDMFCCon_1', 'std DDMFCCon_2', 'std DDMFCCon_3', 'std DDMFCCon_4', 'std DDMFCCon_5', 'std DDMFCCon_6', 'std DDMFCCon_7', 'std DDMFCCon_8', 'std DDMFCCon_9', 'std DDMFCCon_10', 'std DDMFCCon_11', 'std DDMFCCon_12', 'std BBEoff_1', 'std BBEoff_2', 'std BBEoff_3', 'std BBEoff_4', 'std BBEoff_5', 'std BBEoff_6', 'std BBEoff_7', 'std BBEoff_8', 'std BBEoff_9', 'std BBEoff_10', 'std BBEoff_11', 'std BBEoff_12', 'std BBEoff_13', 'std BBEoff_14', 'std BBEoff_15', 'std BBEoff_16', 'std BBEoff_17', 'std BBEoff_18', 'std BBEoff_19', 'std BBEoff_20', 'std BBEoff_21', 'std BBEoff_22', 'std MFCCoff_1', 'std MFCCoff_2', 'std MFCCoff_3', 'std MFCCoff_4', 'std MFCCoff_5', 'std MFCCoff_6', 'std MFCCoff_7', 'std MFCCoff_8', 'std MFCCoff_9', 'std MFCCoff_10', 'std MFCCoff_11', 'std MFCCoff_12', 'std DMFCCoff_1', 'std DMFCCoff_2', 'std DMFCCoff_3', 'std DMFCCoff_4', 'std DMFCCoff_5', 'std DMFCCoff_6', 'std DMFCCoff_7', 'std DMFCCoff_8', 'std DMFCCoff_9', 'std DMFCCoff_10', 'std DMFCCoff_11', 'std DMFCCoff_12', 'std DDMFCCoff_1', 'std DDMFCCoff_2', 'std DDMFCCoff_3', 'std DDMFCCoff_4', 'std DDMFCCoff_5', 'std DDMFCCoff_6', 'std DDMFCCoff_7', 'std DDMFCCoff_8', 'std DDMFCCoff_9', 'std DDMFCCoff_10', 'std DDMFCCoff_11', 'std DDMFCCoff_12', 'std F1', 'std DF1', 'std DDF1', 'std F2', 'std DF2', 'std DDF2', 'skewness BBEon_1', 'skewness BBEon_2', 'skewness BBEon_3', 'skewness BBEon_4', 'skewness BBEon_5', 'skewness BBEon_6', 'skewness BBEon_7', 'skewness BBEon_8', 'skewness BBEon_9', 'skewness BBEon_10', 'skewness BBEon_11', 'skewness BBEon_12', 'skewness BBEon_13', 'skewness BBEon_14', 'skewness BBEon_15', 'skewness BBEon_16', 'skewness BBEon_17', 'skewness BBEon_18', 'skewness BBEon_19', 'skewness BBEon_20', 'skewness BBEon_21', 'skewness BBEon_22', 'skewness MFCCon_1', 'skewness MFCCon_2', 'skewness MFCCon_3', 'skewness MFCCon_4', 'skewness MFCCon_5', 'skewness MFCCon_6', 'skewness MFCCon_7', 'skewness MFCCon_8', 'skewness MFCCon_9', 'skewness MFCCon_10', 'skewness MFCCon_11', 'skewness MFCCon_12', 'skewness DMFCCon_1', 'skewness DMFCCon_2', 'skewness DMFCCon_3', 'skewness DMFCCon_4', 'skewness DMFCCon_5', 'skewness DMFCCon_6', 'skewness DMFCCon_7', 'skewness DMFCCon_8', 'skewness DMFCCon_9', 'skewness DMFCCon_10', 'skewness DMFCCon_11', 'skewness DMFCCon_12', 'skewness DDMFCCon_1', 'skewness DDMFCCon_2', 'skewness DDMFCCon_3', 'skewness DDMFCCon_4', 'skewness DDMFCCon_5', 'skewness DDMFCCon_6', 'skewness DDMFCCon_7', 'skewness DDMFCCon_8', 'skewness DDMFCCon_9', 'skewness DDMFCCon_10', 'skewness DDMFCCon_11', 'skewness DDMFCCon_12', 'skewness BBEoff_1', 'skewness BBEoff_2', 'skewness BBEoff_3', 'skewness BBEoff_4', 'skewness BBEoff_5', 'skewness BBEoff_6', 'skewness BBEoff_7', 'skewness BBEoff_8', 'skewness BBEoff_9', 'skewness BBEoff_10', 'skewness BBEoff_11', 'skewness BBEoff_12', 'skewness BBEoff_13', 'skewness BBEoff_14', 'skewness BBEoff_15', 'skewness BBEoff_16', 'skewness BBEoff_17', 'skewness BBEoff_18', 'skewness BBEoff_19', 'skewness BBEoff_20', 'skewness BBEoff_21', 'skewness BBEoff_22', 'skewness MFCCoff_1', 'skewness MFCCoff_2', 'skewness MFCCoff_3', 'skewness MFCCoff_4', 'skewness MFCCoff_5', 'skewness MFCCoff_6', 'skewness MFCCoff_7', 'skewness MFCCoff_8', 'skewness MFCCoff_9', 'skewness MFCCoff_10', 'skewness MFCCoff_11', 'skewness MFCCoff_12', 'skewness DMFCCoff_1', 'skewness DMFCCoff_2', 'skewness DMFCCoff_3', 'skewness DMFCCoff_4', 'skewness DMFCCoff_5', 'skewness DMFCCoff_6', 'skewness DMFCCoff_7', 'skewness DMFCCoff_8', 'skewness DMFCCoff_9', 'skewness DMFCCoff_10', 'skewness DMFCCoff_11', 'skewness DMFCCoff_12', 'skewness DDMFCCoff_1', 'skewness DDMFCCoff_2', 'skewness DDMFCCoff_3', 'skewness DDMFCCoff_4', 'skewness DDMFCCoff_5', 'skewness DDMFCCoff_6', 'skewness DDMFCCoff_7', 'skewness DDMFCCoff_8', 'skewness DDMFCCoff_9', 'skewness DDMFCCoff_10', 'skewness DDMFCCoff_11', 'skewness DDMFCCoff_12', 'skewness F1', 'skewness DF1', 'skewness DDF1', 'skewness F2', 'skewness DF2', 'skewness DDF2', 'kurtosis BBEon_1', 'kurtosis BBEon_2', 'kurtosis BBEon_3', 'kurtosis BBEon_4', 'kurtosis BBEon_5', 'kurtosis BBEon_6', 'kurtosis BBEon_7', 'kurtosis BBEon_8', 'kurtosis BBEon_9', 'kurtosis BBEon_10', 'kurtosis BBEon_11', 'kurtosis BBEon_12', 'kurtosis BBEon_13', 'kurtosis BBEon_14', 'kurtosis BBEon_15', 'kurtosis BBEon_16', 'kurtosis BBEon_17', 'kurtosis BBEon_18', 'kurtosis BBEon_19', 'kurtosis BBEon_20', 'kurtosis BBEon_21', 'kurtosis BBEon_22', 'kurtosis MFCCon_1', 'kurtosis MFCCon_2', 'kurtosis MFCCon_3', 'kurtosis MFCCon_4', 'kurtosis MFCCon_5', 'kurtosis MFCCon_6', 'kurtosis MFCCon_7', 'kurtosis MFCCon_8', 'kurtosis MFCCon_9', 'kurtosis MFCCon_10', 'kurtosis MFCCon_11', 'kurtosis MFCCon_12', 'kurtosis DMFCCon_1', 'kurtosis DMFCCon_2', 'kurtosis DMFCCon_3', 'kurtosis DMFCCon_4', 'kurtosis DMFCCon_5', 'kurtosis DMFCCon_6', 'kurtosis DMFCCon_7', 'kurtosis DMFCCon_8', 'kurtosis DMFCCon_9', 'kurtosis DMFCCon_10', 'kurtosis DMFCCon_11', 'kurtosis DMFCCon_12', 'kurtosis DDMFCCon_1', 'kurtosis DDMFCCon_2', 'kurtosis DDMFCCon_3', 'kurtosis DDMFCCon_4', 'kurtosis DDMFCCon_5', 'kurtosis DDMFCCon_6', 'kurtosis DDMFCCon_7', 'kurtosis DDMFCCon_8', 'kurtosis DDMFCCon_9', 'kurtosis DDMFCCon_10', 'kurtosis DDMFCCon_11', 'kurtosis DDMFCCon_12', 'kurtosis BBEoff_1', 'kurtosis BBEoff_2', 'kurtosis BBEoff_3', 'kurtosis BBEoff_4', 'kurtosis BBEoff_5', 'kurtosis BBEoff_6', 'kurtosis BBEoff_7', 'kurtosis BBEoff_8', 'kurtosis BBEoff_9', 'kurtosis BBEoff_10', 'kurtosis BBEoff_11', 'kurtosis BBEoff_12', 'kurtosis BBEoff_13', 'kurtosis BBEoff_14', 'kurtosis BBEoff_15', 'kurtosis BBEoff_16', 'kurtosis BBEoff_17', 'kurtosis BBEoff_18', 'kurtosis BBEoff_19', 'kurtosis BBEoff_20', 'kurtosis BBEoff_21', 'kurtosis BBEoff_22', 'kurtosis MFCCoff_1', 'kurtosis MFCCoff_2', 'kurtosis MFCCoff_3', 'kurtosis MFCCoff_4', 'kurtosis MFCCoff_5', 'kurtosis MFCCoff_6', 'kurtosis MFCCoff_7', 'kurtosis MFCCoff_8', 'kurtosis MFCCoff_9', 'kurtosis MFCCoff_10', 'kurtosis MFCCoff_11', 'kurtosis MFCCoff_12', 'kurtosis DMFCCoff_1', 'kurtosis DMFCCoff_2', 'kurtosis DMFCCoff_3', 'kurtosis DMFCCoff_4', 'kurtosis DMFCCoff_5', 'kurtosis DMFCCoff_6', 'kurtosis DMFCCoff_7', 'kurtosis DMFCCoff_8', 'kurtosis DMFCCoff_9', 'kurtosis DMFCCoff_10', 'kurtosis DMFCCoff_11', 'kurtosis DMFCCoff_12', 'kurtosis DDMFCCoff_1', 'kurtosis DDMFCCoff_2', 'kurtosis DDMFCCoff_3', 'kurtosis DDMFCCoff_4', 'kurtosis DDMFCCoff_5', 'kurtosis DDMFCCoff_6', 'kurtosis DDMFCCoff_7', 'kurtosis DDMFCCoff_8', 'kurtosis DDMFCCoff_9', 'kurtosis DDMFCCoff_10', 'kurtosis DDMFCCoff_11', 'kurtosis DDMFCCoff_12', 'kurtosis F1', 'kurtosis DF1', 'kurtosis DDF1', 'kurtosis F2', 'kurtosis DF2', 'kurtosis DDF2']```


## Prosody

- 103 features are computed:

### Features based on F0
                                
- 1-6     F0-contour                                                       Avg., Std., Max., Min., Skewness, Kurtosis

- 7-12    Tilt of a linear estimation of F0 for each voiced segment        Avg., Std., Max., Min., Skewness, Kurtosis

- 13-18   MSE of a linear estimation of F0 for each voiced segment         Avg., Std., Max., Min., Skewness, Kurtosis

- 19-24   F0 on the first voiced segment                                   Avg., Std., Max., Min., Skewness, Kurtosis

- 25-30   F0 on the last voiced segment                                    Avg., Std., Max., Min., Skewness, Kurtosis

### Features based on energy
- 31-34   energy-contour for voiced segments                               Avg., Std., Skewness, Kurtosis

- 35-38   Tilt of a linear estimation of energy contour for V segments     Avg., Std., Skewness, Kurtosis

- 39-42   MSE of a linear estimation of energy contour for V segment       Avg., Std., Skewness, Kurtosis

- 43-48   energy on the first voiced segment                               Avg., Std., Max., Min., Skewness, Kurtosis

- 49-54   energy on the last voiced segment                                Avg., Std., Max., Min., Skewness, Kurtosis

- 55-58   energy-contour for unvoiced segments                             Avg., Std., Skewness, Kurtosis

- 59-62   Tilt of a linear estimation of energy contour for U segments     Avg., Std., Skewness, Kurtosis

- 63-66   MSE of a linear estimation of energy contour for U segments      Avg., Std., Skewness, Kurtosis

- 67-72   energy on the first unvoiced segment                             Avg., Std., Max., Min., Skewness, Kurtosis

- 73-78   energy on the last unvoiced segment                              Avg., Std., Max., Min., Skewness, Kurtosis

### Features based on duration
                                
- 79      Voiced rate                                                      Number of voiced segments per second

- 80-85   Duration of Voiced                                               Avg., Std., Max., Min., Skewness, Kurtosis

- 86-91   Duration of Unvoiced                                             Avg., Std., Max., Min., Skewness, Kurtosis

- 92-97   Duration of Pauses                                               Avg., Std., Max., Min., Skewness, Kurtosis

- 98-103  Duration ratios                                                  Pause/(Voiced+Unvoiced), Pause/Unvoiced, Unvoiced/(Voiced+Unvoiced),
                                                                         Voiced/(Voiced+Unvoiced), Voiced/Puase, Unvoiced/Pause

```['F0avg', 'F0std', 'F0max', 'F0min', 'F0skew', 'F0kurt', 'F0tiltavg', 'F0mseavg', 'F0tiltstd', 'F0msestd', 'F0tiltmax', 'F0msemax', 'F0tiltmin', 'F0msemin', 'F0tiltskw', 'F0mseskw', 'F0tiltku', 'F0mseku', '1F0mean', '1F0std', '1F0max', '1F0min', '1F0skw', '1F0ku', 'lastF0avg', 'lastF0std', 'lastF0max', 'lastF0min', 'lastF0skw', 'lastF0ku', 'avgEvoiced', 'stdEvoiced', 'skwEvoiced', 'kurtosisEvoiced', 'avgtiltEvoiced', 'stdtiltEvoiced', 'skwtiltEvoiced', 'kurtosistiltEvoiced', 'avgmseEvoiced', 'stdmseEvoiced', 'skwmseEvoiced', 'kurtosismseEvoiced', 'avg1Evoiced', 'std1Evoiced', 'max1Evoiced', 'min1Evoiced', 'skw1Evoiced', 'kurtosis1Evoiced', 'avglastEvoiced', 'stdlastEvoiced', 'maxlastEvoiced', 'minlastEvoiced', 'skwlastEvoiced', 'kurtosislastEvoiced', 'avgEunvoiced', 'stdEunvoiced', 'skwEunvoiced', 'kurtosisEunvoiced', 'avgtiltEunvoiced', 'stdtiltEunvoiced', 'skwtiltEunvoiced', 'kurtosistiltEunvoiced', 'avgmseEunvoiced', 'stdmseEunvoiced', 'skwmseEunvoiced', 'kurtosismseEunvoiced', 'avg1Eunvoiced', 'std1Eunvoiced', 'max1Eunvoiced', 'min1Eunvoiced', 'skw1Eunvoiced', 'kurtosis1Eunvoiced', 'avglastEunvoiced', 'stdlastEunvoiced', 'maxlastEunvoiced', 'minlastEunvoiced', 'skwlastEunvoiced', 'kurtosislastEunvoiced', 'Vrate', 'avgdurvoiced', 'stddurvoiced', 'skwdurvoiced', 'kurtosisdurvoiced', 'maxdurvoiced', 'mindurvoiced', 'avgdurunvoiced', 'stddurunvoiced', 'skwdurunvoiced', 'kurtosisdurunvoiced', 'maxdurunvoiced', 'mindurunvoiced', 'avgdurpause', 'stddurpause', 'skwdurpause', 'kurtosisdurpause', 'maxdurpause', 'mindurpause', 'PVU', 'PU', 'UVU', 'VVU', 'VP', 'UP'] ```


