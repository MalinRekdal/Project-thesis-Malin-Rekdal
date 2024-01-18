"""
This python file contains functions used in multiple files the rest of the project (mostly to extract the right data), and functions to evaluate a model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from constants import class_labels

# Functions used that is reused in different files: 

def get_health_state(value, dict=class_labels):
    """
    Function to get health state from number 0 or 1 using the label mapping dict in the opposite way. 
    
    The functionality to get the key from the value from a dict is taken from: 
    https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/

    Args:
        value (int): The value 0 or 1 we want to get the health state for. 
        dict (dictionary, optional): The mapping between 0 and 1 and health states. (default to class_labels). Defaults to class_labels.

    Returns:
        string: the string showing if it is HC or PD (the health state). 
    """
    key = list(filter(lambda x: dict[x] == value, dict))[0]
    return key


def extend_paths(path_list, base_path):
    """
    Extends all elements in path_list with the base_path first

    Args:
        path_list (list): list of paths 
        base_path (str): str of base path

    Returns:
        list: new list with all paths extended with the base path
    """
    for i in range(len(path_list)):
        if base_path not in path_list[i]:
            path_list[i] = base_path + path_list[i]
    return path_list
    

def find_certain_paths_from_all_paths(paths, d_type):
    """
    Takes inn a list of paths and a data types parameter and 
    returns all paths that contain that data types.
    If none of d_type is in the paths then it returns an warning about that.  

    Args:
        paths (list): list of paths we want to reduce
        d_type (list): list of data types we want to keep

    Returns:
        list: new list of paths that only contain data type. 
    """
    res = [elem for elem in paths if any(term in elem for term in d_type)]
    if res != []:
        return res
    else:
        print("Could not find ", d_type)
        return False
    
    
def get_features_from_one_path(path, choosen_f):
    """
    Get the choosen features according to a path. 

    Args:
        path (string): Path to where we have files to get data from. 
        choosen_f (list): List of the features we want to extract

    Returns:
        df: df with features for all cases in that path
    """
    feat = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(path):
        if filenames != []:  # Avoids folder without filenames
            for filename in filenames:
                f = choosen_f[filename.split(".csv")[0]]
                data = pd.read_csv(os.path.join(dirpath, filename), usecols=f)
                feat = pd.concat([feat, data], axis=1)
    return feat


def get_features(paths, choosen_f):
    """
    Finds all choosen features for all paths and returns them as a df. 
    
    Also: In case we have multiple id's with us, we drop the duplicates.
        (Reference to method used for this: https://sparkbyexamples.com/python/pandas-remove-duplicate-columns-from-dataframe/)

    Args:
        paths (list): List of all paths to where we have files to get data from. 
        choosen_f (list): List of the features we want to extract

    Returns:
        df: DataFrame with the wanted features from the paths
    """
    feat = pd.DataFrame()
    for one_path in paths:
        new_features = get_features_from_one_path(one_path, choosen_f)
        feat = pd.concat([feat, new_features], axis=0).reset_index(drop=True)
    return feat.loc[:,~feat.columns.duplicated()] 



def add_metadata_columns(data, metadata, metadata_columns):
    """
    Takes in a df of features and adds columns corresponding to the metadata we want to add. 

    Args:
        data (df): DataFrame to add metadata to
        metadata_columns (list): List of the metadata types we want to add

    Returns:
        df: data with added metadata
    """
    for elem in metadata_columns:
        data[elem] = ""
    for index, row in data.iterrows():
        id = row["id"].split("_")[0]
        # Add metadata info: 
        for elem in metadata_columns:
            data.at[index, elem] = metadata.loc[metadata['RECODING ORIGINAL NAME'] == id, elem].iloc[0]
    return data



########## FEATURE PLOTTING ##################


def plot_features(f, names, colors, metadata_columns, gaussian = True):
    """
    Plots given features and finds average between all people. 
    Compares features from two different types. For example female vs male or HC vs PD. 
    Both plots feature values vs person number, and a gaussian distribution of these features. 
    
    Note that the amount of features showing on the x axis is only one of the 2 data types (HC vs PD or female vs male),
    so in total we are looking at twice the amount of people. 
    
    Note: len(f) = 2, and every time we iterate through those we look at the 2 data types separately. 

    Args:
        f (df): DataFrame of features to be plotted. Contains data for both data types. 
        names (list): Names of the data types we will be plotting. For example female and male or HC and PD. 
        colors (list): Colors that will be used for the plotting for the 2 data types. 
        metadata_columns (list): list of the types of metadata we have
        gaussian (bool, optional): True or False depending on if you want the gaussian plot as well or not. Defaults to True.
    """
    # Remove metadata columns before plotting all features
    for i in range(len(f)):
        f[i] = f[i].drop(columns=metadata_columns)
    
    feature_types = f[0].keys().to_list()
    f = [np.array(f[0]), np.array(f[1])]
    num_row = [len(f[0]), len(f[1])]

    for i in range(len(feature_types)):
        
        avg = [np.average(f[0][:,i]), np.average(f[1][:,i])]
        std = [np.std(f[0][:,i]), np.std(f[1][:,i])]
        
        print(" ")
        print(f"The average value and std for {feature_types[i]} is: ")
        for j in range(len(f)):
            print(f"   Over the {num_row[j]} {names[j]} the average was {round(avg[j], 2)} and the standard deviation was {round(std[j], 2)}")

        x = [list(range(1, num_row[0] + 1)), list(range(1, num_row[1] + 1))]

        # Plot of features 
        for j in range(len(f)):
            plt.scatter(x[j], f[j][:, i], color=colors[j], label=names[j])
            plt.axhline(y=avg[j], color=colors[j], linestyle='--', label='Avg for ' + names[j])
            plt.axhline(y=avg[j] + std[j], color=colors[j], linestyle=':', label='Std for '+ names[j])
            plt.axhline(y=avg[j] - std[j], color=colors[j], linestyle=':')
            # plt.text(-200, avg[j], np.round(avg[j]), color=colors[j], verticalalignment='bottom') # To write the average value in the plot. 

        plt.plot()
        plt.title(feature_types[i] + ' for different '+ names[0] + ' and ' + names[1], fontsize=18)
        plt.xlabel('Sample nr', fontsize=16)
        plt.ylabel(feature_types[i], fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc='upper right', fontsize=14)
        plt.show()
        
        # Note that it is not necessary Gaussian distributed, but assumed it is we can plot the distributions: 
        if gaussian: # Optional to not plot this by defining gaussian = False
            x_range_0 = np.linspace(avg[0] - 3 * std[0], avg[0] + 3 * std[0], 100)
            x_range_1 = np.linspace(avg[1] - 3 * std[1], avg[1] + 3 * std[1], 100)
            plt.plot(x_range_0, norm.pdf(x_range_0, avg[0], std[0]), color=colors[0], linestyle='--', label="Distribution for " + names[0])
            plt.plot(x_range_1, norm.pdf(x_range_1, avg[1], std[1]), color=colors[1], label="Distribution for " + names[1])
            plt.title('Gaussian distributions for feature: ' + feature_types[i], fontsize=18)
            plt.xlabel(feature_types[i] + " values", fontsize=16)
            plt.ylabel('Probability density', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(loc='upper right', fontsize=14)
            plt.show()


def plot_features_together(f, f_types):
    """
    Plots the 2 features defined in f_types together with the values given from 
    the features list f. Usefull to see how the features work together. 
    
    Note: If both features have 0 as value for the same people then it wont show that there is a lot of 0's,
    but if only one of them has a lot of 0's it will be a clear line of scatters along the 0 line of that feature. 

    Args:
        f (df): DataFrame of all features
        f_types (2 element long list): features to be plotted together
    """
    x = f_types[0]
    y = f_types[1]
    
    plt.scatter(np.array(f["HC"][x]), np.array(f["HC"][y]), color="darkgreen", label="HC")
    plt.scatter(np.array(f["PD"][x]), np.array(f["PD"][y]), color="firebrick", label="PD")

    plt.plot()
    plt.title(f"{x} and {y} for different HC and PD people", fontsize=18)
    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=14)
    plt.show()     



########## EVALUATION FUNCTIONS ##################

def plot_histogram(data, xlabel, title, color='skyblue', bins=False):
    """
    Plots histogram for data. 

    Args:
        data (list): list of data to plot histogram fore
        xlabel (str): x label
        title (str): title
        color (str, optional): Color to plot histogram in. Defaults to 'skyblue'.
        bins (bool, optional): bins we want the histogram to be devided into. Defaults to False, and if False then plt.hist uses its natural choice.
    """
    if bins:
        data.plot(kind='hist', color=color, edgecolor='black', bins = bins)
    else: 
        data.plot(kind='hist', color=color, edgecolor='black')
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Number of people', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculates and prints confusion matrix for both number of instances and percentage, and
    plots confusion matrix for number of instances. Calculates from true and predicted labels. 

    Args:
        y_pred (list): list of predicted labels
        test_labels (list): list of true labels
    """
    # Generate the confusion matrix with labels
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(class_labels.values()))

    # Display the confusion matrix: 
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=list(class_labels.keys()))
    disp.plot()
    plt.title("Confusion matrix")
    plt.show()
    
    
def print_classification_report(y_true, y_pred):
    """
    Creates and prints out more detailed classification report from true and predicted labels. 

    Args:
        test_labels (list): list of true labels
        y_pred (list): list of predicted labels
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(class_labels.keys())))
    

def sensitivity_and_specificity(y_true, y_pred, write_out=True):
    """
    Uses true labels and predicted labels to find sensitivity and specificity.
    Formulas gotten from: https://www.analyticsvidhya.com/blog/2021/06/classification-problem-relation-between-sensitivity-specificity-and-accuracy/ 

    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        write_out (bool, optional): Variable to decide if you want sensitivity and specificity values printed or not. Defaults to True.

    """
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(class_labels.values()))
    TN = confusion_mat[0, 0] # True Negative (True: HC, Pred: HC)
    TP = confusion_mat[1, 1] # True Positive (True: PD, Pred: PD)
    FN = confusion_mat[1, 0] # False Negative (True: PD, Pred: HC)
    FP = confusion_mat[0, 1] # False Positive (True: HC, Pred: PD)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    if write_out:
        print(f"Sensitivity (Recall of PD): {sensitivity* 100:.2f}%")
        print(f"Specificity (Recall of HC): {specificity* 100:.2f}%")
    return sensitivity, specificity


def autolabel(rects, ax):
    """
    Function that ads numbers above the bars in the gender distribution plot
    Note: This function is created with generative AI (Chat GPT) 
    Args:
        rects (bars): The bars for male and female for either correct or wrongly classified
        ax (ax): the axis
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom')


def plot_gender_distribution(gender_counts):
    """
    Plots gender distribution from meta data, used to plot amount of female and male that is correctly and wrongly classified for PD vs HC. 
    Note: This function is created with generative AI (Chat GPT) and then modified
    Args:
        gender_counts (list): List containing counts of [num_correct_f, num_wrong_f], [num_correct_m, num_wrong_m]
    """
    gender_counts = np.array(gender_counts)
    genders = ['Female', 'Male']
    
    fig, ax = plt.subplots()
    bar_width = 0.4
    indices = np.arange(len(genders))

    rects1 = ax.bar(indices + 0 * bar_width, gender_counts[:, 0], width=bar_width, color="skyblue", label='Correct classified', alpha=0.7, edgecolor='black')
    autolabel(rects1, ax)
    rects2 = ax.bar(indices + 1 * bar_width, gender_counts[:, 1], width=bar_width, color="salmon", label='Wrong classified', alpha=0.7, edgecolor='black')
    autolabel(rects2, ax)

    plt.xticks(indices + bar_width / 2, genders, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Distribution of gender in Correct and Wrong Classifications', fontsize=14)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Number of people', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def evaluate_metadata(test_labels, y_pred, test_metadata, metadata_columns):
    """
    Look into the correct and wrongly classified people with regards to the metadata. 

    Args:
        test_labels (list): list of true labels
        y_pred (list): list of predicted labels
        test_metadata (list): list of corresponding metadata to the true and predicted labels
        metadata_columns (list): list of the types of metadata we have
    """

    correct_metadata = []
    wrong_metadata = []
    for i, (true, pred) in enumerate(zip(test_labels, y_pred)):
        if true == pred:
            correct_metadata.append(test_metadata[i])
        else:
            wrong_metadata.append(test_metadata[i])

    # Can also use correct and wrong metadata here to get the id's and investigate the waveforms and features of correct and wrongly classified data. 
    correct_metadata = pd.DataFrame(correct_metadata, columns=metadata_columns)
    wrong_metadata = pd.DataFrame(wrong_metadata, columns=metadata_columns)

    # Stats on gender
    correct_women = correct_metadata[correct_metadata['SEX'] == 'F']
    correct_men = correct_metadata[correct_metadata['SEX'] == 'M']
    print(f"We have {len(correct_women)} women and {len(correct_men)} men in the {len(correct_metadata)} correctly classified people. ")

    wrong_women = wrong_metadata[wrong_metadata['SEX'] == 'F']
    wrong_men = wrong_metadata[wrong_metadata['SEX'] == 'M']
    print(f"We have {len(wrong_women)} women and {len(wrong_men)} men in the {len(wrong_metadata)} wrongly classified people. ")

    plot_gender_distribution([[len(correct_women), len(wrong_women)], [len(correct_men), len(wrong_men)]])

    # Stats on age
    plot_histogram(wrong_metadata["AGE"], "Age", "Distribution of age among wrongly classified people")
    plot_histogram(correct_metadata["AGE"], "Age", "Distribution of age among correctly classified people")

    # Possibility to add stats on other metadata for PD patients classified as HC


def shuffle_data(x, y):
    """Functions that takes in data and labels and shuffles them in the same way. 
    Used to make sure the new data have a random set up of HC and PD after each other. 

    Args:
        x (list): data
        y (list): labels
    """
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    shuffled_x = x[indices]
    shuffled_y = y[indices]

    return shuffled_x, shuffled_y



def x_fold_cross_val(x, y, model, num_folds = 5, random_state=42, write_out=True):
    """
    Does a x fold cross validation using data and the labels and already defined model. 
    The type of splitting used makes it so that all of the values are test values at some point, 
    so for 5 folds we have a 80-20 split. 
    
    Can also change num_folds to be something else if we want another amount of folds. 
    
    Args:
        x (list): all data
        y (list): all labels
        model (model): The trained model we want to predict from.
        num_folds (int, optional): Number of folds wanted. Defaults to 5.
        random_state (int, optional): Random state. Defaults to 42.
        write_out (bool, optional): Variable to decide if you want fold values and conf matrix printed or not. Defaults to True.
    """

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state) # Initialize StratifiedKFold

    # List to store accuracy, sensitivity and specificity scores for each fold
    accuracy_scores = []  
    sensitivity_scores = []
    specificity_scores = []
    confusion_mat_sum = np.array([[0, 0], [0, 0]]) # To store confusion matrix sum from all folds
    fold_num = 0

    for train_index, test_index in skf.split(x, y):  # Iterate through folds 
        fold_num += 1
        if write_out:
            print(" ")
            print(f"This is data for fold number {fold_num}: ")

        # Split the data into train and test sets for this fold
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # The data is first only 0's and then only 1's so to get it more random it gets shuffled first. 
        x_train, y_train = shuffle_data(x_train, y_train)
        x_test, y_test = shuffle_data(x_test, y_test)
        
        # Standardize the data
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit(x_train, y_train)  # Train the model
        y_pred = model.predict(x_test) # Evaluate the model on the test set
        test_accuracy = accuracy_score(y_test, y_pred)
        
            
        # Append accuracy scores
        accuracy_scores.append(test_accuracy)
        
        # Generate the confusion matrix with labels
        confusion_mat = confusion_matrix(y_test, y_pred, labels=list(class_labels.values()))
        confusion_mat_sum += confusion_mat     
        
        sensitivity, specificity = sensitivity_and_specificity(y_true=y_test, y_pred = y_pred, write_out=False)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)  
        
        # Print result for fold: 
        if write_out:
            print(f"Fold Test Accuracy: {test_accuracy * 100:.2f}%")
            
            print("Confusion Matrix:")
            print(confusion_mat)
    
    # Display average accuracy across all folds
    print(f"Results over all {fold_num} folds:")
    rounded_accuracy_scores = [round(elem * 100, 2) for elem in accuracy_scores]
    print(" ")
    print(f"Accuracy over all {fold_num} folds: {rounded_accuracy_scores}")
    print(f"Average Accuracy: {np.mean(accuracy_scores) * 100:.1f}%")
    print(f"The accuracy is variating between {np.min(rounded_accuracy_scores)}% and {np.max(rounded_accuracy_scores)}%")
    print(f"The std of the accuracy: {np.std(accuracy_scores) * 100:.1f}%")
    
    rounded_sensitivity_scores = [round(elem * 100, 2) for elem in sensitivity_scores]
    print(" ")
    print(f"Sensitivity over all {fold_num} folds: {rounded_sensitivity_scores}")
    print(f"Average Sensitivity: {np.mean(sensitivity_scores) * 100:.1f}%")
    print(f"The Sensitivity is variating between {np.min(rounded_sensitivity_scores)}% and {np.max(rounded_sensitivity_scores)}%")
    
    rounded_specificity_scores = [round(elem * 100, 2) for elem in specificity_scores]
    print(" ")
    print(f"Specificity over all {fold_num} folds: {rounded_specificity_scores}")
    print(f"Average Specificity: {np.mean(specificity_scores) * 100:.1f}%")
    print(f"The Specificity is variating between {np.min(rounded_specificity_scores)}% and {np.max(rounded_specificity_scores)}%")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_sum, display_labels=list(class_labels.keys()))
    disp.plot()
    plt.title(f"Confusion matrix sum over all {fold_num} folds:")
    plt.show()
