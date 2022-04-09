import random
import math
from ms2deepscore import MS2DeepScore


def create_sets(master_subset: list, size_training: float, size_validation: float, size_testing: float):
    """this method randomly splits a subset into user-defined portions for training, validation and testing.
    master_subset: Subset which you want to split into separate parts
    size_training: Size of desired training set. Expected argument is percentage as float e.g. 0.8
    size_validation: Size of desired validation set. Same expectancies as size_training
    size_testing: Size of desired testing set. Same expectancies as size_training
    Returns: three lists of spectra, training_final, validation_final and testing_final"""
    testing_amount = math.floor(size_testing*len(master_subset))
    validation_amount = math.floor(size_validation*len(master_subset))
    training_amount = math.floor(size_training*len(master_subset))
    spectrum_ids = [d.get('spectrumid') for d in master_subset]
    testing_proposed = random.sample(spectrum_ids, testing_amount)
    validation_and_training = list(set(spectrum_ids) - set(testing_proposed))
    validation_proposed = random.sample(validation_and_training, validation_amount)
    training_proposed = list(set(validation_and_training) - set(validation_proposed))
    training_final = list(filter(lambda x: x.get('spectrumid') in training_proposed, master_subset))
    validation_final = list(filter(lambda x: x.get('spectrumid') in validation_proposed, master_subset))
    testing_final = list(filter(lambda x: x.get('spectrumid') in testing_proposed, master_subset))
    return training_final, validation_final, testing_final


def create_pairs(testingset: list):
    """This function turns a list into a new list of half its size, but with each entry being randomly paired and each
    spectrum being used only ONCE for a pairing.
    testingset: The testingset that you want to be turned into a list of pairs
    Returns: a list of randomly paired spectra"""
    outputlist = []
    currentset = testingset.copy()
    while len(currentset) != 0:
        pair = random.sample(currentset, 2)
        p1 = pair[0]
        p2 = pair[1]
        outputlist.append(pair)
        currentset.remove(p1)
        currentset.remove(p2)
    return outputlist

def generate_scores_pairs(list_of_pairs: list, ms2deepscore_model, tanimoto_df):
    """"This function uses a previously trained and validated MS2DeepScore model to generate similarity measures.
    list_of_pairs: This is your testingset, randomly split into a list of paired spectra
    ms2deepscore_model: The MS2DeepScore model that will predict the similarity measures
    tanimoto_df: A dataframe containing the correct similarity measures"""
    similarity_measure = MS2DeepScore(ms2deepscore_model)
    scorelist = []
    for pair in range(len(list_of_pairs)):
        current_pair = list_of_pairs[pair]
        score = similarity_measure.pair(current_pair[0], current_pair[1])
        spectrum_id1 = current_pair[0].get('spectrumid')
        spectrum_id2 = current_pair[1].get('spectrumid')
        inchikey1 = current_pair[0].get('inchikey')
        correctinchi1 = inchikey1[0:14]
        inchikey2 = current_pair[1].get('inchikey')
        correctinchi2 = inchikey2[0:14]
        true_score = tanimoto_df[correctinchi1].get(correctinchi2)
        final = (spectrum_id1, spectrum_id2, score, true_score)
        scorelist.append(final)
    return scorelist

def generate_scores_matrix(testingset: list, ms2deepscore_model, tanimoto_df):
    """This function uses a previously trained and validated MS2DeepScore model to generate similarity measures.
    testingset: a set of spectra that you wish to predict similarity measures for
    ms2deepscore_model: The MS2DeepScore model that will predict these similarity measures
    tanimoto_df: A dataframe containing the correct similarity measures
    Returns: A matrix with each field consisting of a list, in format [INCHI Spectrum 1, INCHI Spectrum 2,
    Predicted score, true score, deviation]. Column and row names will be the spectrum IDs."""
    similarity_measure = MS2DeepScore(ms2deepscore_model)
    predicted_score_matrix = similarity_measure.matrix(testingset[:], testingset[:])
    final_score_matrix = {}
    spectrum_ids = [d.get('spectrumid') for d in testingset]
    #this is the part where we populate the output matrix
    for id in spectrum_ids:
        print(id)
        rows = {}
        for s in spectrum_ids:
            current_spectrum_1 = spectrum_ids.index(id)
            current_spectrum_2 = spectrum_ids.index(s)
            current_predicted_score = predicted_score_matrix[current_spectrum_1][current_spectrum_2]
            current_inchikey_1 = (testingset[current_spectrum_1].get("inchikey"))[0:14]
            current_inchikey_2 = (testingset[current_spectrum_2].get('inchikey'))[0:14]
            current_true_score = tanimoto_df[current_inchikey_1].get(current_inchikey_2)
            deviation = current_predicted_score - current_true_score
            entry = [current_inchikey_1, current_inchikey_2, current_predicted_score, current_true_score, deviation]
            rows.update({s: entry})
        final_score_matrix[id] = rows
    return final_score_matrix


def calculate_errors_pairs(list_of_scores):
    """This function takes the output from generate_scores_pairs and calculates the difference between the predicted score and
    the true score.
    list_of_scores: a list of predicted and calculated scores in pairs
    returns: a list of residuals"""
    output_list = []
    for entry in range(len(list_of_scores)):
        current_entry = list_of_scores[entry]
        predicted_score = current_entry[3]
        true_score = current_entry[2]
        error = true_score - predicted_score
        output_list.append(error)
    return output_list

def extract_errors_from_matrix(score_matrix):
    """This function calculates the difference between the predicted and the true score.
     score_matrix: the output from generate_scores_matrix
     returns: a list of residuals."""
    output_list = []
    for spectrumid1, spectrumid2 in score_matrix:
        current_score = spectrumid2[4]
        output_list.append(current_score)
    return output_list

