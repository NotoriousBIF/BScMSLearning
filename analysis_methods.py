import random
import math
from ms2deepscore import MS2DeepScore
from typing import List
from matchms import Spectrum
import pandas as pd
import numpy as np

def create_sets(master_subset: list, size_training: float, size_validation: float, size_testing: float):
    """this method randomly splits a subset into user-defined portions for training, validation and testing.
    master_subset: Subset which you want to split into separate parts
    size_training: Size of desired training set. Expected argument is percentage as float e.g. 0.8
    size_validation: Size of desired validation set. Same expectancies as size_training
    size_testing: Size of desired testing set. Same expectancies as size_training
    Returns: three lists of spectra, training_final, validation_final and testing_final"""
    sum_of_fractions = size_training + size_testing + size_validation
    assert (sum_of_fractions == 1,
            f"Sum of fractions is not equal to 1, equals {sum_of_fractions}. Please change sizes accordingly.")
    testing_amount = math.floor(size_testing*len(master_subset))
    validation_amount = math.floor(size_validation*len(master_subset))
    training_amount = math.floor(size_training*len(master_subset))
    index_list = [n for n in range(len(master_subset))]
    random.shuffle(index_list)
    testing_proposed = index_list[:testing_amount]
    validation_proposed = index_list[testing_amount:(testing_amount+validation_amount)]
    training_proposed = index_list[(testing_amount+validation_amount):]
    training_final = [master_subset[i] for i in training_proposed]
    validation_final = [master_subset[x] for x in validation_proposed]
    testing_final = [master_subset[y] for y in testing_proposed]
    return training_final, validation_final, testing_final

def select_predictions_for_test_spectra(tanimoto_df: pd.DataFrame,
                                        test_spectra: List[Spectrum]) -> np.ndarray:
    """Select the predictions for test_spectra from df with correct predictions

    tanimoto_df:
        Dataframe with as index and columns Inchikeys of 14 letters
    test_spectra:
        list of test spectra
    """
    inchikey_idx_test = np.zeros(len(test_spectra))
    for i, spec in enumerate(test_spectra):
        inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]

    inchikey_idx_test = inchikey_idx_test.astype("int")
    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_test[:], inchikey_idx_test[:])].copy()
    return scores_ref

def tanimoto_dependent_losses(scores, scores_ref, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).

    Parameters
    ----------

    scores
        Scores that should be evaluated
    scores_ref
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))
        idx = np.where((scores_ref >= low) & (scores_ref < high) & (~np.eye(scores_ref.shape[0], dtype=bool)))
        bin_content.append(idx[0].shape[0])
        maes.append(np.abs(scores_ref[idx] - scores[idx]).mean())
        rmses.append(np.sqrt(np.square(scores_ref[idx] - scores[idx]).mean()))
    return bin_content, bounds, rmses, maes

def calculate_bins(bin_amount):
    output = []
    for n in range(bin_amount+1):
        current_number = round(n/bin_amount, 2)
        output.append(current_number)
    return output

def generate_results(ms2deepscore_model, testspectra, tanimoto_df, amount_of_bins):
    """Combines earlier functions to easily compute results for a ms2deepscore model on a given testset
     ms2deepscore_model: the model you want to use for predictions
     testspectra: the spectra you want to test performance on
     tanimoto_df: the dataframa containing the true scores
     amount_of_bins: the amount of bins you want to distribute the errors over
     returns: list with the output from tanimoto_independent_losses and the average RMSE"""
    bins = calculate_bins(amount_of_bins)
    similarity_measure = MS2DeepScore(ms2deepscore_model)
    predicted_scores = similarity_measure.matrix(testspectra[:], testspectra[:], is_symmetric=True)
    true_scores = select_predictions_for_test_spectra(tanimoto_df, testspectra)
    bin_content, bounds, rmses, maes = tanimoto_dependent_losses(predicted_scores, true_scores, bins)
    global_rmse = sum(rmses)/len(rmses)
    output = [bin_content, bounds, rmses, maes, global_rmse]
    return output

def generate_error_bars(rmses:list, bin_sizes:list):
    lower_bounds = []
    upper_bounds = []
    for i in range(len(rmses)):
        current_lb = rmses[i] * (1 - np.sqrt((1 - ((1.96*2**0.5)/(np.sqrt(bin_sizes[i] - 1))))))
        current_ub = rmses[i] * (np.sqrt(1+((1.96*2**0.5)/np.sqrt(bin_sizes[i] - 1))) - 1)
        lower_bounds.append(current_lb)
        upper_bounds.append(current_ub)
    return lower_bounds, upper_bounds

##uitzoekscripts
from collections import Counter
q_testing = pd.read_pickle("C:/Users/remco/Documents/Thesis/Datafiles/quadrupole_testingset.pickle")
q_training = pd.read_pickle("C:/Users/remco/Documents/Thesis/Datafiles/quadrupole_trainingset.pickle")
q_validation = pd.read_pickle("C:/Users/remco/Documents/Thesis/Datafiles/quadrupole_validationset.pickle")
q_trainingphase = q_training + q_validation
q_complete = q_testing + q_training + q_validation
inchis = [d.get("inchikey") for d in q_complete]
inchikey_occurrences = Counter(inchis)

peaks = [d.peaks.intensities.size for d in q_trainingphase]
average_peak_per_spectrum = sum(peaks)/len(peaks)
peak_occurrences = Counter(peaks)

