from analysis_methods import generate_results, select_predictions_for_test_spectra, calculate_bins, tanimoto_dependent_losses
import pandas as pd
from ms2deepscore.models import load_model
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

tanimoto_scores = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/Tanimoto scores/GNPS_15_12_2021_pos_tanimoto_scores.pickle")
bin_amount = 10
generic_model = load_model("G:/Remco Bsc Thesis/Models/ms2deepscore_model_generic.hdf5")

#generic vs orbitrap
orbitrap_testing = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/Subsets/Orbitrap model v1 sets/orbitrap_testingset.pickle")
orbitrap_model = load_model("G:/Remco Bsc Thesis/Models/ms2deepscore_model_orbitrap.hdf5")
orbitrap_results = generate_results(orbitrap_model, orbitrap_testing, tanimoto_scores, bin_amount)
generic_orbitrap_results = generate_results(generic_model, orbitrap_testing, tanimoto_scores, bin_amount)
#generic vs Fourier Transform
ft_testing = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/ft_testingset.pickle")
ft_model = load_model("G:/Remco Bsc Thesis/Models/ms2deepscore_model_ft.hdf5")
ft_results = generate_results(ft_model, ft_testing, tanimoto_scores, bin_amount)
generic_ft_results = generate_results(generic_model, ft_testing, tanimoto_scores, bin_amount)
#generic vs TOF
tof_testing = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/TOF_testingset.pickle")
tof_model = load_model("G:/Remco Bsc Thesis/Models/ms2deepscore_model_TOF.hdf5")
tof_results = generate_results(tof_model, tof_testing, tanimoto_scores, bin_amount)
generic_tof_results = generate_results(generic_model, tof_testing, tanimoto_scores, bin_amount)
#generic vs quadrupole
quadrupole_testing = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/quadrupole_testingset.pickle")
quadrupole_model = load_model("G:/Remco Bsc Thesis/Models/ms2deepscore_model_quadrupole.hdf5")
quadrupole_results = generate_results(quadrupole_model, quadrupole_testing, tanimoto_scores, bin_amount)
generic_quadrupole_results = generate_results(generic_model, quadrupole_testing, tanimoto_scores, bin_amount)

#plots
def gather_results_for_plots():
    generic_results = []
    specific_results = []
    return generic_results, specific_results

def plot_barplot(generic_global_rmse:list, specific_global_rmse:list):
    fig = plt.subplots(figsize=(12, 8))
    barWidth = 0.25
    y_generic = generic_global_rmse
    y_specific = specific_global_rmse
    generic_bars = np.arange(len(y_generic))
    specific_bars = [x+barWidth for x in generic_bars]
    plt.bar(generic_bars, y_generic, color="green", width=barWidth, label="Generic")
    plt.bar(specific_bars, y_specific, color="blue", width=barWidth, label="Specific")
    plt.ylabel("Global RMSE", fontsize=15)
    plt.xlabel("Specific testingset and model", fontsize=15)
    plt.xticks(ticks=[r+barWidth for r in range(len(y_generic))], labels=["Orbitrap", "Quadrupole", "TOF", "Fourier Transform"])
    plt.title(label="Global RMSE of Generic model vs global RMSE of specific models")
    plt.legend(loc="upper right")
    output = plt.show()
    return output

def plot_errorbars_rmse(generic_rmses:list, specific_rmses:list, figure_name:str):
    generic_standard_deviation = [n**2 for n in generic_rmses]
    specific_standard_deviation = [n**2 for n in specific_rmses]
    x_labels = ["-inf < 0.1", "0.1 < 0.2", "0.2 < 0.3", "0.3 < 0.3", "0.4 < 0.5", "0.5 < 0.6", "0.6 < 0.7", "0.7 < 0.8",
                "0.8 < 0.9", "0.9 < inf"]
    plt.errorbar(x_labels, generic_rmses, yerr=generic_standard_deviation, label="Generic")
    plt.errorbar(x_labels, specific_rmses, yerr=specific_standard_deviation, label="Specific")
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(loc="upper right")
    plt.title(figure_name)
    output = plt.show()
    return output

def plot_scatter_rmse(generic_rmses:list, specific_rmses:list, figure_name:str):
    x_labels = ["-inf < 0.1", "0.1 < 0.2", "0.2 < 0.3", "0.3 < 0.3", "0.4 < 0.5", "0.5 < 0.6", "0.6 < 0.7", "0.7 < 0.8",
                "0.8 < 0.9", "0.9 < inf"]
    plt.scatter(x_labels, generic_rmses, c="b", label="Generic")
    plt.plot(x_labels, generic_rmses)
    plt.scatter(x_labels, specific_rmses, c="y", label="Specific")
    plt.plot(x_labels, specific_rmses)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xlabel('Tanimoto scoring bins')
    plt.ylabel('RSME')
    plt.legend(loc="upper right")
    plt.title(figure_name)
    output = plt.show()
    return output

def plot_scatter_similarities_with_bins(results:list, figure_name:str):
    y_values = results[0]
    x_labels = ["-inf < 0.1", "0.1 < 0.2", "0.2 < 0.3", "0.3 < 0.3", "0.4 < 0.5", "0.5 < 0.6", "0.6 < 0.7", "0.7 < 0.8",
                "0.8 < 0.9", "0.9 < inf"]
    plt.scatter(x_labels, y_values)
    plt.plot(x_labels, y_values)
    plt.ylabel('# of spectral pairs')
    plt.xlabel('Tanimoto scoring bins')
    plt.title(figure_name)
    output = plt.show()
    return output

