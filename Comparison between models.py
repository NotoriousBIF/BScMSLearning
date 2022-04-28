from analysis_methods import *
import pandas as pd
from ms2deepscore.models import load_model

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
