from analysis_methods import *
from methods import *

all_spectra = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/ALL_GNPS_15_12_2021_positive_annotated.pickle')
occurrences_list = count_instrument_types(all_spectra)
orbitrap_regexp_terms = re.compile(".*orbitrap.*|.*hcd.*|.*q-exactive.*|.*lumos.*|.*velos.*", re.IGNORECASE)
instrument_entries, occurrences = unzip_types_occurrences(occurrences_list)
orbitrap_matches = create_keyword_list(orbitrap_regexp_terms, instrument_entries)
instrument_aliases = {}
instrument_aliases["Orbitrap"] = orbitrap_matches

orbitrap_only = instrument_filter(instrument_aliases, all_spectra)
orbitrap_spectra = subset_creation(orbitrap_only, 'Orbitrap')

orbitrap_training, orbitrap_validation, orbitrap_testing = create_sets(orbitrap_spectra, 0.8, 0.1, 0.1)

export_as_pickle(orbitrap_training, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_trainingset')
export_as_pickle(orbitrap_validation, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_validationset')
export_as_pickle(orbitrap_testing, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_testingset')


