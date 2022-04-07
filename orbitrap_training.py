from analysis_methods import *
from methods import *
from ms2deepscore import SpectrumBinner
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore.models import load_model

#loading data
tanimoto_scores_df = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/Tanimoto scores/GNPS_15_12_2021_pos_tanimoto_scores.pickle')
all_spectra = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/ALL_GNPS_15_12_2021_positive_annotated.pickle')

#filtering out orbitrap spectra
occurrences_list = count_instrument_types(all_spectra)
orbitrap_regexp_terms = re.compile(".*orbitrap.*|.*hcd.*|.*q-exactive.*|.*lumos.*|.*velos.*", re.IGNORECASE)
instrument_entries, occurrences = unzip_types_occurrences(occurrences_list)
orbitrap_matches = create_keyword_list(orbitrap_regexp_terms, instrument_entries)
instrument_aliases = {}
instrument_aliases["Orbitrap"] = orbitrap_matches

orbitrap_only = instrument_filter(instrument_aliases, all_spectra)
orbitrap_spectra = subset_creation(orbitrap_only, 'Orbitrap')

sorted_random_subsets = create_sets(orbitrap_spectra, 0.8, 0.1, 0.1)
orbitrap_training = subset_creation(sorted_random_subsets, 'training')
orbitrap_validation = subset_creation(sorted_random_subsets, 'validation')
orbitrap_testing = subset_creation(sorted_random_subsets, 'testing')

#saving our subsets for later use
export_as_pickle(orbitrap_training, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_trainingset')
export_as_pickle(orbitrap_validation, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_validationset')
export_as_pickle(orbitrap_testing, 'G:/Remco Bsc Thesis/Datafiles', 'orbitrap_testingset')

#binning
spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100.0)
binned_spectrums_training = spectrum_binner.fit_transform(orbitrap_training)
binned_spectrums_val = spectrum_binner.transform(orbitrap_validation)

training_inchikeys = np.unique([s.get("inchikey")[:14] for s in orbitrap_training])

same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
dimension = len(spectrum_binner.known_bins)
training_generator = DataGeneratorAllInchikeys(
    binned_spectrums_training, training_inchikeys, tanimoto_scores_df, dim=dimension,
    same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)

validation_inchikeys = np.unique([s.get("inchikey")[:14] for s in orbitrap_validation])
validation_generator = DataGeneratorAllInchikeys(
    binned_spectrums_val, validation_inchikeys, tanimoto_scores_df, dim=dimension, same_prob_bins=same_prob_bins,
    num_turns=10, augment_removal_max=0, augment_removal_intensity=0,
    augment_intensity=0, augment_noise_max=0, use_fixed_set=True)

model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                         dropout_rate=0.2)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

# Save best model and include earlystopping
earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1, restore_best_weights=True)
model.summary()

history = model.model.fit(training_generator, validation_data=validation_generator,
                          epochs=150, verbose=1, callbacks=[earlystopper_scoring_net])

model_path = 'G:/Remco Bsc Thesis/Models/'
model_file_name = os.path.join(model_path, "ms2deepscore_model_orbitrap.hdf5")
model.save(model_file_name)

from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




