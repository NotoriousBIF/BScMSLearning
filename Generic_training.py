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

#importing all of the training sets and tanimoto df
tanimoto_scores_df = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/Tanimoto scores/GNPS_15_12_2021_pos_tanimoto_scores.pickle')
ft_training = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/ft_trainingset.pickle')
qqq_training = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/quadrupole_trainingset.pickle')
orbitrap_training = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/Subsets/Orbitrap model v1 sets/orbitrap_trainingset.pickle')
tof_training = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/tof_trainingset.pickle')

#importing all of the validation sets
ft_validation = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/ft_validationset.pickle')
qqq_validation = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/quadrupole_validationset.pickle')
orbitrap_validation = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/Subsets/Orbitrap model v1 sets/orbitrap_validationset.pickle')
tof_validation = pd.read_pickle('G:/Remco Bsc Thesis/Datafiles/tof_validationset.pickle')

#adding all of the sets together
generic_training = ft_training + qqq_training + orbitrap_training + tof_training
generic_validation = ft_validation + qqq_validation + orbitrap_validation + tof_validation

#saving these for later use
export_as_pickle(generic_training, 'G:/Remco Bsc Thesis/Datafiles', 'generic_trainingset')
export_as_pickle(generic_validation, 'G:/Remco Bsc Thesis/Datafiles', 'generic_validationset')

spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100.0)
binned_spectrums_training = spectrum_binner.fit_transform(generic_training)
binned_spectrums_val = spectrum_binner.transform(generic_validation)

training_inchikeys = np.unique([s.get("inchikey")[:14] for s in generic_training])

same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
dimension = len(spectrum_binner.known_bins)
training_generator = DataGeneratorAllInchikeys(
    binned_spectrums_training, training_inchikeys, tanimoto_scores_df, dim=dimension,
    same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)

validation_inchikeys = np.unique([s.get("inchikey")[:14] for s in generic_validation])
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
model_file_name = os.path.join(model_path, "ms2deepscore_model_generic.hdf5")
model.save(model_file_name)

from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
