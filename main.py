##setting up##
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter
import re
from matchms.filtering import default_filters
import os
import gensim

path_data = "..." #enter path here
fileloc = os.path.join(path_data, "ALL_GNPS_15_12_2021_positive_annotated.pickle")
data = pd.read_pickle(fileloc)


def metadata_processing(spectrum):
    spectrum = default_filters(spectrum)
    return spectrum


#getting instrument types#
instrumentType = [d.get("source_instrument") for d in data]
type_occurrences = Counter(instrumentType)
occurrence_list = type_occurrences.items()
occurrence_list = sorted(occurrence_list)
x, y = zip(*occurrence_list)

#plotting values as barplots#
plt.figure(figsize = (15,5), dpi=150, tight_layout=True)
plt.bar(x,y)
plt.title("Occurrences of different instrument types (before filtering)")
plt.xlabel("Instrument Types")
plt.xticks(fontsize=7,rotation=90)
plt.ylabel("Occurrences (log)")
plt.yscale("log")
plt.show()

#filtering methods#
analyzerAliases = {}
analyzerCategories = {}

r1 = re.compile(".*orbitrap.*|.*hcd.*|.*q-exactive.*|.*lumos.*|.*velos.*", re.IGNORECASE)
r2 = re.compile(".*tof.*|.*impact*", re.IGNORECASE)
r3 = re.compile(".*FT.*", re.IGNORECASE)
r4 = re.compile(".*QQQ.*|.*QQ.*|.*Quadrupole.*", re.IGNORECASE)
r5 = re.compile(".*Ion trap.*|.*IT$", re.IGNORECASE)

orbitrap = list(filter(r1.match, x))
tof = list(filter(r2.match, x))
ft = list(filter(r3.match, x))
quadrupole = list(filter(r4.match, x))
quadrupole.append('ESI-LC-ESI-Q')
iontrap = list(filter(r5.match, x))
iontrap.remove('ESI-IT-FT/ion trap with FTMS')

analyzerAliases["Orbitrap"] = orbitrap
analyzerAliases["TOF"] = tof
analyzerAliases["Fourier Transform"] = ft
analyzerAliases["Quadrupole"] = quadrupole
analyzerAliases["Ion Trap"] = iontrap


#ideally add a new filtering method that checks for multiple matches and prompts user to make a decision#


def dict_comparer(dict1, dict2, dict3):
    """Takes a dictionary containing list values and compare those list values to keys of another dictionary.
    The values associated with the second dictionary are counted.
    The key from the first dictionary and the sum are put into a third dictionary. """
    for methd in dict1:
        type_count = 0
        analyzer_list = dict1.get(methd)
        for key in analyzer_list:
            key_num = dict2[key]
            type_count += key_num
        dict3[methd] = type_count
    return dict3

dict_comparer(analyzerAliases, type_occurrences, analyzerCategories)

#maybe change this to a function that can extract any desired combination of metadata fields per spectrum#
def extractType_and_Weight(data):
    """Takes the list of spectra and returns a list of tuples containing the instrument type and the parent
    mass associated with a particular spectrum"""
    types_and_weights = []
    for d in data:
        currentpair = ()
        insttype = d.get("source_instrument")
        pmass = d.get("parent_mass")
        currentpair = (insttype, pmass)
        types_and_weights.append(currentpair)
    return types_and_weights

def massSorter(x,y):
    """Takes a dictionary containing aliases as its x-argument and a list containing spectral metadata as its y-arg.
    sorts every list entry based on the instrument-type and returns a dictionary with the key being instrument-type
    category and the values are the associated parent mass."""
    outputdict = {}
    for kv in x:
        outputlst = []
        aliasList = x.get(kv)
        for tup in y:
            if tup[0] in aliasList:
                outputlst.append(tup[1])
            else:
                pass
        outputdict[kv] = outputlst
    return outputdict

#used for producing a histogram
# colors = ['blue', 'violet', 'orange', 'green', 'red]
# labels = ['Orbitrap', 'Fourier Transform', 'Quadrupole', 'Ion Trap', 'TOF']
#plt.figure(figsize = (10,5), dpi=150, tight_layout=True)
#plt.ylabel('Density')
#plt.xlabel('Parent Mass')
#plt.hist(lweights, histtype='step', density=True, bins=300, align='mid', color=colors, label=labels)
#plt.legend()
