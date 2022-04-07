from collections import Counter
import re
import os
import pickle
import pandas as pd


def count_instrument_types(data):
    """Extracts all of the unique instrument type entries and returns a list of tuples in the form [(x1,y1),..,(xn,yn)]
    with x being a unique entry and y the amount of times it occurs in the dataset"""
    instrument_type = [i.get("source_instrument") for i in data]
    type_occurrences = Counter(instrument_type).items()
    return type_occurrences

def unzip_types_occurrences(types_occurrences):
    """Takes the input from count_instrument_types and returns two lists, one list contains all the different spellings
    for the instrument types and the other the amount of times it occurs. Output is two lists"""
    names, occurrences = zip(*types_occurrences)
    return names, occurrences

def create_keyword_list(regexp_list, name_list):
    """This function uses the regexp objects defined above and the list of names output by unzip_types_occurrences.
    returns a list of all the matches between the search terms and the instrument type entries found in the dataset"""
    match_list = list(filter(regexp_list.match, name_list))
    return match_list

def check_for_duplicates(list1, list2):
    """Checks whether there are any duplicates between list1 & list2. Might expand this so more lists can be added"""
    set1 = set(list1)
    set2 = set(list2)
    for element in set1:
        if set1.intersection(set2):
            print(element)
    else:
        pass

def instrument_filter(instrument_aliases, dataset):
    """takes aliases(dictionary) as x and the to be filtered dataset as y. returns a dictionary with the sorted
    spectra as values and the instrument-type category as key"""
    outputdict = {}
    for category in instrument_aliases:
        outputlist = []
        aliasList = instrument_aliases.get(category)
        for spectrum in dataset:
            if spectrum.get("source_instrument") in aliasList:
                outputlist.append(spectrum)
            else:
                pass
        outputdict[category] = outputlist
    return outputdict

def subset_creation(filtered_dataset, category):
    """Takes output from instrument_filter as filtered_dataset and the desired category as category. Output is a list"""
    output = filtered_dataset.get(category)
    return output

def export_as_pickle(subset, file_location, file_name):
    """Exports the produced subset as a pickle file. Takes the subset (list) as its x, the desired file location as y.
    Prompts the user for a filename"""
    file_name = str(file_name)
    file_export_pickle = os.path.join(file_location, file_name + ".pickle")
    pickle.dump(subset, open(file_export_pickle, "wb"))
    return "Saved as: " + file_name

if __name__ == '__main__':
    spectra = pd.read_pickle("G:/Remco Bsc Thesis/Datafiles/ALL_GNPS_15_12_2021_positive_annotated.pickle")
    occurrences_list = count_instrument_types(spectra)
    """Over here I declare my default search list (which is specific to this dataset, ideally I would add a method that 
    allows for easy creation of new keywords"""
    orbitrap_regexp_terms = re.compile(".*orbitrap.*|.*hcd.*|.*q-exactive.*|.*lumos.*|.*velos.*", re.IGNORECASE)
    tof_regexp_terms = re.compile(".*tof.*|.*impact*", re.IGNORECASE)
    ft_regexp_terms = re.compile(".*FT.*", re.IGNORECASE)
    quadrupole_regexp_terms = re.compile(".*QQQ.*|.*QQ.*|.*Quadrupole.*", re.IGNORECASE)
    iontrap_regexp_terms = re.compile(".*Ion trap.*|.*IT$", re.IGNORECASE)

    """Over here I make the match lists for every instrument category. As a handy in-between function I added a method that
    checks for duplicates"""
    instrument_entries, occurrences = unzip_types_occurrences(occurrences_list)

    orbitrap_matches = create_keyword_list(orbitrap_regexp_terms, instrument_entries)
    tof_matches = create_keyword_list(tof_regexp_terms, instrument_entries)
    ft_matches = create_keyword_list(ft_regexp_terms, instrument_entries)
    quadrupole_matches = create_keyword_list(quadrupole_regexp_terms, instrument_entries)
    iontrap_matches = create_keyword_list(iontrap_regexp_terms, instrument_entries)

    """From checking my own data I found I missed one instrument entry and another was misplaced. So down below I correct
    that. Ideally I would also use a function for this but I would have to think about how to get that working."""
    quadrupole_matches.append('ESI-LC-ESI-Q')
    iontrap_matches.remove('ESI-IT-FT/ion trap with FTMS')

    """For now I removed alias_creation and just use this snippet of code to make a dictionary with aliases. Later on I 
    might add a function that does this automatically"""
    instrument_aliases = {}
    instrument_aliases["Orbitrap"] = orbitrap_matches
    instrument_aliases["TOF"] = tof_matches
    instrument_aliases["Fourier Transform"] = ft_matches
    instrument_aliases["Quadrupole"] = quadrupole_matches
    instrument_aliases["Ion Trap"] = iontrap_matches

