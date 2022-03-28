from collections import Counter
import re
import os
import pickle


def instrNames(x):
    """Extracts all of the unique instrument type entries and returns a list of tuples in the form [(x1,y1),..,(xn,yn)]
    with x being a unique entry and y the amount of times it occurs in the dataset"""
    instType = [i.get("source_instrument") for i in x]
    type_occs = Counter(instType).items()
    return type_occs


def keywordCreation(x):
    """Shows the user a list of unique entries and prompts them to create a list of keywords used for the regexp.
     Expected input is a list of tuples (the output from instrNames). Output is a list of regexp objects"""
    names, occs = zip(*x)
    """some subroutine goes here that shows users the list of names, then prompts them to input keywords with a 
    (user-defined) category name (e.g. user calls a set of keywords 'Orbitrap'), then asks whether they want 
    case-sensitivity and then moves on to the next set of keywords or returns a list of regexp objects"""


def aliasCreation(x,y):
    """Takes a list of regexp objects as x and the list of instrument type entries as y. Returns a dictionary with all
    known aliases. Key:value pairs are category:entries, e.g. TOF:ESI-qToF, Esi-QTOF, ESI-qtof. Checks for duplicates,
    like for instance ESI-qtof being in both TOF and quadrupole and prompts user to make a choice on in which category
    the duplicate should be placed"""


    
def instrFilter(x,y):
    """takes aliases(dictionary) as x and the to be filtered dataset as y. returns a dictionary with the sorted
    spectra as values and the instrument-type category as key"""
    outputdict = {}
    for kv in x:
        outputlist = []
        aliasList = x.get(kv)
        for spectrum in y:
            if spectrum.get("source_instrument") in aliasList:
                outputlist.append(spectrum)
            else:
                pass
        outputdict[kv] = outputlist
    return outputdict


def subsetCreation(x,y):
    """Takes output from instrFilter as its x-arg and the desired category as y. Returns a list of spectra"""
    output = x.get(y)
    return output


def exportAsPickle(x,y):
    """Exports the produced subset as a pickle file. Takes the subset (list) as its x, the desired file location as y.
    Prompts the user for a filename"""
    z = input("Enter the filename: ")
    file_export_pickle = os.path.join(y, z + ".pickle")
    pickle.dump(x, open(file_export_pickle, "wb"))
    return "Saved as: " + z


