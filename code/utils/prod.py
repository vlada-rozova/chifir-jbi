import numpy as np
import pandas as pd

def get_feature_names(feature_sets):
    """
    Return the list of concepts, relations, or composite concepts.
    v3 from 15.07.24
    """
    feature_names = []
    
    for fs in feature_sets:
        if fs=='concepts':
            feature_names.extend(['ClinicalQuery', 'FungalDescriptor', 'Fungus', 
                                  'Invasiveness', 'Stain', 'SampleType', 
                                  'positive', 'equivocal', 'negative'])
        if fs=='relations':
            feature_names.extend(['positive-rel', 'equivocal-rel', 
                                  'negative-rel', 'fungal-description-rel', 
                                  'invasiveness-rel', 'fungus-stain-rel'])
    
        if fs=='composite':
            feature_names.extend(['affirmed_FungalDescriptor', 'affirmed_Fungus', 
                                  'affirmed_Invasiveness', 'affirmed_Stain',
                                  'negated_FungalDescriptor', 'negated_Fungus', 
                                  'negated_Invasiveness', 'negated_Stain'])
        if fs=='termsets':
            feature_names.extend(['preceding_positive', 'following_positive', 
                                  'preceding_negative', 'following_negative'])
            
    return feature_names


def get_ent_types():
    """
    Return the list of concepts to which to apply Negex.
    v1 from 03.01.24
    """
    return ['FungalDescriptor', 'Fungus', 'Invasiveness', 'Stain']
    

def proba2class(y_proba, thresh=None):
    """
    Convert predicted probabilities to crisp class labels.
    v1 from 14.12.23
    """
    assert (y_proba.min() >= 0) & (y_proba.max() <= 1)
    
    if y_proba.shape[1] > 2:
        return np.argmax(y_proba, axis=1)
    else:
        return np.where(y_proba[:,1] > thresh, 1, 0)


