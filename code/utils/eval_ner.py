import numpy as np
import pandas as pd
import re


def add_bies_tags(feature_names):
    """
    Expand the list of feature names by adding BIES tags.
    v1 from 03.01.24
    """
    return [tag + ft for ft in feature_names for tag in ["B-", "I-", "E-", "S-"]]


def precision(x):
    """
    Calculate precision.
    v1 from 05.01.24
    """
    try:
        return x.tp.sum() / (x.tp.sum()+x.fp.sum())
    except:
        return np.nan

def recall(x):
    """
    Calculate recall.
    v1 from 05.01.24
    """
    try:
        return x.tp.sum() / (x.tp.sum()+x.fn.sum())
    except:
        return np.nan
        
def fscore(p, r):
    """
    Calculate F1 score.
    v1 from 05.01.24
    """
    return 2 * (p * r) / (p + r)


def compare_char_position(tc, candidates, mode='exact'):
    """
    For a given true concept, check if any of the candidate concepts 
    match exactly or overlap in their location.
    v1 from 05.01.24
    """
    if mode=='exact':
        cond1 = candidates.start_char == tc.start_char
        cond2 = candidates.end_char == tc.end_char
        return not candidates[cond1 & cond2].empty
    
    if mode=='overlap':
        cond1a = candidates.start_char <= tc.start_char
        cond1b = candidates.start_char >= tc.start_char
        cond2a = candidates.end_char >= tc.end_char
        cond2b = candidates.end_char <= tc.end_char
        cond3 = candidates.start_char <= tc.end_char
        cond4 = candidates.end_char >= tc.start_char
        
        return candidates[(cond1a & cond2a) | (cond1b & cond3) | (cond2b & cond4)].empty
   

    
def match_concepts(ids, true_concepts, detected_concepts, feature_names):
    """
    For each report, calculate the number of TP, FP, and FN 
    based on the provided tables with gold standard and detected concepts.
    v1 from 05.01.24
    """
    
    # Create a dataframe to record the number of tp, fp, fn
    idx = pd.MultiIndex.from_product([ids, 
                                      pd.CategoricalIndex(feature_names,
                                                          categories=feature_names,
                                                          ordered=True)
                                     ],
                                     names=['histopathology_id', 'concept'])
    col = ['tp', 'fp', 'fn']
    counts = pd.DataFrame(0, idx, col)

    for _,tc in true_concepts[true_concepts.histopathology_id.isin(ids) & 
                              true_concepts.concept.isin(feature_names)
                             ].iterrows():

        candidates = detected_concepts[(detected_concepts.histopathology_id==tc.histopathology_id) & 
                                       (detected_concepts.concept==tc.concept)]

        if compare_char_position(tc, candidates, mode='overlap'):
            counts.loc[tc.histopathology_id, tc.concept].fn += 1
        else:
            counts.loc[tc.histopathology_id, tc.concept].tp += 1

    for _,dc in detected_concepts[detected_concepts.histopathology_id.isin(ids) & 
                                  detected_concepts.concept.isin(feature_names)
                                 ].iterrows():

        candidates = true_concepts[(true_concepts.histopathology_id==dc.histopathology_id) & 
                                   (true_concepts.concept==dc.concept)]

        if compare_char_position(dc, candidates, mode='overlap'):
            counts.loc[dc.histopathology_id, dc.concept].fp += 1
            
    return counts


def evaluate_ner(df, true_concepts, detected_concepts, feature_names):
    """
    Calculate the number of TP, FP, and FN per concept.
    v1 from 05.01.24
    """
    # Count the number of TP, FP, and FN
    counts = match_concepts(df.histopathology_id, 
                            true_concepts, detected_concepts, 
                            feature_names)
    
    if df.val_fold.isna().all():
        # Presicion
        print(counts.groupby('concept').apply(precision).round(2))

        # Recall
        print(counts.groupby('concept').apply(recall).round(2))
        
    else:
        # Add fold numbers to calculate metrics over CV split
        counts = counts.reset_index().merge(df, how='left')

        # Presicion
        print(counts.groupby(['concept', 'val_fold'], observed=False
                            ).apply(precision).groupby('concept').agg(['mean', 'std']).round(2))

        # Recall
        print(counts.groupby(['concept', 'val_fold'], observed=False
                            ).apply(recall).groupby('concept').agg(['mean', 'std']).round(2))