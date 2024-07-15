import numpy as np
import pandas as pd
from time import time

from utils import text

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import *


def get_cv_strategy(n_splits=10):
    """
    Return the CV object. Defaults to 10 splits.
    v1 from 13.12.23
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=3)


def get_scoring(n_labels, scorer=True):
    """
    Define scoring functions for binary and multi-class classifications.
    v3 from 19.12.23
    """
    # Binary vs. multi-class classification
    if n_labels == 2:
        scoring = {
            'Average precision': 'average_precision', 
            'ROC AUC': 'roc_auc',
        }
    else:
        scoring = {
            'ROC AUC OvR': 'roc_auc_ovr', 
            'Weighted ROC AUC OvR': 'roc_auc_ovr_weighted',
        }
        
    if scorer:
        # Scorers compatible with scikit-learn functions
        return scoring
    else:
        # Replace scikit-learn scorers with callables    
        for m in scoring.keys():
            if m=='Average precision':
                scoring[m] = lambda x,y: average_precision_score(x, y[:,1])
            if m=='ROC AUC':
                scoring[m] = lambda x,y: roc_auc_score(x, y[:,1])
            if m=='ROC AUC OvR':
                scoring[m] = lambda x,y: roc_auc_score(x, y, multi_class='ovr')
            if m=='Weighted ROC AUC OvR':
                scoring[m] = lambda x,y: roc_auc_score(x, y, multi_class='ovr', average='weighted')
                
        return scoring


def score_cv(model, X, y, groups=None):
    """
    Train and evaluate a model using cross-validation.
    v1 from 14.12.23
    """
    # Define CV strategy
    cv = get_cv_strategy()
    
    # Define scoring functions
    scoring = get_scoring(y.nunique())
    
    # Start timer
    start_time = time()
    
    scores = cross_validate(estimator=model, X=X, y=y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    
    train_time = time() - start_time
    
    # Print results
    print("_" * 80)
    print("Training with %d-fold cross-validation:" % cv.n_splits)
    
    # Print model name
    try:
        print(model[-1])
    except:
        print(model)
        
    # Print training time    
    print("train time: %0.3fs" % train_time)
    
    # Print metric values
    for k,v in scores.items():
        if 'test' in k:
            print(k.split('_')[1] + " score: %0.3f (+/- %0.2f)" % (v.mean(), v.std()))
    print()
    
        
def search_params(model, search_mode, param_grid, X, y, groups=None, n_splits=10, refit=False, verbose=True):
    """
    Perform grid/random search to find optimal hyperparameter values.
    v2 from 19.12.23
    """
    # Define CV strategy
    cv = get_cv_strategy(n_splits)
    cv_generator = cv.split(X, y, groups)
    
    # Define scoring functions
    scoring = list(get_scoring(y.nunique()).values())[0]

    if search_mode=='grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, 
                              cv=cv_generator, scoring=scoring, n_jobs=-1, 
                              refit=refit)
    elif search_mode=='random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, 
                                    cv=cv_generator, scoring=scoring, n_jobs=-1, 
                                    refit=refit)
        
    search_result = search.fit(X, y)
    
    print("Best for current fold: %.3f using %s" % (search_result.best_score_, search_result.best_params_))
    
    if verbose:
        for mean, std, param in zip(search_result.cv_results_['mean_test_score'], 
                                    search_result.cv_results_['std_test_score'], 
                                    search_result.cv_results_['params']):
            print("%.3f (+/- %.3f) with: %r" % (mean, std, param))
        print()
            
    if refit:        
        return search_result.best_estimator_
    else:
        return search_result.best_params_
    
    
def benchmark_nestedcv(model, search_mode, param_grid, X, y, groups=None):
    """
    Perform nested CV: tune hyperparameters in the inner loop 
    and evaluate the model in the outer loop. 
    v2 from 19.12.23
    """
    # Define CV strategies
    cv = get_cv_strategy()
    
    # Define scoring functions
    scoring = get_scoring(y.nunique(), scorer=False)
    
    # Dictionary for storing metric values
    scores = {k: [] for k in scoring.keys()}
    
    print("_" * 80)
    print("Evaluating with outer %d-fold cross-validation and tuning hyperparameters with inner 3-fold cross-validation:" % cv.n_splits)
    try:
        print(model[-1])
    except:
        print(model)
    
    start_time = time()
    
    for inner_idx, outer_idx in cv.split(X, y, groups):
        
        # Inner loop to search for best hyperparameter values
        model = search_params(model=model, search_mode=search_mode, param_grid=param_grid, 
                              X=X.loc[inner_idx], y=y.loc[inner_idx], groups=groups.loc[inner_idx], 
                              n_splits=3, refit=True, verbose=False)
                
        # Make predictions on the outer fold
        y_proba = model.predict_proba(X.loc[outer_idx])
        
        # Record metrics        
        for k,v in scoring.items():
            scores[k].append(v(y.loc[outer_idx], y_proba))
        
    train_time = time() - start_time
    
    # Print training time
    print("train time: %0.3fs" % train_time)
    
    # Print metric values
    for k,v in scores.items():
        print(k + " score: %0.3f (+/- %0.2f)" % (np.mean(v), np.std(v)))
    
    
def predict_cv(model, X, y, groups=None):
    """
    Train a model and make predictions using cross-validation.
    v1 from 14.12.23
    """
    # Define CV strategy
    cv = get_cv_strategy()
    
    # Make predictions within CV
    y_proba = cross_val_predict(estimator=model, X=X, y=y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)
    
    return y_proba


def find_optimal_threshold(y, y_proba, curve_type='PR', beta=1.0):
    """
    Find optimal threshold value based on the ROC/PR curve for binary classification.
    v1 from 14.12.23
    """
    if curve_type=='ROC':
        print("The threshold optimises G-means calculated from the ROC curve.")
        metric = "G-means"
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        values = np.sqrt(tpr * (1-fpr))
        
    elif curve_type=='PR':
        print("The threshold optimises F1-score calculated from the PR curve.")
        metric = "F1-score"
        prec, rec, thresholds = precision_recall_curve(y, y_proba)
        values = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec)
            
    # Find optimal threshold
    idx = np.argmax(values)
    thresh = thresholds[idx]

    # Subtract a small value to classify cases on the decision boundary as positive
    eps = 0.000001
    thresh -= eps
    
    print('Best threshold = %.3f, %s = %.3f' % (thresh, metric, values[idx]))
    print()
    
    return thresh


def thresholds_cv(y, y_proba, curve_type='PR', beta=1.0):
    """
    Print optimal threshold values for individual CV folds.
    v2 from 04.01.24
    """
    df = pd.concat([y, pd.Series(y_proba, name='y_proba', index=y.index)], axis=1)
    
    # Find optimal threshold for each CV fold
    optimal_thresholds = df.groupby('val_fold').apply(lambda x: 
                                                      find_optimal_threshold(x.y.cat.codes, x.y_proba))
    
    print("Average optimal threshold: %0.3f (+/- %0.2f)" % 
          (optimal_thresholds.mean(), optimal_thresholds.std()))
    
    
def extract_cv(df, ner=True):
    """
    Extract information using cross-validation: learn vocabulary, initialise NLP pipeline with termsets and run NER.
    v1 09.01.24
    """
    # Define CV strategy
    cv = get_cv_strategy()

    for train_idx, val_idx in cv.split(df.clean_text, df.y, df.patient_id):

        # Learn vocabulary and termset
        if ner:
            vocab = text.learn_vocab(df.loc[train_idx].histopathology_id, expand=True)
            
        termset = text.learn_termset(df.loc[train_idx].histopathology_id)

        # Load NLP pipeline
        nlp = text.build_nlp_pipeline(termset)
        
        if ner:
            # Match phrases in text to detect concepts
            df.loc[val_idx, 'doc'] = text.detect_concepts(df.loc[val_idx, 'clean_text'], nlp, vocab)
        else:
            # Label known concepts
            df.loc[val_idx, 'doc'] = text.label_concepts(df.loc[val_idx, ['histopathology_id', 'clean_text']], 
                                                         nlp)

    # Transform predictions to a table of concepts
    return text.get_concepts(df)  
    