import numpy as np
import pandas as pd
import re
from time import time

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.tokens import Span, SpanGroup
from spacy.util import filter_spans

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import *

# Pretty plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12

### Helper functions
def print_stats(df):
    """
    Print the number of patients and reports and the proportion of positives in a subset of data.
    """
    print("Number of patients:", df.patient_id.nunique())
    print("Number of reports:", df.report_id.nunique())
    print()
    print("Report-level annotation:")
    print(df.y_report.value_counts())
    print()
    print("Proportion of positive reports: %.1f%%" % 
      ((df.y_report == "Positive").sum() / df.shape[0] * 100))
    
    
def get_filename(patient_id, report_no, file_format='ann'):
    """
    Return the filename of the annotation file.
    """
    return "pt" + str(patient_id) + "_r" + str(report_no) + "." + file_format


def get_feature_names(feature_set):
    """
    Return the list of concepts, relations, or composite concepts.
    """
    if feature_set=="concepts":
        return ['ClinicalQuery', 'FungalDescriptor', 'Fungus', 'Invasiveness', 'Stain', 'SampleType', 
                'positive', 'equivocal', 'negative']
    elif feature_set=="relations":
        return ['positive-rel', 'equivocal-rel', 'negative-rel', 
                'fungal-description-rel', 'invasiveness-rel', 'fungus-stain-rel']
    elif feature_set=="composite":
        return ['affirmedFungalDescriptor', 'affirmedFungus', 'affirmedInvasiveness', 'affirmedStain',
                'negatedFungalDescriptor', 'negatedFungus', 'negatedInvasiveness', 'negatedStain']


def get_ent_types():
    """
    Return the list of concepts to apply Negex to.
    """
    return ['FungalDescriptor', 'Fungus', 'Invasiveness', 'Stain']

    
def get_cv_strategy(n_splits=10):
    """
    Return the CV object.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=3)


def read_report(x, path):
    """
    Import report texts from .txt files.
    """
    # Define filename
    filename = get_filename(x.patient_id, x.report_no, file_format='txt')
    
    # Open and read text file
    with open(path + filename, 'r') as f:
        text = f.read()
    
    return text


### Preprocessing
def clean_text(text):
    """
    Apply simple text preprocessing to reports to remove duplicated whitespaces, 
    add a whitespace after a question mark, 
    add an underscore to "Lab No", 
    and convert to lower case.
    """
    # Replace any number of consequetive space chars with a single whitespace
    pattern = re.compile("\s+")
    text = pattern.sub(r" ", text)
    
    # Replace ; with white space
    pattern = re.compile(";")
    text = pattern.sub(r" ", text)
    
    # Add a whitespace between a question mark and the following word
    pattern = re.compile("\?(?=\w)")
    text = pattern.sub(r"? ", text)
    
    # Add an underscore to Lab No to prevent confusion with cases when no is a negation cue
    pattern = re.compile("Lab No")
    text = pattern.sub(r"Lab_No", text)
    
    # Convert all text to lowercase
    text = text.lower()
    
    return text
    

def ann2counts(df):
    """
    Count the number of concepts and relations in gold standard annotations
    and add these counts to the dataframe.
    """
    # Load gold standard concepts and relations
    concepts = pd.read_csv("../datasets/gold_composite.csv")
    relations = pd.read_csv("../datasets/gold_relations.csv")
    
    # Define fetaure names
    feature_names = get_feature_names("concepts") + get_feature_names("relations") + get_feature_names("composite")
    
    # Add columns to store counts
    df[feature_names] = 0
    
    # Add counts for gold standard annotations
    for i, row in df.iterrows():
        
        # Count concepts
        x = concepts[concepts.report_id==row.report_id]
        df.loc[i, x.concept.unique()] = x.concept.value_counts()
        
        # Count relations
        x = relations[relations.report_id==row.report_id]
        df.loc[i, x.relation.unique()] = x.relation.value_counts()
        
    return df


def preprocess_phrases(x):
    """
    Convert to lowercase and apply the same preprocessing as to report texts.
    """
    # Convert to lowercase
    x = x.lower()
    
    # Ensure the same preprocessing is applied to text and keywords
    pattern = re.compile("\s+")
    x = pattern.sub(r" ", x)
    
    pattern = re.compile("\?(?=\w)")
    x = pattern.sub(r"? ", x)
    
    return x


def create_vocab(report_ids, expand=False):
    """
    Collate a vocabulary of phrases annotated for each concept category.
    """
    # Load gold standard concepts
    concepts = pd.read_csv("../datasets/gold_composite.csv")
    
    # Preprocess concept phrases
    concepts.phrase = concepts.phrase.apply(preprocess_phrases)
    
    # Create an empty dict to store vocabulary
    vocab = {ft: [] for ft in get_feature_names("concepts")}
    
    # Update vocabulary with concept phrases
    vocab.update(concepts[concepts.concept.isin(get_feature_names("concepts")) & 
                          concepts.report_id.isin(report_ids)
                         ].groupby('concept').phrase.unique())
    
    # Convert dict of arrays to dict of lists
    vocab = {k: set(v) for k,v in vocab.items()}
    
    print("Number of unique tokens in each category:", [len(vocab[ft]) for ft in vocab])
        
    # Expand the Invasiveness category
    if expand:
        return expand_vocab(vocab)
    else:
        return vocab


def expand_vocab(vocab):
    """
    A custom function to expand the Invasiveness category with same-root words.
    """
    if any(['angio' in token for token in vocab['Invasiveness']]):
        vocab['Invasiveness'] = vocab['Invasiveness'].union(['angio-invasion',
                                                             'angio-invasive',
                                                             'angioinvasion',
                                                             'angioinvasive'])
        
    if any(['infiltrat' in token for token in vocab['Invasiveness']]):
        vocab['Invasiveness'] = vocab['Invasiveness'].union(['infiltrated',
                                                             'infiltrating',
                                                             'infiltration'])
        
    print("Number of unique tokens in each category after expanding:", [len(vocab[ft]) for ft in vocab])
    
    return vocab
    

### Automated information extraction
def build_nlp_pipeline(vocab):
    """
    Define a Spacy pipeline with a customised NegEx and a custom affirmation detector. Exclude NER. 
    """
    # Load Spacy model without NER
    nlp = spacy.load("en_core_web_sm", exclude=['ner'])
    
    # Default termsets
    ts = termset("en_clinical").get_patterns()
    
    # Get the list of concepts to apply Negex to
    ent_types = get_ent_types()

    # Add custom NegEx to the pipeline
    nlp.add_pipe("negex", name="custom_negex", config={'ent_types': ent_types,
                                                       'neg_termset': {
                                                           'preceding_negations': list(set(ts['preceding_negations']).
                                                                                       union(vocab['negative'])),
                                                           'following_negations': list(set(ts['following_negations']).
                                                                                       union(vocab['negative'])),
                                                       }})
    
    # Add an affirmation detector
    nlp.add_pipe("negex", name="affirmator", config={'ent_types': ent_types, 
                                                     'extension_name': 'affirm', 
                                                     'neg_termset': {
                                                         'preceding_negations': list(vocab['positive']),
                                                         'following_negations': list(vocab['positive']),
                                                     }})
    return nlp


def get_matcher(nlp, vocab):
    """
    Macth concepts and return raw matched spans.
    """
    # Initialise a matcher object
    matcher = PhraseMatcher(nlp.vocab)

    # Add patterns to matcher from vocabulary
    for ft in get_feature_names("concepts"):
        if ft in vocab:
#             print('---', ft, '---')
#             print(vocab[ft])
            patterns = list(nlp.pipe(vocab[ft]))
            matcher.add(ft, None, *patterns)

    return matcher


def span_filter(x):
    """
    A custom function that filters spans to resolve overlapping. 
    """

    filtered_spans = SpanGroup(x.doc, name="filtered_spans", spans=[])

    j = 1
    for span in x.spans:
        if span in filtered_spans:
            j+=1
            continue
        try:
            if (span.start == x.spans[j].start) & (span.end == x.spans[j].end):
                filtered_spans.extend([s for s in [span, x.spans[j]] if s.label_!='ClinicalQuery'])
                j+=1
            else:
                filtered_spans.append(span)
                j+=1
        except:
            filtered_spans.append(span)
            
    x.doc.ents = filter_spans(filtered_spans)
                
    return x.doc


def extract_features_cv(df):
    """
    Learn the vocabulary from training data, initialise an NLP pipeline and apply it to the validation set.
    """
    cv = get_cv_strategy()
    
    for train_idx, val_idx in cv.split(df.clean_text, df.y_report, df.patient_id):
        # Create vocabulary
        vocab = create_vocab(df.loc[train_idx].report_id, expand=True)

        # Load NLP pipeline
        nlp = build_nlp_pipeline(vocab)

        # Create matcher
        matcher = get_matcher(nlp, vocab)

        # Run NLP pipeline
        with nlp.select_pipes(disable=["custom_negex", "affirmator"]):
            df.loc[val_idx, 'doc'] = df.loc[val_idx, 'clean_text'].apply(nlp)

        # Extract spans
        df.loc[val_idx, 'spans'] = df.loc[val_idx, 'doc'].apply(lambda x: matcher(x, as_spans=True))

        # Custom span filter
        df.loc[val_idx, 'doc'] = df.loc[val_idx, ['doc', 'spans']].apply(span_filter, axis=1)

        # Detect negation
        nlp_component = nlp.pipeline[-2][1]
        df.loc[val_idx, 'doc'] = df.loc[val_idx, 'doc'].apply(nlp_component)

        # Detect affirmation
        nlp_component = nlp.pipeline[-1][1]
        df.loc[val_idx, 'doc'] = df.loc[val_idx, 'doc'].apply(nlp_component)
    
    return df


def extract_features(df, vocab):
    """
    Initialise an NLP pipeline using the provided vocabulary and apply it to the test set.
    """
    # Load NLP pipeline
    nlp = build_nlp_pipeline(vocab)

    # Create matcher
    matcher = get_matcher(nlp, vocab)

    # Run NLP pipeline
    with nlp.select_pipes(disable=["custom_negex", "affirmator"]):
        df['doc'] = df.clean_text.apply(nlp)

    # Extract spans
    df['spans'] = df.doc.apply(lambda x: matcher(x, as_spans=True))

    # Custom span filter
    df['doc'] = df[['doc', 'spans']].apply(span_filter, axis=1)

    # Detect negation
    nlp_component = nlp.pipeline[-2][1]
    df['doc'] = df.doc.apply(nlp_component)

    # Detect affirmation
    nlp_component = nlp.pipeline[-1][1]
    df['doc'] = df.doc.apply(nlp_component)
    
    return df


def doc2counts(doc, feature_names):
    """
    For each concept category, count the number of matched tokens in a given document.
    Check if a concept is negated. If it is not negated, check if it is affirmed. 
    """
    
    # Create an empty dict to store counts
    counts = {ft: 0 for ft in feature_names}
    
    # Get the list of concepts to check for affirmation/negation
    ent_types = get_ent_types()

    # Count concepts and relations
    for ent in doc.ents:
        if ent.label_ in feature_names:
            counts[ent.label_] += 1
            if ent.label_ in ent_types and ent._.negex:
                composite_label = 'negated' + ent.label_
                if composite_label in feature_names:
                    counts[composite_label] += 1
            elif ent.label_ in ent_types and ent._.affirm:
                composite_label = 'affirmed' + ent.label_
                if composite_label in feature_names:
                    counts[composite_label] += 1

    return pd.Series(counts)


def doc2tokens(doc, feature_names):
    """
    For each concept category, count the number of matched tokens in a given document.
    Check if a concept is negated. If it is not negated, check if it is affirmed. 
    """
    
    # Create an empty dict to store counts
    tokens = {ft: "" for ft in feature_names}
    
    # Record tokens
    for ent in doc.ents:
        if ent.label_ in feature_names:
            tokens[ent.label_] = tokens[ent.label_] + ent.text + ", "

    return pd.Series(tokens)


def get_features(df, feature_names=None):
    """
    Convert the information contained in a doc object into counts for concepts and composite concepts.
    """
    # Get the feature names
    if not feature_names:
        feature_names = get_feature_names("concepts") + get_feature_names("composite")
    
    # Convert doc to counts
    return df.doc.apply(doc2counts, feature_names=feature_names)
   

def write_ann_file(x, path):
    """
    Reverse engineer text preprocessing to adjust span start/end positions.
    Write extracted annotations in the .ann file format.
    """
    
    # Original positions of extracted concepts
    start_chars = []
    end_chars = []
    for ent in x.doc.ents:
        start_chars.append(ent.start_char)
        end_chars.append(ent.end_char)
        
    n_ents = len(x.doc.ents)
                        
    # For each occurence of repeated whitespaces, calculate the offset 
    # and adjust start/end positions of relevant entities
    pattern = re.compile("\s\s+")
    for m in re.finditer(pattern, x.report_text):
        offset = m.end() - m.start() - 1
        for i in range(n_ents):
            if m.start() < start_chars[i]:
                start_chars[i] += offset
                end_chars[i] += offset
            elif (m.start() > start_chars[i]) & (m.start() < end_chars[i]):
                end_chars[i] += offset
                
    # For each whitespace added after ?, adjust
    # start/end positions of relevant entities by 1
    pattern = re.compile("\?(?=\w)")
    for m in re.finditer(pattern, x.report_text):
        for i in range(n_ents):
            if m.start() < start_chars[i]:
                start_chars[i] -= 1
                end_chars[i] -= 1
    
    # Get filename of the new annotation file
    filename =  get_filename(x.patient_id, x.report_no)
    
    # Get the list of concepts to check for negation/affirmation
    ent_types = get_ent_types()
    
    # Write annotations with adjusted start/end positions
    with open(path + filename, 'w') as f:
        i = 1
        for ent, start, end in zip(x.doc.ents, start_chars, end_chars):
            ent_text = x.report_text[start:end]
            f.write('T' + str(i) +  '\t' + 
                    ent.label_ + ' ' + 
                    str(start) + ' ' + 
                    str(end) + '\t' + 
                    ent_text + '\n')
            i += 1

            if ent.label_ in ent_types and ent._.negex:
                f.write('T' + str(i) +  '\t' + 
                        'negated' + ent.label_ + ' ' + 
                        str(start) + ' ' + 
                        str(end) + '\t' + 
                        ent_text + '\n')
                i += 1
            elif ent.label_ in ent_types and ent._.affirm:
                f.write('T' + str(i) +  '\t' + 
                        'affirmed' + ent.label_ + ' ' + 
                        str(start) + ' ' + 
                        str(end) + '\t' + 
                        ent_text + '\n')
                i += 1
                
                
### Classifier development   
def benchmark_nestedcv(model, search_mode, param_grid, X, y, groups=None):
    """
    Perform nested CV: tune hyperparameters in the inner loop 
    and evaluate the model in the outer loop. 
    """
    # Define CV strategies
    cv = get_cv_strategy()
    
    # Initalise lists to store metrics
    scores = {
        'test_roc': [],
        'test_ap': [],
    }
    
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
        scores['test_roc'].append(roc_auc_score(y.loc[outer_idx], y_proba[:,1]))
        scores['test_ap'].append(average_precision_score(y.loc[outer_idx], y_proba[:,1]))
        
    train_time = time() - start_time
    
    # Print results
    scores = {k: np.array(v) for k,v in scores.items()}
    
    
    print("train time: %0.3fs" % train_time)
    print("ROC AUC score: %0.3f (+/- %0.2f)" % (scores['test_roc'].mean(), scores['test_roc'].std()))
    print("AP score: %0.3f (+/- %0.2f)" % (scores['test_ap'].mean(), scores['test_ap'].std()))
    
    
    
def search_params(model, search_mode, param_grid, X, y, groups=None, n_splits=10, refit=False, verbose=True):
    """
    Perform grid/random search to find optimal hyperparameter values.
    """
    cv = get_cv_strategy(n_splits)
    cv_generator = cv.split(X, y, groups)
    
    if search_mode=='grid':
        search = GridSearchCV(estimator=model, param_grid=param_grid, 
                              cv=cv_generator, scoring='average_precision', n_jobs=-1, 
                              refit=refit)
    elif search_mode=='random':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, 
                                    cv=cv_generator, scoring='average_precision', n_jobs=-1, 
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
    
    
    
def score_cv(model, X, y, groups=None):
    """
    Train and evaluate a model using cross-validation. 
    """
    cv = get_cv_strategy()
    
    scoring = {
        'roc' : 'roc_auc', 
        'ap' : 'average_precision'
    }
    
    start_time = time()
    
    scores = cross_validate(estimator=model, X=X, y=y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1)
    
    train_time = time() - start_time
    
    print("_" * 80)
    print("Training with %d-fold cross-validation:" % cv.n_splits)
    try:
        print(model[-1])
    except:
        print(model)
    print("train time: %0.3fs" % train_time)
    print("ROC AUC score: %0.3f (+/- %0.2f)" % (scores['test_roc'].mean(), scores['test_roc'].std()))
    print("AP score: %0.3f (+/- %0.2f)" % (scores['test_ap'].mean(), scores['test_ap'].std()))
    print()
    
    
    
def predict_cv(model, X, y, groups=None, options=[]):
    """
    Train a model and make predictions using cross-validation.
    """
    cv = get_cv_strategy()
    
    y_proba = cross_val_predict(estimator=model, X=X, y=y, groups=groups, cv=cv, method="predict_proba", n_jobs=-1)
    y_proba = y_proba[:, 1]
    
    if 'plot_curves' in options:
        cv_generator = cv.split(X, y, groups)
        plot_curves_cv(y, y_proba, cv_generator)
        
    if 'select_threshold' in options:
        cv_generator = cv.split(X, y, groups)
        select_threshold_cv(y, y_proba, cv_generator)
    
    return y_proba



def plot_curves_cv(y, y_proba, cv_generator):
    """
    Plot ROC and PR curves for each CV fold.
    """    
    _, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))
    plt.subplots_adjust(hspace=0.4)
    sns.lineplot(x=[0,1], y=[0,1], lw=0.5, color=sns.color_palette()[0], linestyle='--', ax=ax1)
    
    for _, val_idx in cv_generator:
        
        # Plot ROC curves for each fold
        fpr, tpr, _ = roc_curve(y.loc[val_idx], y_proba[val_idx])
        roc_auc = auc(fpr, tpr)
        sns.lineplot(x=fpr, y=tpr, estimator=None, sort=False, label="AUC = %0.2f" % roc_auc, ax=ax1)
        
        # Plot ROC curves for each fold
        prec, rec, _ = precision_recall_curve(y.loc[val_idx], y_proba[val_idx])
        pr_auc = auc(rec, prec)
        sns.lineplot(x=rec, y=prec, estimator=None, sort=False, label="AUC = %0.2f" % pr_auc, ax=ax2)
        
    ax1.set(xlim=[-0.02, 1.01], ylim=[-0.01, 1.02], 
            xlabel="False Positive Rate", ylabel="True Positive Rate",
            title="ROC curve")
    ax1.legend(loc="upper left", title="ROC AUC", bbox_to_anchor=(1.0, 1.0))
    
    ax2.set(xlim=[-0.02, 1.01], ylim=[-0.01, 1.02], 
            xlabel="Recall", ylabel="Precision",
            title="Precision-Recall curve")
    ax2.legend(loc="upper left", title="PR AUC", bbox_to_anchor=(1.0, 1.0))
    
    
    
def plot_curves(y, y_proba, filename=None):
    """
    Plot ROC and PR curves.
    """
    # Plot a histogram of predicted probabilities
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    sns.histplot(x=y_proba, hue=y, bins=25);
    plt.xlabel("Predicted probabilities");
    
    if filename:
        plt.savefig("../results/hist_" + filename + ".png", bbox_inches='tight', dpi=300)
    
    # Plot ROC curves for each fold
    plt.figure();
    
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    sns.lineplot(x=[0,1], y=[0,1], lw=0.5, color=sns.color_palette()[0], linestyle='--')
    sns.lineplot(x=fpr, y=tpr, estimator=None, sort=False,
                 lw=3, color=sns.color_palette()[2], label="AUC = %0.3f" % roc_auc)
    
    plt.xlim([-0.02, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right", title="ROC AUC");
    
    if filename:
        plt.savefig("../results/roc_" + filename + ".png", bbox_inches='tight', dpi=300)
        
    # Plot ROC curves for each fold
    plt.figure();
    
    prec, rec, t = precision_recall_curve(y, y_proba)
    pr_auc = auc(rec, prec)
    sns.lineplot(x=rec, y=prec, estimator=None, sort=False,
                 lw=3, color=sns.color_palette()[3], label="AUC = %0.3f" % pr_auc)
    
    plt.xlim([-0.02, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right", title="PR AUC");
    
    if filename:
        plt.savefig("../results/pr_" + filename + ".png", bbox_inches='tight', dpi=300)
        
        
        
def select_threshold_cv(y, y_proba, cv_generator, method='pr', beta=1.0, verbose=True):
    """
    Find optimal threshold for each CV fold.
    """        
    thresholds = []
    
    for _, val_idx in cv_generator:
        # Select optimal threshold
        thresh = select_threshold(y.loc[val_idx], y_proba[val_idx], method, beta, verbose)
        thresholds.append(thresh)
    
    thresholds = np.array(thresholds)
    
    print("Average optimal threshold: %0.3f (+/- %0.2f)" % (thresholds.mean(), thresholds.std()))
    
    
    
def select_threshold(y, y_proba, method='pr', beta=1.0, verbose=True):
    """
    Find optimal threshold value based ROC/PR curve.
    """ 
    if method=='roc':
        if verbose:
            print("The threshold optimises G-means calculated from the ROC curve.")
        metric = "G-means"
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        values = np.sqrt(tpr * (1-fpr))
        
              
    elif method=='pr':
        if verbose: 
            print("The threshold optimises F1-score calculated from the PR curve.")
        metric = "F1-score"
        precision, recall, thresholds = precision_recall_curve(y, y_proba)  
        values = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
        
    idx = np.argmax(values)
    thresh = thresholds[idx]
    if verbose:
        print('Best threshold for the model = %.3f, %s = %.3f' % (thresh, metric, values[idx]))
        print()
    
    eps = 0.000001
    thresh -= eps
    
    return thresh



def plot_coefficients(intercept, coefs, feature_names, filename=None):
    """
    Bar plot of coefficient values for a logistic regression model.
    """
    plt.rcParams['figure.figsize'] = (4, 8)
    plt.figure();
    sns.lineplot(x=[0,0], y=[0,18]);
    sns.barplot(x=intercept.tolist() + coefs.tolist(), y=['intercept'] + feature_names, orient='h');
    plt.xticks(rotation=90);
    plt.ylabel("Coefficient");
    
    if filename:
        plt.savefig("../results/coefs_" + filename + ".png", bbox_inches='tight', dpi=300)
        
        
        
def evaluate_classification(y, y_proba, thresh, filename=None):
    """
    Evaluate model performance: print classification report and plot confusion matrix.
    """ 
    # Convert to class labels
    y_pred = np.where(y_proba > thresh, 1, 0)
    
    # Proportion of instances predcited as positive 
    print("Proportion of labels predicted as positive: %.1f%%" % (y_pred.sum() / y_pred.shape[0] * 100))
    
    # Print classification report
    print("Classification report:")
    print(classification_report(y, y_pred))
    
    # Print PPV, Sensitivity, Specificity
    print("PPV: %.2f, Sensitivity: %.2f, Specificity: %.2f" % (precision_score(y, y_pred), 
                                                               recall_score(y, y_pred), 
                                                               recall_score(y, y_pred, pos_label=0)))
    # Plot confusion matrix
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    sns.heatmap(confusion_matrix(y, y_pred), 
                annot=True, fmt='d', annot_kws={'size': 16},
                cmap='Blues', cbar=False, 
                xticklabels=("Negative", "Positive"), 
                yticklabels=("Negative", "Positive"))

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");
    
    if filename:
        plt.savefig("../results/cm_" + filename + ".png", bbox_inches='tight', dpi=300)