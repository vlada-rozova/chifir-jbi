import numpy as np
import pandas as pd
import re
from utils.prod import get_feature_names, get_ent_types

import spacy
from spacy.matcher import PhraseMatcher
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.tokens import Span, SpanGroup
from spacy.util import filter_spans


def clean_text(text, return_mapping=False):
    """
    Apply simple text preprocessing to reports and convert to lower case 
    or return the mapping for character positions.
    v2 from 10.01.24
    """
    if return_mapping:
        # Create a list of character position indices
        mapping = list(range(0, len(text)))
    
    # Add a full stop before a section header
    pattern = re.compile("(\s*\n\n[A-Z]{5,})")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start()] + [np.nan]
            i = m.start()

        tmp += mapping[i:]
        mapping = tmp
        
    text = pattern.sub(r".\1", text)
    
    # Separate a plus sign from the preceding word with a space
    pattern = re.compile("(?<=\w)\+(?=\s)")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.start()] + [np.nan]
            i = m.start()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r" +", text)

    # Separate a hyphen from the following word with a space
    pattern = re.compile("(?<=\s)-(?=\w)")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.end()] + [np.nan]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
    
    text = pattern.sub(r"- ", text)
                
    # Separate a question mark from the following word with a space
    pattern = re.compile("\?(?=\w)")
    
    if return_mapping:
        # Adjust indices
        i = 0
        tmp = []
        for m in pattern.finditer(text):
            tmp += mapping[i:m.end()] + [np.nan]
            i = m.end()

        tmp += mapping[i:]
        mapping = tmp
        
    text = pattern.sub(r"? ", text)
        
    # Replace semicolon with a space
    pattern = re.compile(";")
    text = pattern.sub(r" ", text)
        
    # Remove multiple full stops
    pattern = re.compile("\.{2,}")
    
    if return_mapping:
        # Adjust indices
        li = 0
        tmp = []
        for m in pattern.finditer(text):
            ri = m.start() + 1
            tmp += mapping[li:ri]
            li = m.end()

        tmp += mapping[li:]
        mapping = tmp

    text = pattern.sub(r".", text)
        
    # Remove multiple spaces
    pattern = re.compile("\s{2,}")
    
    if return_mapping:
        # Adjust indices
        li = 0
        tmp = []
        for m in pattern.finditer(text):
            ri = m.start() + 1
            tmp += mapping[li:ri]
            li = m.end()

        tmp += mapping[li:]
        mapping = tmp
        
    text = pattern.sub(r" ", text)
    
    # Rstrip
    text = text.rstrip()
    
    # Convert all whitespace characters to space
    pattern = re.compile("\s")
    text = pattern.sub(r" ", text)
    
    # Convert to lowercase
    text = text.lower()
    
    if return_mapping:
        return mapping
    else:
        return text
    
    
def load_annotations(which='concepts'):
    """
    Load gold standard annotations.
    v2 from 10.01.24
    """
    # Load gold standard annotations
    annotations = pd.read_csv("../datasets/gold_" + which + ".csv", converters={'phrase': str})
    
    # Convert concept types to categorical
    if which=='concepts':
        annotations.concept = annotations.concept.astype('category').cat.set_categories(
            get_feature_names(['concepts']))
        
    elif which=='composite':
        annotations.concept = annotations.concept.astype('category').cat.set_categories(
            get_feature_names(['concepts', 'composite']))
        
    elif which=='relations':
        annotations.relation = annotations.relation.astype('category').cat.set_categories(
            get_feature_names(['relations']))
        
    return annotations


def get_concept_counts(report_ids, feature_names, concepts=pd.DataFrame()):
    """
    Count the number of gold standard annotations.
    v2 from 05.01.24
    """
    if concepts.empty:
        # Load gold standard annotations
        concepts = load_annotations('composite')
    
    # Count the number of concepts of each kind
    counts = concepts.groupby('histopathology_id', observed=True).concept.value_counts().reset_index().pivot(
        index='histopathology_id', 
        columns='concept', 
        values='count'
    )
    
    # Merge with report IDs, fill in the missing values with 0s and select features
    return counts.merge(report_ids, 
                        how='right', on='histopathology_id'
                       ).fillna(0).astype(int)[feature_names]


def preprocess_phrase(x):
    """
    Convert to lowercase and apply the same preprocessing as to report texts.
    v1 from 13.12.23
    """
    # Convert to lowercase
    x = x.lower()
    
    # Ensure the same preprocessing is applied to text and keywords
    pattern = re.compile("\s+")
    x = pattern.sub(r" ", x)
    
    pattern = re.compile("\?(?=\w)")
    x = pattern.sub(r"? ", x)
    
    return x


def learn_vocab(report_ids, expand=False):
    """
    For each concept category, learn a dicitonary of unique phrases annotated in the gold standard.
    v1 from 04.01.24
    """
    # Load gold standard annotations
    concepts = load_annotations('concepts')
    
    # Create a dictionary of phrases for each concept category
    vocab = concepts[concepts.histopathology_id.isin(report_ids)
                    ].groupby('concept', observed=False).phrase.unique().to_dict()
    
    # Preprocess and convert to a dict of sets
    vocab = {k: set(preprocess_phrase(t) for t in v) for k,v in vocab.items()} 
    
    print("Number of unique tokens in each category:", [len(vocab[ft]) for ft in vocab])
    
    # Expand the Invasiveness category
    if expand:
        return expand_vocab(vocab)
    else:
        return vocab
    
    
def learn_termset(report_ids):
    """
    For each concept category, learn a dictionary of unique phrases depending on whether it is preceding and following concept.
    v1 from 04.01.24
    """
    # Load gold standard annotations
    concepts = load_annotations('concepts')
    
    # Create an empty dict to store term sets 
    termset = {}

    # Preceding and following termsets
    for loc in ['preceding', 'following']:
        tmp = concepts[concepts.histopathology_id.isin(report_ids) & 
                       concepts.preceding
                      ].groupby('concept', observed=True).phrase.unique().to_dict()

        termset.update({loc + '_' + k: set(preprocess_phrase(t) for t in v) for k,v in tmp.items()})
        
    print("Number of unique tokens in each termset:", [len(termset[ft]) for ft in termset])
    
    return termset


def expand_vocab(vocab):
    """
    A custom function to expand the Invasiveness category with same-root words.
    v1 from 13.12.23
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
    
    
def label_concepts(df, nlp):
    """
    Create a doc object, label gold standard concepts and apply negation and affirmation detection.
    v2 from 05.01.24
    """
    def add_ents(x):
        """
        Add entities to the doc object.
        v2 from 05.01.24
        """
        try:
            start_chars, end_chars, labels = x.spans
            x.doc.ents = [x.doc.char_span(sc, ec, label=l) for sc, ec, l in zip(start_chars, end_chars, labels)]

        except:
            pass

        return x.doc
    
    # Run NLP pipeline
    with nlp.select_pipes(disable=['custom_negex', 'affirmator']):
        df['doc'] = df.clean_text.apply(nlp)
        
    # Load gold standard concepts
    concepts = load_annotations('concepts')

    # Lists of char positions and labels
    spans = concepts.groupby('histopathology_id').apply(lambda x: 
                                                        (x.start_char, x.end_char, x.concept)).rename('spans')

    # Add entities
    doc = df.join(spans, on='histopathology_id').apply(add_ents, axis=1)

    return detect_relations(doc, nlp)


def build_nlp_pipeline(custom_ts):
    """
    Define a Spacy pipeline with a customised NegEx and a custom affirmation detector. Exclude NER.
    v1 from 03.01.24
    """
    # Load Spacy model without NER
    nlp = spacy.load("en_core_web_sm", exclude=['ner'])
    
    # Default termsets
    ts = termset("en_clinical").get_patterns()
    ts['termination'] = [t for t in ts['termination'] if t!='which']
    
    # Get the list of concepts to apply Negex to
    ent_types = get_ent_types()

    # Add custom NegEx to the pipeline
    nlp.add_pipe("negex", name="custom_negex", config={'ent_types': ent_types,
                                                       'neg_termset': {
                                                           'preceding_negations': list(set(ts['preceding_negations']).
                                                                                       union(custom_ts['preceding_negative'])),
                                                           'following_negations': list(set(ts['following_negations']).
                                                                                       union(custom_ts['following_negative'])),
                                                       }})
    
    # Add an affirmation detector
    nlp.add_pipe("negex", name="affirmator", config={'ent_types': ent_types, 
                                                     'extension_name': 'affirm', 
                                                     'neg_termset': {
                                                         'preceding_negations': list(custom_ts['preceding_positive']),
                                                         'following_negations': list(custom_ts['following_positive']),
                                                     }})
    return nlp
    

def get_matcher(nlp, vocab):
    """
    Macth concepts and return raw matched spans.
    v1 from 03.01.24
    """
    # Initialise a matcher object
    matcher = PhraseMatcher(nlp.vocab)

    # Add patterns to matcher from vocabulary
    for ft in vocab:
        patterns = list(nlp.pipe(vocab[ft]))
        matcher.add(ft, None, *patterns)

    return matcher


def span_filter(x):
    """
    A custom function that filters spans to resolve overlapping. 
    v1 from 03.01.24
    """

    filtered_spans = SpanGroup(x.doc, name='filtered_spans', spans=[])

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


def detect_relations(doc, nlp):
    """
    Apply negation and affirmation detection to entities. 
    v1 from 04.01.24
    """
    # Detect negation
    nlp_component = nlp.pipeline[-2][1]
    doc = doc.apply(nlp_component)

    # Detect affirmation
    nlp_component = nlp.pipeline[-1][1]
    doc = doc.apply(nlp_component)
    
    return doc
    
    
def detect_concepts(text, nlp, vocab):
    """
    Create a doc object, detect concepts in text based on the learned vocabulary and apply negation and affirmation detection.
    v1 from 04.01.24
    """
    # Run NLP pipeline
    with nlp.select_pipes(disable=['custom_negex', 'affirmator']):
        doc = text.apply(nlp)
        
    # Create matcher
    matcher = get_matcher(nlp, vocab)

    # Extract spans
    spans = doc.apply(lambda x: matcher(x, as_spans=True))

    # Custom span filter
    doc = pd.DataFrame({'doc':doc,'spans':spans}).apply(span_filter, axis=1)
    
    return detect_relations(doc, nlp)


def doc2concepts(x):
    """
    Convert doc object to a table of concepts.
    v1 from 03.01.24
    """        
    # Create a dataframe to store annotations
    concepts = pd.DataFrame(columns=['histopathology_id', 'patient_id', 'report_no', 
                                     'concept', 'phrase',
                                     'start_char', 'end_char'])   
    
    # Affirmed and negated concepts
    ent_types = get_ent_types()
    
    for i, ent in enumerate(x.doc.ents):
        
        tmp = pd.DataFrame({
            'histopathology_id': x.histopathology_id,
            'patient_id': x.patient_id, 
            'report_no': x.report_no, 
            'concept': ent.label_, 
            'phrase': x.clean_text[ent.start_char:ent.end_char],
            'start_char': ent.start_char,
            'end_char': ent.end_char,
            }, index=[0])

        # Add to the table of concepts
        concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True)
        

        if ent.label_ in ent_types and ent._.negex:
            composite_label = 'negated' + ent.label_
        elif ent.label_ in ent_types and ent._.affirm:
            composite_label = 'affirmed' + ent.label_
        else:
            composite_label = None

        if composite_label:
                        
            tmp = pd.DataFrame({
                'histopathology_id': x.histopathology_id,
                'patient_id': x.patient_id, 
                'report_no': x.report_no, 
                'concept': composite_label, 
                'phrase': x.clean_text[ent.start_char:ent.end_char],
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                }, index=[0])

            # Add to the table of concepts
            concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True) 
            
    return concepts


def get_concepts(df):
    """
    Create a dataframe with all detected concepts in the dataset.
    v1 from 03.01.24
    """
    detected_concepts = pd.concat([doc2concepts(x) for _, x in df.iterrows()], ignore_index=True)
    
    detected_concepts.concept = detected_concepts.concept.astype('category').cat.set_categories(
        get_feature_names(['concepts', 'composite']))
    
    return detected_concepts
    