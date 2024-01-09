import numpy as np
import pandas as pd

from utils.dataset import labels2cat

from sklearn.metrics import *

# Pretty plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 10


def get_roc_curve_coords(x):
    """
    Get interpolated coordinates of the ROC curve.
    v1 from 14.12.23
    """
    # Create a dataframe to store curve coordinates
    curves = pd.DataFrame(columns=['x1', 'x2','pos_class'])

    # Create a grid for x1-axis
    x1_grid = np.linspace(0.0, 1.0, 101)

    for pos_label in enumerate(x.y.cat.categories):

        # Calculate FPR and TPR
        x1, x2, _ = roc_curve(x.y.cat.codes, x[pos_label[1]], pos_label=pos_label[0])

        # Make sure the first TPR value is 0
        fixed_value_idx = 0

        # Interpolate x2 values
        x2_interp = np.interp(x1_grid, x1, x2)
        x2_interp[fixed_value_idx] = 0.0

        curves = pd.concat([curves, 
                            pd.DataFrame({'x1': x1_grid, 'x2': x2_interp, 
                                          'pos_class': pos_label[1]})],
                           axis=0, ignore_index=True)
     
    # Convert labels to categorical
    curves.pos_class = labels2cat(curves.pos_class, categories=x.y.cat.categories)
        
    return curves


def get_pr_curve_coords(x):
    """
    Get interpolated coordinates of the PR curve.
    v1 from 14.12.23
    """
    # Create a dataframe to store curve coordinates
    curves = pd.DataFrame(columns=['x1', 'x2', 'pos_class'])
    
    # Create a grid for x1-axis
    x1_grid = np.linspace(0.0, 1.0, 101)
    
    for pos_label in enumerate(x.y.cat.categories):
        
        # Calculate FPR and TPR
        x2, x1, _ = precision_recall_curve(x.y.cat.codes, x[pos_label[1]], pos_label=pos_label[0])
        
        x1 = list(reversed(x1))
        x2 = list(reversed(x2))
            
        # Make sure the last Precision value is 1.0
        fixed_value_idx = -1

        # Interpolate x2 values
        x2_interp = np.interp(x1_grid, x1, x2)
        x2_interp[fixed_value_idx] = 0.0
        
        curves = pd.concat([curves, 
                            pd.DataFrame({'x1': x1_grid, 'x2': x2_interp, 
                                          'pos_class': pos_label[1]})],
                           axis=0, ignore_index=True)
        
    # Convert labels to categorical
    curves.pos_class = labels2cat(curves.pos_class, categories=x.y.cat.categories)
    
    return curves


def plot_curve(curves, curve_type, palette=None, filename=None):
    """
    Plot diagnostic curve.
    v3 from 20.12.23
    """
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    if curves.pos_class.nunique()>2:
        # Multiclass classification
        curves['label'] = curves.pos_class.astype(str) + "-vs-Rest"
    else:
        # Binary classification
        curves = curves[curves.pos_class.cat.codes==1]
        curves.pos_class = curves.pos_class.cat.remove_unused_categories()
        curves['label'] = curves.pos_class.astype(str)
        
    if curves.index.nlevels>1:
        # CV results
        estimator='mean'
        errorbar='sd'
        auc_values = curves.groupby(['pos_class', 'val_fold']).apply(lambda x: auc(x.x1, x.x2)).unstack(level=1).agg(['mean', 'std'], axis=1)
        curves['label'] = curves.apply(lambda x: x.label + ": AUC = %.2f (+/- %.2f)" % tuple(auc_values.loc[x.pos_class]), axis=1)
    else:
        # Test set results
        estimator=None
        errorbar=None
        auc_values = curves.groupby('pos_class').apply(lambda x: auc(x.x1, x.x2))
        curves['label'] = curves.apply(lambda x: x.label + ": AUC = %.2f" % auc_values.loc[x.pos_class], axis=1)
        
    if curve_type=="ROC":
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        title = "ROC curve"
        legend_title = "ROC AUC"
        
        # Chance
        sns.lineplot(x=[0,1], y=[0,1], lw=0.5, linestyle='--')
        
    elif curve_type=="PR":
        xlabel = "Recall"
        ylabel = "Precision"
        title = "Precision-Recall curve"
        legend_title = "PR AUC"
    
    # Set up colour palette
    if palette:
        palette = sns.color_palette(palette)
    else:
        palette = 'mako'
        
    sns.lineplot(x='x1', y='x2', hue='label', data=curves, 
                 estimator=estimator, errorbar=errorbar, 
                 lw=2.5, palette=palette);
    plt.xlabel(xlabel);
    plt.ylabel(ylabel);
    plt.title(title);
    plt.legend(title=legend_title, loc='lower right');
    
    if filename:
        plt.savefig("../results/" + curve_type.lower() + "_" + filename + ".png", bbox_inches='tight', dpi=300)
                
        
def plot_diagnostic_curves(y, y_proba, palette=None, filename=None):
    """
    Iterate over data to plot diagnostic curver for different labels and/or folds.
    v3 from 04.01.24
    """
    df = pd.concat([y, pd.DataFrame(y_proba, columns=y.cat.categories, index=y.index)], axis=1)
    
    # ROC curves
    if df.index.get_level_values('val_fold').isna().all():
        curves = get_roc_curve_coords(df)
    else:
        curves = df.groupby('val_fold').apply(get_roc_curve_coords)
        
    plot_curve(curves, curve_type='ROC', palette=palette, filename=filename)
    
    # Precision-recall curves
    if df.index.get_level_values('val_fold').isna().all():
        curves = get_pr_curve_coords(df)
    else:
        curves = df.groupby('val_fold').apply(get_pr_curve_coords)
    
    plot_curve(curves, curve_type='PR', palette=palette, filename=filename)
    
    
def evaluate_classification(y, y_pred, filename=None):
    """
    Evaluate model performance: print classification report and plot confusion matrix.
    v3 from 20.12.23
    """ 
    # Map predictions and convert to categories
    categories = y.cat.categories
    y_pred = pd.Series(y_pred).map(dict(enumerate(categories))) 
    y_pred = labels2cat(y_pred, categories=categories)
    
    # Proportion of predcited labels of each class
    print("Proportion of labels predicted as:\n%s\n" % 
          y_pred.value_counts(normalize=True).sort_index().round(2))
    
    # Print classification report
    print("Classification report:")
    print(classification_report(y, y_pred, labels=categories))
    
    # Print PPV, Sensitivity, Specificity
    if y.nunique()==2:
        print("PPV: %.2f, Sensitivity: %.2f, Specificity: %.2f" % (precision_score(y, y_pred, pos_label=categories[1]), 
                                                                   recall_score(y, y_pred, pos_label=categories[1]), 
                                                                   recall_score(y, y_pred, pos_label=categories[0])))
    
    # Plot confusion matrix
    plt.rcParams['figure.figsize'] = (6, 4)
    plt.figure();
    
    sns.heatmap(confusion_matrix(y, y_pred, labels=categories), 
                annot=True, fmt='d', annot_kws={'size': 16},
                cmap='Blues', cbar=False, 
                xticklabels=categories, 
                yticklabels=categories);

    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix");
    
    if filename:
        plt.savefig("../results/cm_" + filename + ".png", bbox_inches='tight', dpi=300)
        

def plot_coefficients(intercept, coefs, feature_names, filename=None):
    """
    Bar plot of coefficient values for a logistic regression model.
    v1 from 21.12.23
    """
    plt.rcParams['figure.figsize'] = (4, 8)
    plt.figure();
    sns.lineplot(x=[0,0], y=[0,18]);
    sns.barplot(x=intercept.tolist() + coefs.tolist(), y=['intercept'] + feature_names, orient='h');
    plt.xticks(rotation=90);
    plt.ylabel("Coefficient");
    
    if filename:
        plt.savefig("../results/coefs_" + filename + ".png", bbox_inches='tight', dpi=300)