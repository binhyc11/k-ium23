import os, json, numpy, csv, pandas
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'browser'

def dump_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def load_json(json_file_name):
    with open(json_file_name, 'r') as file:
        data = json.load(file)
    return data

def cal_simple_acc(preds, labels):
    preds = numpy.asanyarray(preds)
    labels = numpy.asanyarray(labels)
    return (preds == labels).mean()

def cal_sensi_speci(labels, preds):
    """
    @desc:
        - Calculate specificity and sensitivity
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    return (sensitivity, specificity)

def visualize_roc(labels, probs, model_name):
    """
    @description:
        +) drawing ROC curves for multiple classes classification
        +) compute ROC curve and ROC area for each class
    @parameters:
        +) labels:
                -) one-hot vector of the labels
                -) examples: [[0, 0, 1], [0, 1, 0]]
        +) probs:
                -) predicting probs scores
                -) not necessary to be in probability range [0 -> 1]
    """
    lw = 3 #line width
    labels = encode_one_hot_vector(labels, 2)
    probs = numpy.asanyarray(probs)
    labels = numpy.asanyarray(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(labels[:, 1], probs[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])
    # print (roc_auc[0])
    # Plot all ROC curves
    # plt.figure()

    # colors = ['aqua', 'darkorange', 'black']
    # plt.plot(fpr[0], tpr[0], color=colors[2], lw=lw,
    #          label='AUC = {1:0.3f})'
    #            #label='ROC curve of positive class (area = {1:0.3f})'
    #                 ''.format(0, roc_auc[0]))

    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f"{model_name}")
    # plt.legend(loc="lower right")
    # plt.savefig(f'logs/{model_name}_roc.png')
    fig = px.area(
        x=fpr[0], y=tpr[0],
        # title=f'ROC Curve (AUC={roc_auc[0]:.3f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate')
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.add_trace(go.Scatter(x=[0.8], y=[0.1], mode='text', text=f'AUC={roc_auc[0]:.3f}', textposition='bottom center'))
    fig.update_yaxes(constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.show()
    
    
def encode_one_hot_vector(labels, num_classes):
    """
    @description:
    """
    encoder = numpy.eye(num_classes, dtype=numpy.int)
    one_hot_vectors = encoder[labels]
    return one_hot_vectors

def count_neg_pos(labels):
    """
    @desc:
        - Count number of positive and negative samples
    """
    neg = 0
    pos = 0
    for label in labels:
        if label == 0:
            neg += 1
        if label == 1:
            pos += 1
    return neg, pos


def summary(file_name, acc, sensi, speci, labels, title):
    num_neg, num_pos = count_neg_pos(labels)
    with open(f'logs/{file_name}', 'a') as file:
        file.write(f'{title}\n')
        file.write(f'ACC: {acc}\n')
        file.write(f'Sensitivity: {sensi}\n')
        file.write(f'Specificity: {speci}\n')
        file.write(f'#Negative samples: {num_neg}\n')
        file.write(f'#Positive samples: {num_pos}\n')
        
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='binary',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/numpy.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = numpy.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = numpy.trace(cf) / float(numpy.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        #if len(cf)==2:
        #Metrics for Binary Confusion Matrices
        #precision = cf[1,1] / sum(cf[:,1])
        #recall    = cf[1,1] / sum(cf[1,:])
        #f1_score  = 2*precision*recall / (precision + recall)
        #stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        #    accuracy,precision,recall,f1_score)
        #else:
        stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    # plt.savefig(f'logs/{title}_cnf.png')

def create_csv(path, data, header):
    with open (path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
def load_csv (path):
    df = pandas.read_csv(path, dtype=str)
    return df

def cal_auc(labels, probs):
    """
    @description:
        +) drawing ROC curves for multiple classes classification
        +) compute ROC curve and ROC area for each class
    @parameters:
        +) labels:
                -) one-hot vector of the labels
                -) examples: [[0, 0, 1], [0, 1, 0]]
        +) preds:
                -) prediction scores
                -) not necessary to be in probability range [0 -> 1]
    """
    lw = 4 #line width
    labels = encode_one_hot_vector(labels, 2)
    probs = numpy.asanyarray(probs)
    labels = numpy.asanyarray(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(labels[:, 1], probs[:, 1])
    roc_auc[0] = auc(fpr[0], tpr[0])
    return roc_auc[0]

def Precision_Recall_curve(yhat, y_true, lr_probs):
    """
    yhat: predicted label
    y_true:  true_label
    lr_probs = lr_probs[:, 1]  # probabilities for the positive outcome only
    """
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, lr_probs)
    lr_f1, lr_auc = f1_score(y_true, yhat), auc(lr_recall, lr_precision)
    # plot the precision-recall curves
    y_true = numpy.array(y_true)
    no_skill = len(y_true[y_true==1]) / len(y_true)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lr_recall, y=lr_precision, line_color='Gray', fill='tozeroy'))#, line_color='darkcyan', fill='tozeroy'))

    fig.update_layout(
        title='Precision-Recall curve',
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0 = 0, x1 = 1,y0 = no_skill, y1=no_skill, name = 'No skill')
    fig.add_trace(go.Scatter(x=[0.05], y=[no_skill], mode='text',
                             text='No skill model',
                             textposition='top right'))
    fig.add_trace(go.Scatter(x=[0.8], y=[0.1], mode='text',
                             text=f'AUC={auc(lr_recall, lr_precision):.3f}',
                             textposition='bottom center'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', constrain='domain', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', constrain='domain', mirror=True, gridcolor='Lightgray')
    fig.show()
    # summarize scores
    print('f1=%.3f AUPRC=%.3f' % (lr_f1, lr_auc))
    return lr_f1, lr_auc