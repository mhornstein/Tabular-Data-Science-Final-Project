from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter, defaultdict
import graphviz
import re
import pandas as pd
import matplotlib
import matplotlib.colors as mc
import colorsys

# To enable renderring when used locally
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

CM_INDICES_ALL = 'all'
PRESENT_ALL = 'all'
PRESENT_VISITED = 'visited'
WHITE_COLOR = 'white'

##################
## Graphvitz utils

replace_words = ['<thead>', '<tbody>', '</thead>', '</tbody>', '\n', '\rn', '<table border="1" class="dataframe">',
                 '</table>']
pattern = '|'.join(replace_words)

def dataframe_to_graphviz_table(df):
    '''
    Translates a html table to graphviz format
    references:
    https://renenyffenegger.ch/notes/tools/Graphviz/attributes/label/HTML-like/index
    https://graphviz.org/Gallery/gradient/table.html
    '''
    html_table = df.to_html()
    return re.sub(pattern, '', html_table).replace('th', 'td').replace('<td>', '<td border="1" bgcolor = "white">')

def to_graphviz_format(node_header, cm):
    cm_table = dataframe_to_graphviz_table(cm)
    return '<<table border = "0">' \
           '<tr><td border = "0"><b>%s</b></td></tr>' \
            '%s' \
           '</table>>' % (node_header, cm_table)

###################
def get_cm_for_node(nodes_to_visiting_samples_id_map, node_id, predicted_label, true_label, cm_indices):
    if node_id not in nodes_to_visiting_samples_id_map: # no sample got here
        return None
    else:
        samples_ids = nodes_to_visiting_samples_id_map[node_id] # take all relevant samples for node with their true and prdicted label

        if CM_INDICES_ALL == cm_indices:
            samples_true_labels = true_label[samples_ids]
            samples_predicted_labels = predicted_label[samples_ids]
            return pd.DataFrame(confusion_matrix(samples_true_labels, samples_predicted_labels))
        else:
            # When required - out of the relevant samples, take only those for whom the true and predicted are asked for
            filtered_samples_ids = [i for i in samples_ids if (true_label[i], predicted_label[i]) in cm_indices]
            if len(filtered_samples_ids) == 0:
                return None
            else:
                samples_true_labels = true_label[filtered_samples_ids]
                samples_predicted_labels = predicted_label[filtered_samples_ids]
                return pd.DataFrame(confusion_matrix(samples_true_labels, samples_predicted_labels))

def lighten_color(color, amount):
    '''
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    referance: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    '''
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    adjusted_c =  colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return '#%02x%02x%02x' % (int(adjusted_c[0] * 255), int(adjusted_c[1] * 255), int(adjusted_c[2] * 255))


def plot_confusion_path(clf, X, y, color = '#9900ffff', present = 'all', cm_indices='all', show_cm = True):
    '''
    Presents how the data in X traverse through the decision tree with respect to the confusion matrix’s
     entries specified by cm_indices, according to the true labels stored in y.
    :param clf: A trained sklearn classification tree.
    :param X: The data for classification.
    :param y: Data true labels.
    :param color: The color for coloring the nodes (i.e. the heat map color)
    :param present: ‘all’ or ‘visited’ [default: ‘all’]
                    use ‘all’ to present all the nodes in the tree (i.e. not only the visited ones that were in use during classification).
                    Use ‘visited’ to present only nodes that were visited by at least one sample during classification.
    :param cm_indices: a list of tuples representing indices in the confusion matrix or ‘all’ [default: ‘all’]
                    When providing a list of indices, only nodes that were visited during the classification
                    of the relevant entries in the confusion matrix are colored.
                    In this case, cm_indices tuples should be of the structure (true label, predicted label)
    :param show_cm: True or False [default: False]
	                Set to True to present the relevant confusion matrices in the nodes.
    '''
    y = np.array(y) # Will be easier if y is guaranteed to be of type np array
    v = graphviz.Digraph()

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(X)
    classification = clf.predict(X)

    nodes_visits_count = Counter()
    nodes_to_visiting_samples_id_map = defaultdict(list)

    for sample_id in range(len(X)):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ]
        nodes_visits_count.update(node_index)
        for node_id in node_index:
            nodes_to_visiting_samples_id_map[node_id].append(sample_id)

    max_visits = nodes_visits_count.most_common(1)[0][1]

    if PRESENT_ALL == present:
        nodes_ids_to_plot = range(n_nodes)
    else:
        nodes_ids_to_plot = [i for i in range(n_nodes) if len(nodes_to_visiting_samples_id_map[i]) != 0]

    for i in nodes_ids_to_plot:
        node_id = str(i)

        if children_left[i] == children_right[i]: # this is a leaf
            label = str(np.argmax(clf.tree_.value[i]))
            if show_cm:
                cm = get_cm_for_node(nodes_to_visiting_samples_id_map, i, classification, y, cm_indices)
                if cm is None:
                    node_content = label
                else:
                    node_content = to_graphviz_format(label, cm)
            else:
                node_content = label

            visits_percentage = nodes_visits_count[i] / max_visits
            fill_color = lighten_color(color, visits_percentage)
            v.node(node_id, node_content, {'shape': 'rect', 'style': 'filled', 'fillcolor': fill_color})

        else: # this is a split node
            node_feature = 'X[' + str(feature[i]) + ']'
            node_tresshold = threshold[i]

            label = str(node_feature)
            if show_cm:
                cm = get_cm_for_node(nodes_to_visiting_samples_id_map, i, classification, y, cm_indices)
                if cm is None:
                    node_content = label
                else:
                    node_content = to_graphviz_format(label, cm)
            else:
                node_content = label

            visits_percentage = nodes_visits_count[i] / max_visits
            fill_color = lighten_color(color, visits_percentage)
            v.node(node_id, node_content, {'shape': 'rect', 'style': 'filled', 'fillcolor': fill_color})

            if children_left[i] in nodes_ids_to_plot:
                v.edge(node_id, str(children_left[i]), node_feature + "<=" + str(node_tresshold))

            if children_right[i] in nodes_ids_to_plot:
                v.edge(node_id, str(children_right[i]), node_feature + ">" + str(node_tresshold))

    v.render(directory='doctest-output', view=True, format='jpeg')  # In collab: needs to be return v

def common_nodes_in_paths(clf, X, color = '#9900ffff'):
    '''
    plot the decision tree and mark only nodes that were visited by *all* the samples in X during classification.
    :param clf: A trained sklearn classification tree.
    :param X: The data for classification.
    :param color: The color for coloring the nodes.
    '''

    v = graphviz.Digraph()

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(X)
    sample_ids = range(len(X))

    # Boolean array indicating the nodes both samples go through
    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    # Obtain node ids using position in array
    common_node_id = np.arange(n_nodes)[common_nodes]

    # Plotting the tree
    for i in range(n_nodes):
        node_id = str(i)
        fill_color = color if i in common_node_id else WHITE_COLOR

        if children_left[i] == children_right[i]: # this is a leaf
            label = str(np.argmax(clf.tree_.value[i]))
            v.node(node_id, label, {'shape': 'rect', 'style': 'filled', 'fillcolor': fill_color})

        else: # this is a split node
            node_feature = 'X[' + str(feature[i]) + ']'
            node_tresshold = threshold[i]

            label = str(node_feature)
            v.node(node_id, label, {'shape': 'rect', 'style': 'filled', 'fillcolor': fill_color})

            v.edge(node_id, str(children_left[i]), node_feature + "<=" + str(node_tresshold))
            v.edge(node_id, str(children_right[i]), node_feature + ">" + str(node_tresshold))

    v.render(directory='doctest-output', view=True, format='jpeg')  # In collab: needs to be return v

