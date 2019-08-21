from collections import OrderedDict
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# general parameters
Nnbrs = 5
oversample_rate = 25 
color_map = 'rgb'
PLOT = False


# dataset definition
dataset = 'bank'
if dataset == 'iris':
    fin = "iris.csv"
    feature_nametypes = OrderedDict({
        'sepal_length': 'float',
        'sepal_width': 'float',
        'petal_length': 'float',
        'petal_width': 'float',
    })
    target_field = 'species'
    plot_ind_x, plot_ind_y = 0, 3
elif dataset == 'bank':
    # https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    fin = 'bank-additional-full.csv'
    fin_enum = 'bank-additional-full-enum.csv'
    fout = 'bank-synthetic-%dx.csv' % (oversample_rate)
    feature_nametypes = OrderedDict({
        'age': 'int',
        'job': 'enum',
        'marital': 'enum',
        'education': 'enum',
        'credit default': 'enum',
        'housing': 'enum',
        'loan': 'enum',
        'contact': 'enum',
        'month': 'enum',
        'day_of_week': 'enum',  # TODO: 'random' type, to select this randomly from the full range
        'duration': 'int',
        'campaign': 'int',
        'pdays': 'int',
        'previous': 'int',
        'poutcome': 'enum',
    })

    # need some way to find nearest neighbors for enum fields...
    # easiest thing to do is to map them to enums, so try to do that manually in a way that makes sense
    feature_enums = {
        'job': {'unemployed': -4, 'retired': -3, 'student': -2, 'self-employed': -1,
                'unknown': 0, 'hospitality': 1, 'blue-collar': 2, 'entrepreneur': 3,
                'chef': 4, 'admin.': 5, 'management': 6, 'engineer': 7},
        'marital': {'divorced': -2, 'single': -1, 'unknown': 0, 'married': 1},
        'education': {'partial high school': -2, 'GED': -1, 'unknown': 0,
                      'high school diploma': 1, 'vocational diploma': 2,
                      'professional course': 3, 'university degree': 4, 'masters degree': 5},
        'credit default': {'no': -1, 'unknown': 0, 'yes': 1},
        'housing': {'no': -1, 'unknown': 0, 'yes': 1},
        'loan': {'no': -1, 'unknown': 0, 'yes': 1},
        'contact': {'telephone': 0, 'email': 1},
        'month': {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
                  'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},
        'day_of_week': {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4},
        'poutcome': {'failure': -1, 'nonexistent': 0, 'success': 1},
    }
    target_field = 'y'
    plot_ind_x, plot_ind_y = 0, 10


# derived params
feature_names = [x for x in feature_nametypes.keys()]
num_features = len(feature_nametypes)


def main():
    df = pd.read_csv(fin)
    # enum_df = enumerate_categories(df, feature_enums)
    enum_df = pd.read_csv(fin_enum)

    new_data = smote(df, enum_df, feature_nametypes, target_field)
    # new_data.to_csv(fout, index=False)
    new_data.to_csv(fout)

    if not PLOT:
        return

    for color_id, cls in enumerate(df[target_field].unique()):
        class_df = df[df[target_field] == cls]  # select rows for this class
        class_features = np.array(class_df[feature_names])  # select specified columns
        plt.plot(class_features[:, plot_ind_x], class_features[:, plot_ind_y], color_map[color_id] + '.')

        new_class_df = new_data[new_data[target_field] == cls]
        new_class_features = np.array(new_class_df[feature_names])
        plt.plot(new_class_features[:, plot_ind_x], new_class_features[:, plot_ind_y], color_map[color_id] + 'x')
        plt.show()


def smote(df, enum_df, feature_nametypes, target_field):
    # use SMOTE-like algorithm to generate reasonable-looking synthetic data from a base dataset.
    # designed to work well with integer-valued numeric, enum-valued integer, and enum-valued string fields.
    # this feature requires a nonstandard implementation of the KNN component.
    # not intended to be statistically robust, but just to produce data that is reasonable-looking, and
    # maintains basic correlations between fields.
    # expected to be a slight improvement in that respect, compared to generating new rows where each field value is
    # drawn from its respective marginal distribution
    #
    # see also:
    # https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html
    # https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
    #
    # input:
    #   df = dataframe of feature and target values
    #   feature_nametypes = dict of {field_name: field_type, ...}. field_type controls KNN distance metric, and combination method
    #   target_field = name of dependent variable field
    # output:
    #   new_data = dataframe of similar shape to input df
    # globals:
    #   oversample_rate
    #   Nnbrs (only used in get_neighbors_X)

    new_data = np.zeros(shape=(0, num_features+1))
    # TODO looping over classes causes the output data to be sorted by class. possibilities to fix:
    # - shuffle the results (requires moving around a potentially large number of records)
    # - in each iteration, choose a random original sample for generating a new sample (requires maintaining nearest neighbor data)
    #   - can do this by shuffling an index array, which might be more efficient
    for color_id, cls in enumerate(df[target_field].unique()):
        class_df = df[df[target_field] == cls]  # select rows for this class
        class_features = np.array(class_df[feature_names])  # select specified columns
        enum_class_df = enum_df[enum_df[target_field] == cls]
        enum_class_features = np.array(enum_class_df[feature_names])
        print('oversampling %d values from class %s="%s"' % (oversample_rate*len(class_df), target_field, cls))

        # get `Nnbrs` neighbors
        start = time.time()
        NNindices = get_neighbors_sklearn(enum_class_features, Nnbrs)  # requires numeric_data
        print('  found nearest neighbors (%f sec)' % (time.time() - start))
        # NNindices = get_neighbors_naive(class_features, Nnbrs, feature_nametypes)  # too slow for 1e5 values

        new_class_data = np.zeros(shape=(0, num_features+1))
        for k in range(oversample_rate):
            # select one random neighbor
            random_neighbor_indirect = np.random.randint(1, Nnbrs+1, size=(len(class_df),))  # in [1, 5]
            random_neighbor_id = NNindices[np.arange(len(NNindices)), random_neighbor_indirect]  # in [0, len(class_def))

            # compute random combination of sample and neighbor
            # TODO might get nicer data by selecting k out of n nearest neighbors, k>1, then using a combination of all k of those points
            alpha = np.random.rand(len(class_df),)  # linear combination factor
            new = np.zeros(shape=(len(class_df),))  # FIXME shouldn't need to start with actual zero values when building the array
            for name, typ in feature_nametypes.items():  # for each column
                if k == 0:
                    print('  generating %d x %d rows for %s field "%s"' % (oversample_rate, len(class_df), typ, name))
                col = class_df[name].values
                neighbors = col[random_neighbor_id]
                if typ == 'float':
                    # linear combination for floats
                    new_col = alpha * col + (1-alpha) * neighbors
                if typ == 'int':
                    # rounded linear combination for ints
                    new_col = np.round(alpha * col + (1-alpha) * neighbors)
                else:
                    # general combination for all other types (enum strings)
                    # new_col = np.where(alpha > .5, col, neighbors)  # select nearest
                    dice = np.random.rand(len(class_df),)  # select base sample with probability alpha
                    new_col = np.where(alpha < dice, col, neighbors)
                new = np.vstack((new, new_col))  # FIXME should be hstacking column vectors
            new = np.vstack((new, np.repeat(cls, len(class_df))))
            new_class_data = np.vstack((new_class_data, new[1:, :].T))  # FIXME shouldn't need to index and transpose
        new_data = np.vstack((new_data, new_class_data))

    columns = [x for x in feature_nametypes.keys()] + [target_field]
    new_df = pd.DataFrame(new_data, columns=columns)

    return new_df


def get_neighbors_sklearn(data, K, feature_nametypes=None):
    # only works for numeric data
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices


def get_neighbors_categorical_kdtree(data, K, feature_nametypes):
    # TODO: use something like a k-d tree, but that works for categorical data.
    # this is a vague notion, but it makes sense in my head.
    # idea: categorical axes 
    pass


def get_neighbors_hybrid(data, K, feature_nametypes):
    numeric_idx = [n for n, kv in enumerate(feature_nametypes.items()) if kv[1] != 'enum']
    numeric_data = data[:, numeric_idx]
    numeric_nbrs = NearestNeighbors(n_neighbors=4*K+1, algorithm='ball_tree').fit(numeric_data)
    numeric_distances, numeric_indices = numeric_nbrs.kneighbors(numeric_data)


def get_neighbors_naive(data, K, feature_nametypes):
    # compute indices to the `K` nearest neighbors of each row in `data`.
    # use `feature_nametypes` to determine what distance metric to use.
    # works with float, int, enum-valud int, enum-valued string.
    # could be extended to work with text-valued string (edit distance as the metric).
    nearest_indices = np.zeros((len(data), Nnbrs+1), dtype=int)
    for i in range(len(data)):
        nearest_ids, nearest_dists = [-1], [1e12]
        for j in range(len(data)):
            # dont want to bother to compute this, but easier to match sklearn implementation
            # if i == j:
            #    continue
            dist2 = 0
            for n, (name, typ) in enumerate(feature_nametypes.items()):
                # compute distance with custom metrics
                if typ in ['float', 'int']:
                    dist2 += dist_euclidean(data[i][n], data[j][n])
                else:
                    dist2 += dist_enum(data[i][n], data[j][n])

            # insert into sorted list
            # TODO implement something to handle this operation properly
            if dist2 < nearest_dists[-1]:
                k = len(nearest_dists)-1
                while dist2 < nearest_dists[k] and k >= 0:
                    k -= 1
                nearest_dists = nearest_dists[:k+1] + [dist2] + nearest_dists[k+1:]
                nearest_ids = nearest_ids[:k+1] + [j] + nearest_ids[k+1:]
                if len(nearest_dists) > Nnbrs+1:
                    nearest_dists = nearest_dists[:Nnbrs+1]
                    nearest_ids = nearest_ids[:Nnbrs+1]
        nearest_indices[i] = nearest_ids

    return nearest_indices


def dist_euclidean(x, y):
    return (x-y) ** 2


def dist_enum(x, y, scale=1):
    if x == y:
        return 0
    return scale


def enumerate_categories(df, feature_maps):
    enum_df = df.copy(deep=True)
    print('converting categorical fields...')
    start = time.time()
    for name, enum_map in feature_maps.items():
        for n in range(len(df)):
            enum_df[name][n] = enum_map[enum_df[name][n]]
            if name == 'job' and (n+1) % 1000 == 0:
                print('%d/%d, %f sec' % (n+1, len(df), time.time()-start))



main()
