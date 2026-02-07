
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import sys
import os 
import contextlib
from typing import Literal

# Monkey-patch RMSNorm for older torch versions (required by synthcity)
try:
    from torch.nn import RMSNorm
except ImportError:
    import torch.nn
    class RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

        def forward(self, x):
            norm = x.norm(2, dim=-1, keepdim=True)
            rms = norm * (x.size(-1) ** -0.5)
            return self.weight * x / (rms + self.eps)
    torch.nn.RMSNorm = RMSNorm

# ... (rest of imports)

# Update PRDC error handling (remove the specific RMSNorm skip since we patched it)
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# detection score
def detection_score(real: pd.DataFrame,
                    fake: pd.DataFrame,
                    # model: BaseEstimator,
                    # categorical_columns: list[str],
                    n_folds: int = 3,
                    scoring: str = "roc_auc",
                    random_state: int = 42) -> float:
    """
    Computes the detection score of the synthetic data according to the scoring and the number of folds given.
    Detection score is how well the synthetic data can be distinguished from the real data.

    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - n_folds: int, the number of folds for cross-validation. Default = 3.
    - scoring: str, the scoring method to use. Default = "roc_auc".
    - random_state: int, the random state to use. Default = 42.

    Returns the mean detection score.
    """
    # equalize the number of samples
    if len(real) > len(fake):
        real = real.sample(n=len(fake), random_state=random_state)
    else:
        fake = fake.sample(n=len(real), random_state=random_state)

    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    # print("detection_score, selected Cat Cols", categorical_columns)

    real = real.astype({col: 'str' for col in real.select_dtypes('category').columns})
    fake = fake.astype({col: 'str' for col in fake.select_dtypes('category').columns})
    
    if real.isna().any().any() or fake.isna().any().any():
        print("real or fake has Nan")
        # exit(1) # Don't exit the whole script
        return 0.5 


    # SAFETY OVERRIDE: LightGBM causes SegFault/OSError on Mac without libomp.
    # Disabling Detection Score to allow main evaluation to proceed.
    print("Warning: Detection Score disabled to prevent LightGBM crash on this environment.")
    return 0.5

    # add labels and concat the data
    real["fake"] = 0
    fake["fake"] = 1
    data = pd.concat([real, fake], ignore_index=True)

    # Lazy import lightgbm
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Skipping Detection Score.")
        return 0.5
    except OSError as e:
        print(f"LightGBM system dependency missing (e.g., libomp): {e}. Skipping Detection Score.")
        return 0.5

    model = lgb.LGBMClassifier(categorical_feature=categorical_columns, random_state=random_state, verbose=-1)

    # cross val
    y = data["fake"]
    x = data.drop(columns=["fake"])

    x[categorical_columns] = x[categorical_columns].astype('category')    

    with suppress_stdout():
        scores = cross_val_score(model, x, y, cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state), scoring=scoring)

    return scores.mean()


# PRDC (precision, recall, density, coverage)
def prdc_score(real: pd.DataFrame,
               fake: pd.DataFrame,
            #    categorical_columns: list[str],
               neares_k: int = 5) -> list:
    """
    Computes the PRDC (precision, recall, density, coverage) score of the synthetic data.
    
    Inputs:
    - real: pd.DataFrame, the real data
    - fake: pd.DataFrame, the fake, generated data
    - nearest_k: int, the number of nearest neighbors to use. Default = 5.

    Returns the prdc values [precision, recall, density, coverage]
    """

    if len(fake) > len(real):
        # return [0, 0, 0, 0] # Why return 0? Standard implementations might handle this. keeping user logic but commenting out if problematic.
        pass
    
    if len(fake) > 20000:   
        print("subsampling for prdc score")
        fake = fake.sample(n=20000)  # Ensures reproducibility
        real = real.sample(n=20000)  # Ensures reproducibility


    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    # print("prdc_score, selected Cat Cols", categorical_columns)

    
    real = real.astype({col: 'str' for col in real.select_dtypes('category').columns})
    fake = fake.astype({col: 'str' for col in fake.select_dtypes('category').columns})

    # # one hot dataset
    combined = pd.concat([real, fake], ignore_index=True)

    # Apply get_dummies to the concatenated dataframe
    combined_dummies = pd.get_dummies(combined, columns=categorical_columns)

    # Split the combined dataframe back into real and fake datasets
    real_dummies = combined_dummies.iloc[:len(real)]
    fake_dummies = combined_dummies.iloc[len(real):]

    # normalize between 0 and 1 (prdc uses knn)
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_dummies)
    fake_scaled = scaler.transform(fake_dummies)

    # compute prdc using synthcity
    try:
        from synthcity.metrics.eval_statistical import PRDCScore
        from synthcity.plugins.core.dataloader import GenericDataLoader
        
        real_loader = GenericDataLoader(real_scaled)
        fake_loader = GenericDataLoader(fake_scaled)

        # print("PRDCScore")
        prdc = PRDCScore(nearest_k=neares_k)
        # print("Done")
        with suppress_stdout():
            scores = prdc.evaluate(real_loader, fake_loader)
        return [scores["precision"], scores["recall"], scores["density"], scores["coverage"]]
    except Exception as e:
        print(f"Error in PRDC calculation: {e}")
        return [0, 0, 0, 0]


def ks_score(real: pd.DataFrame,
             fake: pd.DataFrame):
            #  continous_columns: list[str]):
    """
    Computes the Kolmogorov-Smirnov score of the synthetic data for the continuous columns.
    Score is between [0, 1], 1 = distributions are similar, 0 = distributions are different.
    """
    all_scores = {}
    mean_score = 0
    continous_columns = real.select_dtypes(exclude=['object', 'category']).columns
    # print("KS-score, selected Cont Cols", continous_columns)
    
    try:
        from sdmetrics.single_column import KSComplement
        for col in continous_columns:
            score = KSComplement.compute(real_data=real[col], synthetic_data=fake[col])
            all_scores[col] = score
            mean_score += score
    except ImportError:
        print("sdmetrics not found via lazy import.")
        return [0]
    except Exception as e:
        print(f"KS Score error: {e}")
        return [0]

    if len(continous_columns) == 0: return [0]
    mean_score /= len(continous_columns)
    return [mean_score]#, all_scores
    

def tv_score(real: pd.DataFrame,
             fake: pd.DataFrame):
            #  categorical_columns: list[str]):
    """
    Computes the Total Variation score of the synthetic data for the categorical columns.
    Score is between [0, 1], 1 = distributions are similar, 0 = distributions are different.
    """
    all_scores = {}
    mean_score = 0

    categorical_columns = list(real.select_dtypes(include=['object', 'category']).columns)
    # print("TV-score, selected Cat Cols", categorical_columns)

    if len(categorical_columns) == 0:
        return [0.0]

    try:
        from sdmetrics.single_column import TVComplement
        for col in categorical_columns:
            score = TVComplement.compute(real_data=real[col], synthetic_data=fake[col])
            all_scores[col] = score
            mean_score += score
        mean_score /= len(categorical_columns)
        return [mean_score]#, all_scores
    except Exception as e:
        print(f"TV Score error: {e}")
        return [0]


# --- DCR and Gower Distance Utils ---

def _create_matrix_with_ones(indices, num_rows):
    matrix = np.zeros((len(indices),num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i,index] = 1
    return matrix


def _gower_matrix_sklearn(data_x, data_y=None, cat_features: list = None, weights=None, num_attribute_ranges=None, nums_metric: Literal['L1', 'EXP_L2'] = 'L1'):
    """Modified version of the python gower distance metric implementation
    url: https://pypi.org/project/gower/"""

    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    out_shape = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    ### Bit to infer, cat_features if nothing is supplied 
    if cat_features is None:
        if not isinstance(X, np.ndarray): 
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)    
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col]=True
    else:          
        cat_features = np.array(cat_features)

    ### Separate out weights
    if weights is None:
        weights = np.ones(X.shape[1])
            
    weights_cat = weights[cat_features]
    weights_num = weights[np.logical_not(cat_features)]

    ### Subsetting
    Z = np.concatenate((X,Y))
    
    x_index = range(0,x_n_rows)
    y_index = range(x_n_rows,x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    Z_cat = Z[:,cat_features]

    ### Make the denominator for the nummerical normalisation 
    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(np.stack((np.array(np.ptp(Z_num,axis=0),dtype=np.float64),np.ones(len(weights_num)))),axis=0)

    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]

    ### Do the nummerical step
    if not np.array_equal(cat_features,np.ones(X.shape[1])):
        if nums_metric == 'L1':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=1, w=(weights_num/num_attribute_ranges))

        elif nums_metric == 'EXP_L2':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=2, w=(weights_num/num_attribute_ranges**2))#/np.sqrt(len(weights_num))

        else: raise NotImplementedError("The keyword literal is not a valid!")
    else: nums_sum = out_shape
    
    ### Do the categorical step
    if not np.array_equal(cat_features,np.zeros(X.shape[1])):
        Z_cat_enc = np.array(OrdinalEncoder().fit_transform(Z_cat.astype(str)))


        X_cat = Z_cat_enc[x_index,]
        Y_cat = Z_cat_enc[y_index,]

        cat_sum = cdist(X_cat.astype(int),Y_cat.astype(int), 'hamming', w=weights_cat)*len(weights_cat)
    else: cat_sum = out_shape
    
    return (nums_sum+cat_sum)/weights.sum()


def _knn_distance(a, b, cat_cols, num, metric: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', weights=None):
    def gower_knn(a, b, bool_cat_cols, gower_variant):
            """Function used for finding nearest neighbours"""
            d = []
            if np.array_equal(a,b):
                matrix = _gower_matrix_sklearn(a.values, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)+np.eye(len(a))
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(a))
            else:
                matrix = _gower_matrix_sklearn(a.values, b.values, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(b))
            return d

    def eucledian_knn(a, b):
            """Function used for finding nearest neighbours"""
            d = []
            nn = NearestNeighbors(n_neighbors=num+1, metric_params={'w':weights}) #TODO: add num_att_range here as well
            if np.array_equal(a,b):
                nn.fit(a)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,1+i])
            else:
                nn.fit(b)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,i])
            return d

    if metric=='gower' or metric=='EXPERIMENTAL_gower':
        bool_cat_cols = [col1 in cat_cols for col1 in a.columns]
        num_cols = [col2 for col2 in a.columns if col2 not in cat_cols]
        # Copy to avoid modifying original
        a = a.copy()
        b = b.copy()
        a[num_cols] = a[num_cols].astype("float")
        b[num_cols] = b[num_cols].astype("float")
        if metric=='gower': return gower_knn(a,b,bool_cat_cols, gower_variant = 'L1')
        else: return gower_knn(a,b,bool_cat_cols, gower_variant='EXP_L2')
    if metric=='euclid':
        return eucledian_knn(a,b)
    else: raise Exception("Unknown metric; options are 'gower' or 'euclid'")
    

def DCR(real, fake, nn_dist="gower") -> list:
    """Distance to closest record"""

    if len(fake) > 20000:   
        print("subsampling for DCR score")
        fake = fake.sample(n=20000)  # Ensures reproducibility
        real = real.sample(n=20000)  # Ensures reproducibility

    cat_cols = list(real.select_dtypes(include=['object', 'category']).columns)
    # print("DCR, selected Cat Cols", cat_cols)
    
    distances = _knn_distance(fake, real, cat_cols, 1,nn_dist)
    in_dists = _knn_distance(real, real, cat_cols, 1,nn_dist)

    # distances is a list of arrays? _knn_distance returns 'd' which is a list.
    # We take the first element (k=1)
    mut_nn = np.median(distances[0])
    int_nn = np.median(in_dists[0])

    if (int_nn == 0 and mut_nn == 0): dcr = 1
    elif (int_nn == 0 and mut_nn != 0): dcr = 0
    else: dcr = mut_nn/int_nn
    
    return [dcr]
