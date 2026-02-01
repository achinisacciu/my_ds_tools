# ==============================================================================
# 0. INTERNAL MODULES EXPORT
# ==============================================================================
from . import data
from . import eda
from . import ml
from . import stats
from . import theme
from . import viz

# ==============================================================================
# 1. SYSTEM & DATA MANIPULATION LIBRARIES
# ==============================================================================
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import kagglehub
from IPython.display import display, Markdown

# ==============================================================================
# 2. DATA VISUALIZATION
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio

# ==============================================================================
# 3. STATISTICS & ANALYSIS
# ==============================================================================
import scipy
from scipy.stats import (
    chi2_contingency, mannwhitneyu, kruskal, 
    uniform, randint, loguniform
)
import statsmodels.api as sm

# ==============================================================================
# 4. PREPROCESSING & SCALING
# ==============================================================================
import sklearn
from sklearn import set_config
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, 
    RobustScaler, QuantileTransformer, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ==============================================================================
# 5. PIPELINES & MODEL SELECTION
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    StratifiedKFold, RandomizedSearchCV
)

# ==============================================================================
# 6. MACHINE LEARNING MODELS (SKLEARN)
# ==============================================================================
# Linear Models
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, 
    SGDClassifier, PassiveAggressiveClassifier
)
# Support Vector Machines
from sklearn.svm import SVC, LinearSVC, NuSVC
# Neighbors
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# Decision Trees
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
# Ensemble Methods
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    BaggingClassifier, AdaBoostClassifier, 
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
# Discriminant Analysis
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
# Neural Networks (MLP)
from sklearn.neural_network import MLPClassifier

# ==============================================================================
# 7. EVALUATION METRICS
# ==============================================================================
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve

# ==============================================================================
# 8. EXTERNAL BOOSTING LIBRARIES (SOTA)
# ==============================================================================
# Conditional imports to handle environments where these might be missing
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    xgb = None
    XGBClassifier = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    lgb = None
    LGBMClassifier = None

try:
    import catboost as cb
    from catboost import CatBoostClassifier
except ImportError:
    cb = None
    CatBoostClassifier = None

# ==============================================================================
# 9. UTILITIES & EXPLAINABILITY
# ==============================================================================
import joblib
import tqdm
from tqdm.auto import tqdm

try:
    import shap
except ImportError:
    shap = None

try:
    import imblearn
except ImportError:
    imblearn = None

# ==============================================================================
# 10. GLOBAL CONFIGURATION
# ==============================================================================
# Ignore non-critical warnings
warnings.filterwarnings('ignore')
# Pandas display options
pd.set_option('display.max_columns', None)

# ==============================================================================
# EXPORT LIST (__all__)
# ==============================================================================
# This defines what is exported when someone does "from my_tools import *"
__all__ = [
    # Internal Modules
    'data', 'eda', 'ml', 'stats', 'theme', 'viz',

    # System & Data
    'os', 'sys', 'time', 'warnings', 'np', 'pd', 'kagglehub', 'display', 'Markdown',

    # Visualization
    'plt', 'sns', 'px', 'go', 'ff', 'make_subplots', 'pio', 'ListedColormap',

    # Statistics
    'scipy', 'sm', 'chi2_contingency', 'mannwhitneyu', 'kruskal',

    # Preprocessing
    'sklearn', 'OneHotEncoder', 'StandardScaler', 'MinMaxScaler', 
    'RobustScaler', 'QuantileTransformer', 'LabelEncoder', 'SimpleImputer',
    'ColumnTransformer', 'set_config',

    # Model Selection
    'Pipeline', 'train_test_split', 'cross_val_score', 
    'GridSearchCV', 'RandomizedSearchCV', 'StratifiedKFold',

    # Models (Sklearn)
    'LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'PassiveAggressiveClassifier',
    'SVC', 'LinearSVC', 'NuSVC',
    'KNeighborsClassifier', 'NearestCentroid',
    'GaussianNB', 'BernoulliNB',
    'DecisionTreeClassifier', 'ExtraTreeClassifier',
    'RandomForestClassifier', 'ExtraTreesClassifier', 'BaggingClassifier', 
    'AdaBoostClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
    'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis',
    'MLPClassifier',

    # Metrics
    'roc_curve', 'auc', 'roc_auc_score', 'precision_recall_curve', 
    'average_precision_score', 'confusion_matrix', 'classification_report',
    'accuracy_score', 'f1_score', 'precision_score', 'recall_score',
    'calibration_curve', 'mean_squared_error', 'r2_score',

    # Boosting & External
    'xgb', 'XGBClassifier',
    'lgb', 'LGBMClassifier',
    'cb', 'CatBoostClassifier',
    'shap', 'imblearn', 'joblib', 'tqdm'
]
