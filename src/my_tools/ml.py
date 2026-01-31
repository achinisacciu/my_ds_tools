"""
Machine Learning Module
Consolidates imports for Scikit-Learn, Boosting libraries (XGBoost, LightGBM, CatBoost), 
metrics, and preprocessing tools. Designed for rapid experimentation.
"""

# ==============================================================================
# PREPROCESSING & SCALING
# ==============================================================================
from sklearn import set_config
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, 
    RobustScaler, QuantileTransformer, LabelEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer

# ==============================================================================
# PIPELINE & MODEL SELECTION
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    RandomizedSearchCV, GridSearchCV, learning_curve
)

# ==============================================================================
# MACHINE LEARNING MODELS (SKLEARN)
# ==============================================================================
# Linear Models
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier, 
    LinearRegression, Lasso, Ridge
)
# Support Vector Machines
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR
# Neighbors
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KNeighborsRegressor
# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# Decision Trees
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor
# Ensemble Methods
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Neural Networks (MLP)
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    classification_report, mean_squared_error, r2_score, mean_absolute_error,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

# ==============================================================================
# EXTERNAL LIBRARIES (Optional Imports)
# ==============================================================================

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

# Model Explainability
try:
    import shap
except ImportError:
    shap = None

# Gradient Boosting (SOTA)
# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None

# CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = CatBoostRegressor = None

# ==============================================================================
# EXPORT
# ==============================================================================
__all__ = [
    # Preprocessing
    'set_config', 'OneHotEncoder', 'StandardScaler', 'MinMaxScaler', 
    'RobustScaler', 'QuantileTransformer', 'LabelEncoder', 'SimpleImputer', 'KNNImputer',
    
    # Pipeline
    'Pipeline', 'ColumnTransformer', 'train_test_split', 'cross_val_score', 
    'StratifiedKFold', 'KFold', 'RandomizedSearchCV', 'GridSearchCV', 'learning_curve',
    
    # Linear & SVM
    'LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'PassiveAggressiveClassifier',
    'LinearRegression', 'Lasso', 'Ridge', 'SVC', 'LinearSVC', 'NuSVC', 'SVR',
    
    # Neighbors & Bayes
    'KNeighborsClassifier', 'NearestCentroid', 'KNeighborsRegressor', 'GaussianNB', 'BernoulliNB',
    
    # Trees & Ensembles
    'DecisionTreeClassifier', 'ExtraTreeClassifier', 'DecisionTreeRegressor',
    'RandomForestClassifier', 'ExtraTreesClassifier', 'BaggingClassifier',
    'AdaBoostClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
    'RandomForestRegressor', 'GradientBoostingRegressor',
    
    # Others
    'LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'MLPClassifier', 'MLPRegressor',
    
    # Metrics
    'roc_curve', 'auc', 'precision_recall_curve', 'average_precision_score',
    'confusion_matrix', 'accuracy_score', 'f1_score', 'precision_score', 'recall_score',
    'classification_report', 'mean_squared_error', 'r2_score', 'mean_absolute_error',
    'ConfusionMatrixDisplay', 'calibration_curve',
    
    # External Libs
    'SMOTE', 'shap',
    'XGBClassifier', 'XGBRegressor',
    'LGBMClassifier', 'LGBMRegressor',
    'CatBoostClassifier', 'CatBoostRegressor'
]
