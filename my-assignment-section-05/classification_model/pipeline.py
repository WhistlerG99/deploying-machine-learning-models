from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

titanic_pipe = Pipeline(
    [
        # impute categorical variables with string missing
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                fill_value="Missing",
                variables=config.model_config.categorical_vars,
            )
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(
                variables=config.model_config.numerical_vars
            )
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars
            )
        ),
        # Extract letter from cabin
        (
            "extract_letter",
            pp.ExtractLetterTransformer(
                variables=config.model_config.cabin_vars,
                fill_value="Missing"
            )
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05,
                n_categories=1,
                variables=config.model_config.categorical_vars
            )
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True,
                variables=config.model_config.categorical_vars
            )
        ),
        (
            'yeojohnson',
            YeoJohnsonTransformer(
                variables=config.model_config.yeojohnson_vars
            )
        ),
        # scale
        (
            'scaler',
            SklearnTransformerWrapper(
                transformer=StandardScaler(),
                variables=config.model_config.yeojohnson_vars
            )
        ),
        (
            "Logit",
            LogisticRegression(
                C=config.model_config.alpha,
                random_state=config.model_config.random_state
            )
        ),
    ]
)
