
"""
File Name: cache.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains Flask cache memoization functions so that data can be reused between callbacks without recalculation
    - Memoization caches the output of a function for each set of unique parameters.
    - For parameters, we will be using a "session_id" defined in app.py and stored in "session", a dcc.Store component (also in app.py)
    - Main Functions:
        - query_data() - store/retrieve the full dataframe
            - stored: page 1
            - retrieved: page 2-6
        - query_comparisons() - store/retrieve intermediate calculations and statistics
            - stored: page 2
            - retrieved: page 3-6
        - query_model() - store/retrieve linear model estimates and residuals
            - stored: page 6
            - retrieved: page 6
"""

import numpy as np
import pandas as pd
import datetime
from flask_caching import Cache
import uuid
from utils import generator
import base64
import io
import dash
from io import StringIO
from utils import comparisons
import statsmodels.api as sm

# define cache configuration
cache = Cache(dash.get_app().server, config={
    "CACHE_TYPE": "filesystem", # will not work on systems with ephemeral filesystems like Heroku
    "CACHE_DIR": "cache",
    "CACHE_DEFAULT_TIMEOUT": 3000,
    "CACHE_THRESHOLD": 5  # maximum number of users on the app at a single time
})

# define custom exception for error handeling
class InvalidInputError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# utility function for retrieving cache keys
def get_key(func, *args, **kwargs):
    return func.make_cache_key(func.uncached, *args, **kwargs)

# completely clear cache for troubleshooting (only called when the recycle button in header is clicked)
def clear_cache():
    cache.clear()
    return

# store/retrieve full dataframe
    # session_id - a session ID defined and managed in app.py
    # params - dictionary of parameters for synthetic data generation (for storing data only)
    # upload - uploaded data for when we are using data from file (for storing data only)
    # filename - filename for when we are using data from file (for storing data only)
    # check - if True, simply check if data already exists for this specific session_id (for retrieval only)
def query_data(session_id, params=None, upload=None, filename=None, check=False):
    # check if data needs to be (over) written
    doOverWrite = not (params is None and (upload is None or filename is None)) and not check

    @cache.memoize()
    def create_data(session_id):
        if not doOverWrite:
            raise InvalidInputError("doOverWrite is False")

        print(f"Session {session_id}: creating data...")

        # store data from file
        if upload is not None and filename is not None:
            print(f"Storing data from file")
            try:
                content_type, content_string = upload.split(",")
                decoded = base64.b64decode(content_string)
                if "csv" in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                elif "xls" in filename:
                    df = pd.read_excel(io.BytesIO(decoded))
                else:
                    raise InvalidInputError("Invalid file type.")
                return df.to_json(), datetime.datetime.now()
            except Exception as e:
                print(f"There was an error when storing the file data: {e}")
                raise

        # generate and save synthetic data
        if params is not None:
            print(f"Creating synthetic data")
            try:
                sample_size = params["sample_size"]
                gender_ratio = params["gender_ratio"]
                gender_gap = params["gender_gap"]
                np.random.seed(42)
                df = generator.generate_dataset(N=sample_size, ratio=gender_ratio, gap=gender_gap)
                return df.to_json(), datetime.datetime.now()
            except Exception as e:
                print(f"There was an error when creating synthetic data: {e}")
                raise

        # input was something else unexpected
        raise InvalidInputError("'params' and one of 'upload' or 'filename' were None")

    # get cache key
    key = get_key(create_data, session_id)

    # return cache status if optional arg 'check' is true
    if check:
        return cache.has(key)

    # delete existing cached data if it needs to be overwritten
    if doOverWrite:
        cache.delete(key)

    # call create_data() and convert json to dataframe
    try:
        data, timestamp = create_data(session_id)
        return pd.read_json(StringIO(data)), timestamp
    except InvalidInputError as e:
        print(f"InvalidInputError during caching, going to fallback. Message: {e}")
        return None, datetime.datetime.min


# store/retrieve intermediate calculations
    # session_id - a session ID defined and managed in app.py
    # comps - dictionary of parameters specifying variables of interest (for storing data only)
    # check - if True, simply check if data already exists for this specific session_id (for retrieval only)
def query_comparisons(session_id, comps=None, check=False): # timestamp

    @cache.memoize()
    def get_comparisons(session_id): # timestamp
        if comps is None:
            raise InvalidInputError("'comps' was None")
        else:
            df, _ = query_data(session_id)
            result = comparisons.create_comparisons(df, "pay", "gender", comps)
            return result

    key = get_key(get_comparisons, session_id) # timestamp

    if check:
        return cache.has(key)

    if comps is not None:
        cache.delete(key)

    try:
        data = get_comparisons(session_id) # timestamp
        return data
    except Exception as e:
        print(f"Error while querying comparisons. Message: {e}")
        return None


# store/retrieve model estimates and residuals
    # session_id - a session ID defined and managed in app.py
    # y - post-processed response variable column as input to model (for storing data only)
    # X - post-processed (predictor) design matrix as input to model (for storing data only)
    # check - if True, simply check if data already exists for this specific session_id (for retrieval only)
def query_model(session_id, y=None, X=None, check=False): # timestamp

    @cache.memoize()
    def get_model(session_id): # timestamp
        if y is None:
            raise InvalidInputError("'y' was None")

        if X is None:
            raise InvalidInputError("'X' was None")

        model = sm.OLS(y, X).fit()

        result = dict(
            n=model.nobs,
            p=len(model.params),

            # overall stats
            r2=model.rsquared,
            ar2=model.rsquared_adj,
            fstat=model.fvalue,
            pval=model.f_pvalue,
            aic=model.aic,
            bic=model.bic,

            # coefficients
            pred=np.array(model.params.index),
            beta=model.params.values,
            bse=np.array(model.bse),

            # residuals
            fit=np.array(model.fittedvalues),
            res=np.array(model.resid),
            stu=model.get_influence().resid_studentized_internal,
            lev=model.get_influence().hat_matrix_diag,
            cookd=model.get_influence().cooks_distance[0],
        )
        return result

    key = get_key(get_model, session_id) # timestamp

    if check:
        return cache.has(key)

    if y is not None and X is not None:
        cache.delete(key)

    try:
        model = get_model(session_id) # timestamp
        return model
    except Exception as e:
        print(f"Error while querying model. Message: {e}")
        return None
