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

cache = Cache(dash.get_app().server, config={
    "CACHE_TYPE": "filesystem", # will not work on systems with ephemeral filesystems like Heroku
    "CACHE_DIR": "cache",
    "CACHE_DEFAULT_TIMEOUT": 3000,
    "CACHE_THRESHOLD": 5  # maximum number of users on the app at a single time
})

# define custom exception
class InvalidInputError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def get_key(func, *args, **kwargs):
    return func.make_cache_key(func.uncached, *args, **kwargs)

def query_data(session_id, params=None, upload=None, filename=None, check=False):
    # check if data needs to be overwritten
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



def query_comparisons(session_id, timestamp, comps=None, check=False):

    @cache.memoize()
    def get_comparisons(session_id, timestamp):
        if comps is None:
            raise InvalidInputError("'comps' was None")
        else:
            df, _ = query_data(session_id)
            result = comparisons.create_comparisons(df, "pay", "gender", comps)
            return result

    key = get_key(get_comparisons, session_id, timestamp)

    if check:
        return cache.has(key)

    if comps is not None:
        cache.delete(key)

    try:
        data = get_comparisons(session_id, timestamp)
        return data
    except Exception as e:
        print(f"Error while querying comparisons. Message: {e}")
        return None



def query_model(session_id, timestamp, y=None, X=None, check=False):

    @cache.memoize()
    def get_model(session_id, timestamp):
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
            pred=model.params.index,
            beta=model.params.values,
            bse=model.bse,

            # residuals
            fit=model.fittedvalues,
            res=model.resid,
            stu=model.get_influence().resid_studentized_internal,
            lev=model.get_influence().hat_matrix_diag,
            cookd=model.get_influence().cooks_distance[0],
        )
        return result

    key = get_key(get_model, session_id, timestamp)

    if check:
        return cache.has(key)

    if y is not None and X is not None:
        cache.delete(key)

    try:
        model = get_model(session_id, timestamp)
        return model
    except Exception as e:
        print(f"Error while querying comparisons. Message: {e}")
        return None






def get_data(session_id, params=None, upload=None, filename=None, check=False):
    print(f"get_data call:\n\tsession_id: {session_id}\n\tparams: {params}\n\tupload: {upload}\n\tfilename: {filename}")

    @cache.memoize()
    def create_data(session_id):
        print(f"Session {session_id}: creating data...")
        timestamp = datetime.datetime.now()

        if params is None:
            print(f"'params' was None.")
            return None, datetime.datetime.min

        try:
            sample_size = params["sample_size"]
            gender_ratio = params["gender_ratio"]
            gender_gap = params["gender_gap"]
            np.random.seed(42)
            df = generator.generate_dataset(N=sample_size, ratio=gender_ratio, gap=gender_gap)

        except Exception as e:
            print(f"There was an error when creating the synthetic data: {e}")
            return None, datetime.datetime.min

        else:
            return df.to_json(), timestamp

    @cache.memoize()#make_name=make_cache_key)
    def store_data(session_id):
        print(f"Session {session_id}: storing data...")
        timestamp = datetime.datetime.now()

        if upload is None or filename is None:
            print(f"'upload' or 'filename' was None.")
            return None, datetime.datetime.min

        try:
            content_type, content_string = upload.split(",")
            decoded = base64.b64decode(content_string)
            if "csv" in filename:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif "xls" in filename:
                df = pd.read_excel(io.BytesIO(decoded))

        except Exception as e:
            print(f"There was an error when storing the file data: {e}")
            return None, datetime.datetime.min
        else:
            return df.to_json(), timestamp

    def is_cached(func, *args, **kwargs):
        key = func.make_cache_key(func.uncached, *args, **kwargs)
        return cache.has(key)

    if check:
        print("FLAG 1")
        check1 = is_cached(create_data, session_id)
        check2 = is_cached(store_data, session_id)
        return check1 or check2

    if params is None and upload is None:
        print(f"My session_id: {session_id}")
        df1, timestamp1 = create_data(session_id)
        df2, timestamp2 = store_data(session_id)

        if df1 is None and df2 is None:
            print("FLAG 2")
            cache.delete_memoized(create_data, session_id)
            cache.delete_memoized(store_data, session_id)
            return None
        else:
            df = df1 if timestamp1 > timestamp2 else df2
            df = pd.read_json(StringIO(df))
            print(f"df 3:\n{df}")
            print("FLAG 3")
            return df

    elif upload is not None:

        cache.delete_memoized(store_data, session_id)

        df, timestamp = store_data(session_id)
        df = pd.read_json(StringIO(df))
        print(f"df upload:\n{df}")
        return df

    else: # params is not None
        cache.delete_memoized(create_data, session_id)

        df, timestamp = create_data(session_id)
        df = pd.read_json(StringIO(df))
        print(f"df create: {df}")
        return df



""" EXAMPLE:
def get_dataframe(session_id):
    @cache.memoize()
    def query_and_serialize_data(session_id):
        now = datetime.datetime.now()

        x = np.random.normal(80000, 10000, shape=100)
        y = np.random.normal(60000, 10000, shape=100)

        df = pd.DataFrame({"x":x, "y":y})
        return df.to_json()

    return pd.read_json(query_and_serialize_data(session_id))
"""
