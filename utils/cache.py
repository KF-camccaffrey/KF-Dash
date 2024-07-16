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

cache = Cache(dash.get_app().server, config={
    "CACHE_TYPE": "filesystem", # will not work on systems with ephemeral filesystems like Heroku
    "CACHE_DIR": "cache",
    "CACHE_THRESHOLD": 5  # maximum number of users on the app at a single time
})

def get_data(session_id, params=None, upload=None, filename=None):
    print(f"get_data call:\n\tsession_id: {session_id}\n\tparams: {params}\n\tupload: {upload}\n\tfilename: {filename}")

    @cache.memoize()#make_name=make_cache_key)
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


    if params is None and upload is None:
        print(f"My session_id: {session_id}")
        df1, timestamp1 = create_data(session_id)
        df2, timestamp2 = store_data(session_id)

        if df1 is None and df2 is None:
            return None
        else:
            df = df1 if timestamp1 > timestamp2 else df2
            df = pd.read_json(StringIO(df))
            print(f"df 3:\n{df}")
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
