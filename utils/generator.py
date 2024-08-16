
"""
File Name: generator.py
Author: Connor McCaffrey
Date: 8/16/2024

Description:
    - This file contains functions for generating synthetic data, utilized on page 1
    - Main function here is generate_dataset()
    - Helper functions include personal_info() and linear_predict()
"""

import numpy as np
import pandas as pd
from faker import Faker

job_family = [
    "Sales",
    "Marketing",
    "Finance",
    "Human Resources",
    "Information Technology (IT)",
    "Customer Service",
    "Research and Development (R&D)",
    "Operations",
    "Legal",
    "Product Management",
    "Quality Assurance (QA)",
    "Supply Chain",
    "Administration",
    "Business Development",
    "Public Relations (PR)"
]

metros = [
    ("New York City", "NY"),
    ("Los Angeles", "CA"),
    ("Chicago", "IL"),
    ("Dallas-Fort Worth", "TX"),
    ("Houston", "TX"),
    ("Washington, D.C.", "DC"),
    ("Miami", "FL"),
    ("Philadelphia", "PA"),
    ("Atlanta", "GA"),
    ("Boston", "MA"),
    ("Phoenix", "AZ"),
    ("San Francisco Bay Area", "CA"),
    ("Riverside-San Bernardino", "CA"),
    ("Detroit", "MI"),
    ("Seattle", "WA"),
    ("Minneapolis-St. Paul", "MN"),
    ("San Diego", "CA"),
    ("Tampa Bay Area", "FL"),
    ("Denver", "CO"),
    ("St. Louis", "MO"),
    ("Baltimore", "MD"),
    ("Charlotte", "NC"),
    ("Orlando", "FL"),
    ("San Antonio", "TX"),
    ("Portland", "OR"),
    ("Sacramento", "CA"),
    ("Pittsburgh", "PA"),
    ("Las Vegas", "NV"),
    ("Cincinnati", "OH"),
    ("Kansas City", "MO")
]

def personal_info(N, faker):
    n = len(metros)
    ind = np.random.randint(n, size=N)
    metro_arr = np.array(metros)
    city = metro_arr[ind, 0]
    state = metro_arr[ind, 1]

    department = np.random.choice(
        job_family,
        size=N,
    )

    eid, job, last = zip(*[(faker.bothify("???#??", "abcdefghijklmnopqrstuvwxyz"), faker.job(), faker.last_name()) for _ in range(N)])

    df = pd.DataFrame({"last": last, "eid": eid, "city": city, "state": state, "department": department, "job": job})
    return df


def generate_dataset(N=10000, ratio=0.5, gap=0, seed=42):
    data = []

    gender = np.random.choice(
        ["Male", "Female", "Other"],
        size=N,
        p=[0.9 * ratio, 0.9 * (1-ratio), 0.1],
    )

    race = np.random.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        size=N,
        p=[0.59, 0.13, 0.19, 0.06, 0.03],
    )

    age = np.random.choice(
        [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
        size=N,
        p=[0.032, 0.086, 0.104, 0.111, 0.112, 0.108, 0.101, 0.094, 0.091, 0.075, 0.044, 0.022, 0.012, 0.008],
    )

    age += np.random.randint(5, size=N)

    YoE = (age - 22 + np.random.randint(-5, 6, size=N)).clip(min=0)

    ageAdjust = age + np.random.normal(500, 50, size=N)
    eduCond = [ageAdjust > 650, ageAdjust > 600, ageAdjust > 525]

    education = np.select(
        condlist=eduCond,
        choicelist=["Doctorate", "Master", "Bachelor"],
        default="Secondary",
    )

    expAdjust = YoE + np.random.normal(0, 5, size=N)
    lvlCond = [expAdjust >= 40, expAdjust >= 20, expAdjust >= 10]
    level = np.select(
        condlist=lvlCond,
        choicelist=["Senior", "Mid", "Low"],
        default="Entry"
    )



    df = pd.DataFrame({
        "gender": gender,
        "race": race,
        "age": age,
        "YoE": YoE,
        "education": education,
        "level": level,
    })

    df_dummies = pd.get_dummies(
        df,
        columns=["gender", "education", "level"],
        prefix=["gen", "edu", "lvl"],
        dtype=int,
    )
    df_dummies = df_dummies.drop(
        columns=["gen_Male", "edu_Secondary", "lvl_Entry"]
    )

    coeffs = {
        "intercept": 40000,
        "gen_Female": (-gap),
        "gen_Other": (-gap/2),
        "edu_Bachelor": 20000,
        "edu_Master": 25000,
        "edu_Doctorate": 28000,
        "lvl_Low": 5000,
        "lvl_Mid": 10000,
        "lvl_Senior": 20000,
        "YoE": 2000,
    }

    pay = linear_predict(df_dummies, coeffs, 10000)

    df["pay"] = pay.round(2)

    fake = Faker()
    fake.seed_instance(seed)

    def fake_first(gender):
        if gender.lower() in ["male", "man", "m"]:
            return fake.first_name_male()
        elif gender.lower() in ["female", "woman", "f", "w"]:
            return fake.first_name_female()
        else:
            return fake.first_name_nonbinary()

    df["first"] = df["gender"].apply(fake_first)
    personal = personal_info(N, fake)
    result = pd.concat([df, personal], axis=1)

    cols = ["last", "first", "eid", "gender", "race", "age", "job", "city", "state", "department", "education", "YoE", "level", "pay"]
    return result[cols]

def linear_predict(df, coeffs, sigma=1000):
    N = df.shape[0]
    result = np.zeros(N)

    for col in coeffs:
        if col == "intercept":
            result += coeffs[col]
        else:
            result += coeffs[col] * df[col]

    return result + np.random.normal(0, sigma, N)
