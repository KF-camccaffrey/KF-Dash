import numpy as np
import pandas as pd

def basic(m_range, f_range, gender_ratio):
    # total sample N + M + F
    N = 10000

    # P(M)
    p_M = gender_ratio / 100

    # slice indices
    start = int(N*p_M)
    stop = start + N

    # calculate parameters
    ranges = np.array((f_range, m_range))
    means = np.average(ranges, 1)
    stds = (ranges[:, 1] - means) / 3

    print(f"Ranges: {ranges}")
    print(f"Means: {means}")
    print(f"StDs: {stds}")

    # draw samples
    np.random.seed(42)
    pay = np.random.normal(means, stds, [N, 2]).T.flatten().round(2)
    sex = np.repeat(["Female", "Male"], N)

    # create dataframe
    df = pd.DataFrame({"pay": pay[start:stop],
                       "sex": sex[start:stop]})
    return df


def generate_dataset(N=10000, ratio=0.5, gap=0):
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


    return df

def linear_predict(df, coeffs, sigma=1000):
    N = df.shape[0]
    result = np.zeros(N)

    for col in coeffs:
        if col == "intercept":
            result += coeffs[col]
        else:
            result += coeffs[col] * df[col]

    return result + np.random.normal(0, sigma, N)




if __name__ == "__main__":

    # import plotly
    import plotly.express as px
    import matplotlib.pyplot as plt

    df = generate_dataset()

    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):  # more options can be specified also
        print(df.head(100))


    male_pay = df[df['gender'] == 'Male']['pay']
    female_pay = df[df['gender'] == 'Female']['pay']

    bins = 20

    plt.hist(male_pay, bins=bins, alpha=0.5, label='Male Pay')
    plt.hist(female_pay, bins=bins, alpha=0.5, label='Female Pay')
    plt.show()
