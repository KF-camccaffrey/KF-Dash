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

if __name__ == "__main__":

    # import plotly
    import plotly.express as px

    df = pd.DataFrame({"test": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    df['percentile'] = df['test'].rank(pct=True) * 100
    print(df)

"""
    # test generation methods
    male_pay_range = [60000, 80000]
    female_pay_range = [60000, 80000]
    gender_ratio = 0.5

    df = basic(male_pay_range, female_pay_range, gender_ratio)

    fig = px.histogram(df, x='pay', color='sex',
                        labels={'pay': 'Pay', 'sex': 'Sex'},
                        marginal="box")
    fig.show()
"""
