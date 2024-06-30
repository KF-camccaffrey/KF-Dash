import numpy as np
import pandas as pd

def basic(m_range, f_range, ratio):
    N = 10000
    m_mean = np.average(m_range)
    f_mean = np.average(f_range)
    m_std = (np.max(m_range) - m_mean) / 3
    f_std = (np.max(f_range) - f_mean) / 3
    m_n = np.round(N*ratio)
    f_n = N - m_n

    m_pay = np.random.normal(m_mean, m_std, 50)
    f_pay = np.random.normal(f_mean, f_std, 50)

    #pay = np.concat((m_pay, f_pay), axis=None)
    #sex = np.concat((np.repeat("Male", 50), np.repeat("Female", 50)), axis=None)
    #df = pd.DataFrame(np.c_[pay, sex], columns=["pay", "sex"])
    return m_pay, f_pay
