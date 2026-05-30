import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

random_state = 13


def main():
    df = pd.read_csv('insurance.csv')

    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])


if __name__ == '_-main__':
    main()
