import numpy as np
import pandas as pd

def get_abs_strikes(df_in: pd.DataFrame) -> pd.DataFrame:
    """ Assumption: Absolute spreads are used """
    df = df_in.copy()
    abs_strikes=np.zeros(df.iloc[:, 3:].shape)
    for row in range(df.iloc[:, 3:].shape[0]):
        for col in range(df.iloc[:, 3:].shape[1]):
            # print(f"row {row}, col {col}")
            # print(f"fwd {df.Fwd.iloc[row] }, bps {np.array(df.columns[3+col])}")
            abs_strikes[row,col] = df.Fwd.iloc[row] + 0.0001*np.array(df.columns[3+col])
            # print(f"result {abs_strikes[row,col]}")
    
    df.iloc[:,3:] = abs_strikes
    return df