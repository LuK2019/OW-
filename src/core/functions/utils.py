import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_prediction_for_row(predictor, row, optimal_parameters, strike_matrix, vola_matrix, df, beta):
    print(f"Shape df {df.shape}")
    print(f"Beta={beta}")
    param_0=tuple(optimal_parameters[row][0])
    strikes_0 =strike_matrix[row]
    fwd_0 = np.array(df.Fwd[row])
    expiry_0 = np.array(df.Expiry[row])
    prediction = [round(predictor(params= param_0, beta=beta, fwd=fwd_0, strike=strike, expiry=expiry_0),4) for strike in strikes_0]
    
    print(f"In row: {row}, there is \n strikes: {strikes_0} \n forward {fwd_0} \n expiry {expiry_0} \n prediction {prediction}")
    
    rmse = sum((prediction - vola_matrix[row])**2)**0.5
    print(f"Rmse for row {row} is {rmse}")
    return  prediction, rmse

def plot_prediction_vs_reality(predictor, row, optimal_parameters, strike_matrix, vola_matrix, df, beta):
    bps = np.array(df.columns[3:])
    df_filtered = df.iloc[row,3:]
    vola_true = df_filtered.to_numpy()
    prediction, rmse = get_prediction_for_row(predictor, row, optimal_parameters, strike_matrix, vola_matrix, df, beta)
    
    df_plot = pd.DataFrame({"bps": bps, "vola_true": vola_true, "vola_predicted": prediction})
    sns.scatterplot(x="bps", y="vola_true", label = "Market volatility", color="r", s=150, data=df_plot)
    sns.lineplot(x="bps", y="vola_predicted", label="vola_predicted", linewidth=3, color="black", data=df_plot).set_title(f"Tenor: {df.iloc[row,0]}, Expiry: {df.iloc[row,1]},\n Parameters: {tuple(optimal_parameters[row][0])}, Beta: {beta}")

def get_abs_strikes(df_in: pd.DataFrame) -> pd.DataFrame:
    """ Assumption: Absolute spreads are used """
    df = df_in.copy()
    abs_strikes=np.zeros(df.iloc[:, 3:].shape)
    for row in range(df.iloc[:, 3:].shape[0]):
        for col in range(df.iloc[:, 3:].shape[1]):
            abs_strikes[row,col] = df.Fwd.iloc[row] + 0.0001*np.array(df.columns[3+col])
    
    df.iloc[:,3:] = abs_strikes
    return df
