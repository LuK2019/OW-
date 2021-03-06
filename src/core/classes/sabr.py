import numpy as np
import pandas as pd
from typing import Tuple
import scipy


class SABR:

    @staticmethod
    def fit(objective, initial_guess: Tuple[float, float, float], beta: float, fwds: np.array, expiries: np.array, strike_matrix: np.array, vola_matrix: np.array, eps=0.0001) -> np.array:
        """For n=len(fwds) options the optimal parameters (alpha, beta, rho, nu) are fitted.
        Args:
            objective: objective function
            initial_guess: (alpha_0, rho_0, nu_0)
            beta: fixed value of beta between [0,1]
            fwds: vector of fwd prices of shape (n,)
            expiries: vector of expiries of shape (n,)
            strike_matrix: absolute strike prices of shape (n, num_spreads)
            vola_matrix: market volatilities of shape (n, num_spreads)
            eps: precision of boundaries

        Returns:
            np.array of shape (n, 3), with the optimal parmateres for each option
        """
        results = []
        for i in range(len(fwds)):
            bounds = ((0 + eps, None), (-1 + eps, 1 - eps), (0 + eps, None))
            parameteters = scipy.optimize.minimize(fun=objective, \
                            x0=initial_guess, \
                            args=(beta, fwds[i], strike_matrix[i], expiries[i], vola_matrix[i]),\
                            bounds=bounds, \
                            method='SLSQP',
                            options={"disp": True}
                            )
            results.append([parameteters.x])
        return results

    @staticmethod
    def objective(params, beta, fwd, strikes, expiry, volas):
        """Objective function to fit the parameters of the SABR model.
        Args:
            params: (alpha, rho, nu)
            beta: fixed value of beta between [0,1]
            fwd: float
            strikes: absolute strike values corresponding to fwd
            expiry: float, expiry of the fwd
            volas: market volatilities corresponding to the fwd at different spreads

        Returns: 
            root squared error 
        """
        assert len(strikes) == len(volas), f"Strikes and volas do not have the same size, strikes: {strikes.shape}, volas: {volas.shape}"
        se = 0
        if strikes[0]<=0:
            raise NotImplementedError("Negative strikes haven't been implemented yet.")
        for i in range(len(strikes)):
            if volas[i] == 0:
                difference = 0
            elif fwd == strikes[i]:  # ATM TODO: CHECK CORRECTNESS
                scalar = params[0] / (fwd ** (1 - beta))
                sum_1 = ((1 - beta)** 2 / 24) * params[0]** 2 / (fwd ** (2 - 2 * beta))
                sum_2 = 0.25 * (params[1] * beta * params[0] * params[2]) / (fwd ** (1 - beta))
                sum_3 = params[2]** 2 * (2 - 3 * params[1]** 2) / 24
                bracket = (sum_1 + sum_2 + sum_3) * expiry

                prediction = scalar*(1 + bracket)

                difference = prediction - volas[i]
            elif fwd != strikes[i]: 
                log = np.log(fwd / strikes[i])
                z = params[2] / params[0] * (fwd * strikes[i])**((1 - beta) / 2) * log
                
                arg = ((1 -  2*params[1] * z + z**2)**(1 / 2) + z - params[1]) / (1 - params[1])
                assert arg > 0, f"For strike: {strikes[i]}, number {i}: x argument is negative @ {arg}"
                x = np.log(arg)

                scalar_1 = (fwd ** strikes[i])**((1 - beta) / 2)
                sum_1 = 1 + (1 - beta)** 2 / 24 * log ** 2 + (1 - beta) / 1920 * log ** 4
                denominator = scalar_1 * sum_1
                fact_1 = params[0] / denominator
                fact_2 = z / x
                sum_2_1 = (1 - beta)** 2 / 24 * params[0]** 2 / ((fwd * strikes[i])**(1 - beta))
                sum_2_2 = 0.25 * (params[0] * beta * params[1] * params[2]) / scalar_1
                sum_2_3 = ((2 - 3 * params[1]** 2) / 24) * params[2]** 2
                bracket = sum_2_1 + sum_2_2 + sum_2_3
                fact_3 = 1 + expiry * bracket

                prediction = fact_1 * fact_2 * fact_3
                
                difference = prediction - volas[i]
            se = se + difference ** 2
            rse = se ** (1 / 2)
        return rse

    @staticmethod
    def predict(params: Tuple[float, float, float], beta, fwd:float, strike:float, expiry:float) -> float:
        """SABR prediction of implied volatility given the fwd, strike and expiry for a set of params.

        Args:
            params: (alpha, beta, rho, nu)
            beta: fixed value of beta between [0,1]
            fwd: float
            strike: absolute strike value
            expiry: float, expiry of the fwd

        Returns: 
            implied volatility  
        """
        if strike<=0:
            raise NotImplementedError("Negative strikes haven't been implemented yet.")

        elif fwd == strike:  # ATM TODO: CHECK CORRECTNESS
            scalar = params[0] / (fwd ** (1 - beta))
            sum_1 = ((1 - beta)** 2 / 24) * params[0]** 2 / (fwd ** (2 - 2 * beta))
            sum_2 = 0.25 * (params[1] * beta * params[0] * params[2]) / (fwd ** (1 - beta))
            sum_3 = params[2]** 2 * (2 - 3 * params[1]** 2) / 24
            bracket = (sum_1 + sum_2 + sum_3) * expiry

            prediction = scalar*(1 + bracket)

        elif fwd != strike: 
            log = np.log(fwd / strike)
            z = params[2] / params[0] * (fwd * strike)**((1 - beta) / 2) * log
            
            arg = ((1 -  2*params[1] * z + z**2)**(1 / 2) + z - params[1]) / (1 - params[1])
            assert arg > 0, f"For strike: {strike} x argument is negative @ {arg}"
            x = np.log(arg)

            scalar_1 = (fwd ** strike)**((1 - beta) / 2)
            sum_1 = 1 + (1 - beta)** 2 / 24 * log ** 2 + (1 - beta) / 1920 * log ** 4
            denominator = scalar_1 * sum_1
            fact_1 = params[0] / denominator
            fact_2 = z / x
            sum_2_1 = (1 - beta)** 2 / 24 * params[0]** 2 / ((fwd * strike)**(1 - beta))
            sum_2_2 = 0.25 * (params[0] * beta * params[1] * params[2]) / scalar_1
            sum_2_3 = ((2 - 3 * params[1]** 2) / 24) * params[2]** 2
            bracket = sum_2_1 + sum_2_2 + sum_2_3
            fact_3 = 1 + expiry * bracket

            prediction = fact_1 * fact_2 * fact_3
            
        return prediction

