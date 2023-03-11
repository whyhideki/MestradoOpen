import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


class PerformanceTracker:
    def __init__(self, daily_returns, market_returns=None):
        self.daily_returns = daily_returns
        self.market_returns = market_returns

    def sharpe_ratio(self, annual_risk_free=0):
        daily_risk_free = (1 + annual_risk_free) ** (1 / 252) - 1
        excess_returns = self.daily_returns - daily_risk_free

        annualized_mean_of_excess_returns = self.annualized_daily_mean_returns(excess_returns)
        annualized_std_of_excess_returns = self.annualized_daily_std_returns(excess_returns)
        sharpe_ratio = annualized_mean_of_excess_returns / annualized_std_of_excess_returns
        return sharpe_ratio

    def max_drawdown(self):
        cumulative_return = np.cumprod(1 + self.daily_returns)
        rolling_max = np.maximum.accumulate(cumulative_return)
        drawdown = (cumulative_return - rolling_max) / rolling_max
        max_drawdown = np.min(drawdown)
        return max_drawdown * 100

    def portfolio_beta(self):
        if isinstance(self.market_returns, pd.Series):
            covariance = np.cov(self.daily_returns, self.market_returns, ddof=0)[0, 1]
            market_variance = np.var(self.market_returns, ddof=0)
            beta = covariance / market_variance
            return beta
        else:
            return None

    def portfolio_value_at_risk(self, alpha=0.05):
        return scipy.stats.norm.ppf(alpha, np.mean(self.daily_returns), np.std(self.daily_returns))

    def annualized_return(self):
        cumulative_return = (1 + self.daily_returns).prod()
        years = len(self.daily_returns) / 252
        annual_return = (cumulative_return ** (1 / years)) - 1
        return 100 * annual_return

    def annualized_std_return(self):
        std_dev = self.daily_returns.std()
        annual_std_dev = std_dev * np.sqrt(252)
        return 100 * annual_std_dev

    @staticmethod
    def annualized_daily_mean_returns(daily_returns):
        return (1 + daily_returns.mean()) ** 252 - 1

    @staticmethod
    def annualized_daily_std_returns(daily_returns):
        return daily_returns.std() * np.sqrt(252)

    def plot_cumulative_returns(self):
        if isinstance(self.market_returns, pd.Series):
            portfolio_cum_returns = (1 + self.daily_returns).cumprod()
            market_cum_returns = (1 + self.market_returns).cumprod()
            plt.plot(portfolio_cum_returns.index, portfolio_cum_returns, label='Portfolio')
            plt.plot(market_cum_returns.index, market_cum_returns, label='Market Index')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.title('Portfolio vs. Market Index Cumulative Returns')
            plt.legend()
            plt.show()

    def __call__(self):
        result = {
            'sharpe': self.sharpe_ratio(),
            'max_drawdown': self.max_drawdown(),
            'value_at_risk_daily_95': self.portfolio_value_at_risk(),
            'beta': self.portfolio_beta(),
            'annual_return': self.annualized_return(),
            'annual_std': self.annualized_std_return()
        }
        self.plot_cumulative_returns()
        return result
