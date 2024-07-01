import pandas as pd
import numpy as np
from IPython.display import display

class PortfolioOptimizer:
    def __init__(self, data, params):
        self.data = data
        self.params = params
        self.constraints=params['constraints']
        self.verbose = params.get('verbose', False)
        self.preprocess_data()
        self.print=True
    
    def preprocess_data(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y-%m-%d')
        self.params['initial_data'] = pd.to_datetime(self.params['initial_data'], format='%Y-%m-%d')
        self.params['end_data'] = pd.to_datetime(self.params['end_data'], format='%Y-%m-%d')
        self.params['start_test_data'] = pd.to_datetime(self.params['start_test_data'], format='%Y-%m-%d')
        self.params['end_test_data'] = pd.to_datetime(self.params['end_test_data'], format='%Y-%m-%d')
        self.data = self.data[self.data['Ticker'].isin(self.params['tickers'])]
        self.model_data = self.data[(self.data['Date'] >= self.params['initial_data']) & (self.data['Date'] <= self.params['end_data'])]
        self.test_data = self.data[(self.data['Date'] >= self.params['start_test_data']) & (self.data['Date'] <= self.params['end_test_data'])]

        if self.verbose:
            print(f"Initial Data Date Range: {self.params['initial_data']} to {self.params['end_data']}")
            print(f"Model Data Date Range: {self.model_data['Date'].min()} to {self.model_data['Date'].max()}")
            print(f"Start Test Data Date Range: {self.params['start_test_data']} to {self.params['end_test_data']}")
            print(f"Test Data Date Range: {self.test_data['Date'].min()} to {self.test_data['Date'].max()}")
            print(f"Model Data Shape: {self.model_data.shape}")
            print(f"Test Data Shape: {self.test_data.shape}")

        self.returns = self.model_data.pivot_table(index='Date', columns='Ticker', values='Returns')
        self.cov_matrix = self.returns.cov()

    def objective_function(self, weights, z):
        """Objective function to minimize (portfolio variance using KKT conditions)."""
        M = self.params.get('M', 1000)
        tickers = self.params['tickers']
        kkt1 = all(np.dot(self.cov_matrix.loc[ticker], weights) <= z + M * (1 - weight) for ticker, weight in zip(tickers, weights))
        kkt2 = all(np.dot(self.cov_matrix.loc[ticker], weights) >= -z - M * (1 - weight) for ticker, weight in zip(tickers, weights))
        if kkt1 and kkt2:
            return z
        else:
            return np.inf
    
    def check_constraints(self, weights):
        weighted_data = self.model_data.copy()
        weighted_data['Weight'] = weighted_data['Ticker'].apply(lambda x: weights[self.params['tickers'].index(x)])
        weighted_data['Weighted Volume'] = weighted_data['Volume'] * weighted_data['Weight']
        weighted_data['Weighted Beta'] = weighted_data['Beta'] * weighted_data['Weight']
        weighted_data['Weighted Alpha'] = weighted_data['Alpha'] * weighted_data['Weight']
        weighted_data['Weighted Sharpe'] = weighted_data['Sharpe Ratio'] * weighted_data['Weight']

        portfolio_volume = weighted_data.groupby('Ticker')['Weighted Volume'].mean().sum()
        portfolio_beta = weighted_data.groupby('Ticker')['Weighted Beta'].mean().sum()
        portfolio_alpha = weighted_data.groupby('Ticker')['Weighted Alpha'].mean().sum()
        portfolio_sharpe = weighted_data.groupby('Ticker')['Weighted Sharpe'].mean().sum()
        portfolio_sectors = (weighted_data.groupby('Sector')['Weight'].sum() > 0).sum()
        portfolio_size = (weighted_data.groupby('Ticker')['Weight'].sum() > 0).sum()
        
        if self.print:
            display(weighted_data)
            self.print=False
        
        constraints = {
            'PortfolioVolume': portfolio_volume >= self.constraints['PortfolioVolumeThreshold'] if not np.isnan(self.constraints['PortfolioVolumeThreshold']) else True,
            'PortfolioBeta': portfolio_beta >= self.constraints['PortfolioBetaThreshold'] if not np.isnan(self.constraints['PortfolioBetaThreshold']) else True,
            'PortfolioAlpha': portfolio_alpha >= self.constraints['PortfolioAlphaThreshold'] if not np.isnan(self.constraints['PortfolioAlphaThreshold']) else True,
            'PortfolioSharpe': portfolio_sharpe >= self.constraints['PortfolioSharpeRatioThreshold'] if not np.isnan(self.constraints['PortfolioSharpeRatioThreshold']) else True,
            'PortfolioSectors': portfolio_sectors <= self.constraints['PortfolioSectorThreshold'] if not np.isnan(self.constraints['PortfolioSectorThreshold']) else True,
            'PortfolioSize': portfolio_size <= self.constraints['PortfolioSizeThreshold'] if not np.isnan(self.constraints['PortfolioSizeThreshold']) else True
        }
        
        constraints_values = {
            'PortfolioVolume': portfolio_volume,
            'PortfolioBeta': portfolio_beta,
            'PortfolioAlpha': portfolio_alpha,
            'PortfolioSharpe': portfolio_sharpe,
            'PortfolioSectors': portfolio_sectors,
            'PortfolioSize': portfolio_size
        }
        
        constraints_satisfied = all(constraints.values())
        
        return constraints_satisfied, constraints, constraints_values, weighted_data
    
    def tabu_search(self):
        iterations=self.params['iterations']
        tabu_tenure=self.params['tabu_tenure']
        num_assets = len(self.params['tickers'])
        best_weights = np.round(np.random.dirichlet(np.ones(num_assets)), 2)
        z = np.random.rand()
        best_objective = np.inf
        constraints_satisfied, constraints, constraints_values, weighted_data = self.check_constraints(best_weights)
        if constraints_satisfied:
            best_objective = self.objective_function(best_weights, z)
            best_constraints = constraints
        tabu_list = [(best_weights, z)]

        for iteration in range(iterations):

            neighbor_weights = np.round(np.random.dirichlet(np.ones(num_assets)), 2)
            neighbor_z = np.random.rand()
            while any((neighbor_weights == x[0]).all() and neighbor_z == x[1] for x in tabu_list):
                neighbor_weights = np.round(np.random.dirichlet(np.ones(num_assets)), 2)
                neighbor_z = np.random.rand()
            
            neighbor_objective = self.objective_function(neighbor_weights, neighbor_z)
            constraints_satisfied, constraints, constraints_value, weighted_datas = self.check_constraints(neighbor_weights)
            if not constraints_satisfied:
                neighbor_objective = np.inf

            if neighbor_objective < best_objective:
                best_weights = neighbor_weights
                z = neighbor_z
                best_objective = neighbor_objective
                best_constraints = constraints
                tabu_list.append((best_weights, z))
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                if self.verbose:
                    print(f"Iteration {iteration + 1}")
                    print(f'New best found at iteration {iteration+1}')
                    print(f"Best Objective: {best_objective}")
                    print(f"Best Weights: {dict(zip(self.params['tickers'], best_weights))}")
                    print(f'Constraints: {constraints_values}')
        
        constraints_satisfied, final_constraints, constraints_values, weighted_data = self.check_constraints(best_weights)
        if not constraints_satisfied:
            raise ValueError("No feasible solution found that satisfies all constraints.")
        return best_weights, final_constraints, constraints_values,best_objective, weighted_data

    def optimize(self):
        best_weights, best_constraints, constraints_values, objective_value, weighted_data= self.tabu_search()
        weight_dict = dict(zip(self.params['tickers'], best_weights))
        if self.verbose:
            print("Optimization completed.")
            print(f"Optimized Weights: {best_weights}")
            print("Weights for each underlying:")
            print(weight_dict)
            print('Constraints Value:')
            print(constraints_values)
            print(f"Objective Function Value: {objective_value}")
        return best_weights, best_constraints, weight_dict, constraints_values, objective_value, weighted_data
    
    def backtest(self, weights):
        if self.verbose:
            print("Starting backtesting...")
        test_returns = self.test_data.pivot_table(index='Date', columns='Ticker', values='Returns')
        if self.verbose:
            print(f"Test Returns Shape: {test_returns.shape}")
            print(f"Weights Shape: {weights.shape}")

        test_returns = test_returns[self.params['tickers']]
        if self.verbose:
            print(f"Reordered Test Returns Shape: {test_returns.shape}")
        
        if test_returns.empty:
            raise ValueError("Test returns dataframe is empty. Please check the date range and data availability.")

        portfolio_returns = test_returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        weighted_data = self.test_data.copy()
        weighted_data['Weight'] = weighted_data['Ticker'].apply(lambda x: weights[self.params['tickers'].index(x)])
        weighted_data['Weighted Volume'] = weighted_data['Volume'] * weighted_data['Weight']
        weighted_data['Weighted Beta'] = weighted_data['Beta'] * weighted_data['Weight']
        weighted_data['Weighted Alpha'] = weighted_data['Alpha'] * weighted_data['Weight']
        weighted_data['Weighted Sharpe'] = weighted_data['Sharpe Ratio'] * weighted_data['Weight']
        
        portfolio_volume = weighted_data['Weighted Volume'].mean()
        portfolio_beta = weighted_data['Weighted Beta'].mean()
        portfolio_alpha = weighted_data['Weighted Alpha'].mean()
        portfolio_sharpe = weighted_data['Weighted Sharpe'].mean()
        portfolio_sectors = (weighted_data.groupby('Sector')['Weight'].sum() > 0).sum()
        portfolio_size = (weighted_data.groupby('Ticker')['Weight'].sum() > 0).sum()
        
        constraints_values = {
            'PortfolioVolume': portfolio_volume,
            'PortfolioBeta': portfolio_beta,
            'PortfolioAlpha': portfolio_alpha,
            'PortfolioSharpe': portfolio_sharpe,
            'PortfolioSectors': portfolio_sectors,
            'PortfolioSize': portfolio_size
        }

        if self.verbose:
            print("Backtesting completed.")
            print(f"Portfolio Returns: {portfolio_returns}")
            print(f"Cumulative Returns: {cumulative_returns}")
            print(f"Maximum Drawdown: {max_drawdown}")
            print('Constraints during Backtest:')
            for k, v in constraints_values.items():
                print(f"{k}: {v} (Threshold: {self.params.get(k+'Threshold', 'Not Defined')})")
        
        return portfolio_returns, cumulative_returns, max_drawdown, constraints_values
