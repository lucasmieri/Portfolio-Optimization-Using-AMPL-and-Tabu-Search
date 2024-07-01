import pandas as pd
from amplpy import AMPL, Environment, ampl_notebook


ampl = ampl_notebook(
    modules=["highs", "cbc", "gurobi", "cplex"],
    license_uuid="1015c5b2-9c80-410d-affc-c006f619be2b")


def solve_ampl_cplx(model:str):

  # Inicializa o AMPL e define o solver
  ampl = AMPL(Environment())
  ampl.setOption('solver', 'cplex')
  # Carrega o modelo ajustado no AMPL
  ampl.eval(model)

  # Resolve o problema
  ampl.solve()
  return ampl

def get_model():
    # Define the AMPL model
    model = """
    reset;

    # Define sets
    set TICKERS ordered;

    # Define parameters
    param returns {TICKERS};
    param cov_matrix {TICKERS, TICKERS};
    param M default 1000;
    param volume {TICKERS};
    param beta {TICKERS};
    param alpha {TICKERS};
    param sharpe_ratio {TICKERS};
    param PortfolioVolumeThreshold;
    param PortfolioBetaThreshold;
    param PortfolioAlphaThreshold;
    param PortfolioSharpeRatioThreshold;
    param PortfolioSectorThreshold;
    param PortfolioSizeThreshold;

    # Define variables
    param unit_size default 0.01;
    var x {TICKERS} integer, >= 0; # Discrete weight units
    var w {i in TICKERS} = x[i] * unit_size;
    var y {TICKERS} binary;
    var z;

    # Objective function: minimize total variance (linearized)
    minimize Variance: z;

    # Constraints
    subject to TotalWeight:
        sum {i in TICKERS} w[i] = 1;

    subject to MaxSelected:
        sum {i in TICKERS} y[i] <= PortfolioSizeThreshold;

    subject to MinSelected:
        sum {i in TICKERS} y[i] >= 10;

    subject to Selection {i in TICKERS}:
        w[i] <= y[i];

    subject to PositiveWeight {i in TICKERS}:
        w[i] >= unit_size * y[i];

    subject to KKT1 {i in TICKERS}:
        sum {j in TICKERS} cov_matrix[i,j] * w[j] <= z + M * (1 - y[i]);

    subject to KKT2 {i in TICKERS}:
        sum {j in TICKERS} cov_matrix[i,j] * w[j] >= -z - M * (1 - y[i]);

    subject to VolumeConstraint:
        sum {i in TICKERS} volume[i] * w[i] >= PortfolioVolumeThreshold;

    subject to BetaConstraint:
        sum {i in TICKERS} beta[i] * w[i] >= PortfolioBetaThreshold;

    subject to AlphaConstraint:
        sum {i in TICKERS} alpha[i] * w[i] >= PortfolioAlphaThreshold;

    subject to SharpeConstraint:
        sum {i in TICKERS} sharpe_ratio[i] * w[i] >= PortfolioSharpeRatioThreshold;
    """
    return model


def solve_ampl_model(model: str, data: pd.DataFrame, params:dict):
    # Extract parameters from the model_dataFrame
    # Convert date columns to datetime
    constraints=params['constraints']
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    params['initial_data'] = pd.to_datetime(params['initial_data'], format='%Y-%m-%d')
    params['end_data'] = pd.to_datetime(params['end_data'], format='%Y-%m-%d')
    params['start_test_data'] = pd.to_datetime(params['start_test_data'], format='%Y-%m-%d')
    params['end_test_data'] = pd.to_datetime(params['end_test_data'], format='%Y-%m-%d')


    # Filter model_data based on the specified date ranges and tickers
    model_data = data[(data['Date'] >= params['initial_data']) & (data['Date'] <= params['end_data']) & data['Ticker'].isin(params['tickers'])]

    tickers = model_data['Ticker'].unique()
    returns = model_data.groupby('Ticker')['Returns'].mean()
    cov_matrix = model_data.pivot_table(index='Date', columns='Ticker', values='Returns').cov()
    volume = model_data.groupby('Ticker')['Volume'].mean()
    beta = model_data.groupby('Ticker')['Beta'].mean()
    alpha = model_data.groupby('Ticker')['Alpha'].mean()
    sharpe_ratio = model_data.groupby('Ticker')['Sharpe Ratio'].mean()

    # Initialize AMPL and set the solver
    ampl = AMPL(Environment())
    ampl.setOption('solver', 'cplex')
    
    # Load the model
    ampl.eval(model)
    
    # Set model_data in AMPL
    ampl.set['TICKERS'] = tickers
    ampl.param['returns'] = returns.to_dict()
    ampl.param['cov_matrix'] = cov_matrix
    ampl.param['volume'] = volume.to_dict()
    ampl.param['beta'] = beta.to_dict()
    ampl.param['alpha'] = alpha.to_dict()
    ampl.param['sharpe_ratio'] = sharpe_ratio.to_dict()
    
    ampl_dict_values={
            'TICKERS' :tickers,
    'returns': returns.to_dict(),
    'cov_matrix' : cov_matrix,
    'volume' : volume.to_dict(),
    'beta':beta.to_dict(),
    'alpha': alpha.to_dict(),
    'sharpe_ratio':sharpe_ratio.to_dict(),
    }
    
    # Set other parameters from constraints
    for param, value in constraints.items():
        ampl.param[param] = value
    
    # Solve the problem
    ampl.solve()

    # Extract the results
    assignments = ampl.getVariable('w').getValues().toPandas()
    discrete_weights = ampl.getVariable('x').getValues().toPandas()
    objective_value = ampl.getObjective('Variance').value()
    
    return assignments, discrete_weights, objective_value, ampl_dict_values