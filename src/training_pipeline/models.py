from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor

def obtener_modelos():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge':            Ridge(alpha=1.0, random_state=42),
        'BayesianRidge':    BayesianRidge(),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost':         AdaBoostRegressor(random_state=42),
        'Bagging':          BaggingRegressor(random_state=42),
        'MLP':              MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }