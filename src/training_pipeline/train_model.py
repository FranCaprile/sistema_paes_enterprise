from sklearn.metrics import mean_squared_error, r2_score

def entrenar_un_modelo(modelo, X_train, X_test, y_train, y_test, test_size, random_state):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    return {
        'rmse': mean_squared_error(y_test, pred, squared=False),
        'r2':   r2_score(y_test, pred)
    }