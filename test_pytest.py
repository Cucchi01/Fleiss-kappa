from FleissKappaCalculation import calculateFleissKappa 
import pandas as pd
import math

#example from Wikipedia
def test_KappaFleiss():
    d = {"1": [0, 0, 0, 0, 2, 7, 3, 2, 6, 0], "2": [0, 2, 0, 3, 2, 7, 2, 5, 5, 2], "3": [
        0, 6, 3, 9, 8, 0, 6, 3, 2, 2], "4": [0, 4, 5, 2, 1, 0, 3, 2, 1, 3], "5": [14, 2, 6, 0, 1, 0, 0, 2, 0, 7]}
    df = pd.DataFrame(data=d)
    k = calculateFleissKappa(df, ["1", "2", "3", "4", "5"], 10, 14, 0)

    assert math.isclose(0.210, k, rel_tol=1e-3), 'calculateFleissKappa doesn\'t calculate the right result'


test_KappaFleiss()