from scipy import stats

def run_test4weighted(OG1: str, OG2: str, tt: float, tf: float,
                      ft: float, ff: float, direction: str):
    odds, pvalue = stats.fisher_exact(
        [[tt, tf],
         [ft, ff]],
         alternative = direction
    )
    
    return OG1, OG2, odds, pvalue
    
def run_test4transition(OG1: str, OG2: str, t1: int,
                        t2: int, k: int, n: int):
    direction = k
    k = abs(k)
    _, pvalue = stats.fisher_exact(
        [[k, t1-k],
         [t2-k, n-t1-t2+k]],
         alternative='greater'
    )
    
    return OG1, OG2, direction, pvalue