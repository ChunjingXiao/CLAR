
import numpy as np

def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=np.inf):
    """Computes dtw distance between two time series

    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)

    Returns:
        dtw distance
    """

    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N)).astype(np.float32)


    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])


    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    return cost[-1, -1], cost, _traceback(cost)
def _traceback(D):
    m, n = np.array(D.shape) - 1

    path = [(m, n)]
    while ((m > 0 ) or (n > 0)):
        back = np.argmin((D[m-1, n-1], D[m, n-1], D[m-1, n]))
        if m>0 and n>0 and (back == 0):
            m -= 1
            n -= 1
        elif n>0 and (back == 1 or m==0):
            n -= 1
        elif m>0 and (back==2 or n==0):
            m -= 1
        path.append((m, n))
    return list(reversed(path))



