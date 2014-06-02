"""
Attempt to implement a specialized line search algorithm.

Conjugate gradient acceleration of the EM algorithm.
Jamshidian and Jennrich, 1993
Appendix A.5.

"""


class LineSearchError(Exception):
    pass


def box_ok(bounds, x):
    if bounds is None:
        return True
    if len(bounds) != len(x):
        raise ValueError
    for x, (low, high) in zip(bounds, x):
        if not (low <= x <= high):
            return False
    return True


def jj93_linesearch(g, theta, d, a1, bounds=None):
    """
    Find the maximum of F(a) = f(theta + a1*d).

    Assumptions are as follows.
    alpha >= 0.
    The slope of F(alpha) is nonnegative at alpha=0.
    This linesearch is based on the secant method.

    Parameters
    ----------
    g : function
        gradient of function to maximize
    theta : point
        initial point
    d : vector
        direction of the search
    a1 : float
        initial distance in the direction of the search
    bounds : sequence, optional
        inclusive box constraints on the search

    Returns
    -------
    a1 : float
        alpha value for estimated maximum
    x1 : float
        estimated maximum value
    g1 : float
        gradient at x1

    """
    # Step 0
    n = 0
    a0 = 0
    x0 = theta + a0*d
    G0 = np.dot(d, g(x0))
    Ga0 = G0
    while True:
        # Step 1
        while n < 10:
            x1 = theta + a1*d
            if box_ok(bounds, x1):
                ga1 = g(x1)
                Ga1 = np.dot(d, g1)
                n += 1
                break
            else:
                a1 /= 2
                n += 1
        # Step 2
        if n == 10:
            raise LineSearchError('please restart search')
        Ga0m = np.abs(Ga0)
        Ga1m = np.abs(Ga1)
        if n != 1 and Ga1m < 0.1 * G0:
            return a1, x1, ga1
        elif np.sign(a1 - a0) * (Ga0 - Ga1) / (Ga0m + Ga1m) < 1e-5:
            raise LineSearchError('please restart search')
        else:
            a0, Ga0, a1 = a1, Ga1, (a1*Ga0 - a0*Ga1) / (Ga0 - Ga1)

