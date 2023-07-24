import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.linalg import LinAlgError
from warnings import warn
from scipy.optimize._remove_redundancy import (
    _remove_redundancy_svd, _remove_redundancy_pivot_sparse,
    _remove_redundancy_pivot_dense, _remove_redundancy_id
    )

############################ Code Adapted from https://github.com/scipy/scipy/blob/5dcc0f66fe6af9d954d1a7e3c0f451736fa7500a/scipy/optimize/_linprog_ip.py ####

"""Interior-point method for linear programming
The *interior-point* method uses the primal-dual path following algorithm
outlined in [1]_. This algorithm supports sparse constraint matrices and
is typically faster than the simplex methods, especially for large, sparse
problems. Note, however, that the solution returned may be slightly less
accurate than those of the simplex methods and will not, in general,
correspond with a vertex of the polytope defined by the constraints.
    .. versionadded:: 1.0.0
References
----------
.. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
       optimizer for linear programming: an implementation of the
       homogeneous algorithm." High performance optimization. Springer US,
       2000. 197-232.
"""


def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    """
    An implementation of [4] equation 8.21
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    # [4] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [4] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha


def _get_message(status):
    """
    Given problem status code, return a more detailed message.
    Parameters
    ----------
    status : int
        An integer representing the exit status of the optimization::
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered
    Returns
    -------
    message : str
        A string descriptor of the exit status of the optimization.
    """
    messages = (
        ["Optimization terminated successfully.",
         "The iteration limit was reached before the algorithm converged.",
         "The algorithm terminated successfully and determined that the "
         "problem is infeasible.",
         "The algorithm terminated successfully and determined that the "
         "problem is unbounded.",
         "Numerical difficulties were encountered before the problem "
         "converged. Please check your problem formulation for errors, "
         "independence of linear equality constraints, and reasonable "
         "scaling and matrix condition numbers. If you continue to "
         "encounter this error, please submit a bug report."
         ])
    return messages[status]


def _get_solver(M, sparse=False, lstsq=False, sym_pos=True,
                cholesky=True, permc_spec='MMD_AT_PLUS_A'):
    """
    Given solver options, return a handle to the appropriate linear system
    solver.
    Parameters
    ----------
    M : 2-D array
        As defined in [4] Equation 8.31
    sparse : bool (default = False)
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool (default = False)
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool (default = True)
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool (default = True)
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        Sparsity preservation strategy used by SuperLU. Acceptable values are:
        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.
        See SuperLU documentation.
    Returns
    -------
    solve : function
        Handle to the appropriate solver function
    """
    try:
        if sparse:
            if lstsq:
                def solve(r, sym_pos=False):
                    return sps.linalg.lsqr(M, r)[0]
            elif cholesky:
                try:
                    # Will raise an exception in the first call,
                    # or when the matrix changes due to a new problem
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                except Exception:
                    _get_solver.cholmod_factor = cholmod_analyze(M)
                    _get_solver.cholmod_factor.cholesky_inplace(M)
                solve = _get_solver.cholmod_factor
            else:
                if has_umfpack and sym_pos:
                    solve = sps.linalg.factorized(M)
                else:  # factorized doesn't pass permc_spec
                    solve = sps.linalg.splu(M, permc_spec=permc_spec).solve

        else:
            if lstsq:  # sometimes necessary as solution is approached
                def solve(r):
                    return sp.linalg.lstsq(M, r)[0]
            elif cholesky:
                L = sp.linalg.cho_factor(M)

                def solve(r):
                    return sp.linalg.cho_solve(L, r)
            else:
                # this seems to cache the matrix factorization, so solving
                # with multiple right hand sides is much faster
                def solve(r, sym_pos=sym_pos):
                    if sym_pos:
                        return sp.linalg.solve(M, r, assume_a="pos")
                    else:
                        return sp.linalg.solve(M, r)
    # There are many things that can go wrong here, and it's hard to say
    # what all of them are. It doesn't really matter: if the matrix can't be
    # factorized, return None. get_solver will be called again with different
    # inputs, and a new routine will try to factorize the matrix.
    except KeyboardInterrupt:
        raise
    except Exception:
        return None
    return solve

def _get_delta(A, b, c, x, y, z, tau, kappa, gamma, eta, sparse=False,
               lstsq=False, sym_pos=True, cholesky=True, pc=True, ip=False,
               permc_spec='MMD_AT_PLUS_A'):
    """
    Given standard form problem defined by ``A``, ``b``, and ``c``;
    current variable estimates ``x``, ``y``, ``z``, ``tau``, and ``kappa``;
    algorithmic parameters ``gamma and ``eta;
    and options ``sparse``, ``lstsq``, ``sym_pos``, ``cholesky``, ``pc``
    (predictor-corrector), and ``ip`` (initial point improvement),
    get the search direction for increments to the variable estimates.
    Parameters
    ----------
    As defined in [4], except:
    sparse : bool
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    pc : bool
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial. Even though it requires
        the solution of an additional linear system, the factorization
        is typically (implicitly) reused so solution is efficient, and the
        number of algorithm iterations is typically reduced.
    ip : bool
        True if the improved initial point suggestion due to [4] section 4.3
        is desired. It's unclear whether this is beneficial.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        (Has effect only with ``sparse = True``, ``lstsq = False``, ``sym_pos =
        True``.) A matrix is factorized in each iteration of the algorithm.
        This option specifies how to permute the columns of the matrix for
        sparsity preservation. Acceptable values are:
        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.
        This option can impact the convergence of the
        interior point algorithm; test different values to determine which
        performs best for your problem. For more information, refer to
        ``scipy.sparse.linalg.splu``.
    Returns
    -------
    Search directions as defined in [4]
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    if A.shape[0] == 0:
        # If there are no constraints, some solvers fail (understandably)
        # rather than returning empty solution. This gets the job done.
        sparse, lstsq, sym_pos, cholesky = False, False, True, False
    n_x = len(x)

    # [4] Equation 8.8
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [4] Equation 8.31
    Dinv = x / z

    if sparse:
        M = A.dot(sps.diags(Dinv, 0, format="csc").dot(A.T))
    else:
        M = A.dot(Dinv.reshape(-1, 1) * A.T)
    solve = _get_solver(M, sparse, lstsq, sym_pos, cholesky, permc_spec)

    # pc: "predictor-corrector" [4] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    n_corrections = 1 if pc else 0

    i = 0
    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    while i <= n_corrections:
        # Reference [4] Eq. 8.6
        rhatp = eta(gamma) * r_P
        rhatd = eta(gamma) * r_D
        rhatg = eta(gamma) * r_G

        # Reference [4] Eq. 8.7
        rhatxs = gamma * mu - x * z
        rhattk = gamma * mu - tau * kappa

        if i == 1:
            if ip:  # if the correction is to get "initial point"
                # Reference [4] Eq. 8.23
                rhatxs = ((1 - alpha) * gamma * mu -
                          x * z - alpha**2 * d_x * d_z)
                rhattk = ((1 - alpha) * gamma * mu -
                    tau * kappa -
                    alpha**2 * d_tau * d_kappa)
            else:  # if the correction is for "predictor-corrector"
                # Reference [4] Eq. 8.13
                rhatxs -= d_x * d_z
                rhattk -= d_tau * d_kappa

        # sometimes numerical difficulties arise as the solution is approached
        # this loop tries to solve the equations using a sequence of functions
        # for solve. For dense systems, the order is:
        # 1. scipy.linalg.cho_factor/scipy.linalg.cho_solve,
        # 2. scipy.linalg.solve w/ sym_pos = True,
        # 3. scipy.linalg.solve w/ sym_pos = False, and if all else fails
        # 4. scipy.linalg.lstsq
        # For sparse systems, the order is:
        # 1. sksparse.cholmod.cholesky (if available)
        # 2. scipy.sparse.linalg.factorized (if umfpack available)
        # 3. scipy.sparse.linalg.splu
        # 4. scipy.sparse.linalg.lsqr
        solved = False
        while not solved:
            try:
                # [4] Equation 8.28
                p, q = _sym_solve(Dinv, A, c, b, solve)
                # [4] Equation 8.29
                u, v = _sym_solve(Dinv, A, rhatd -
                                  (1 / x) * rhatxs, rhatp, solve)
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    raise LinAlgError
                solved = True
            except (LinAlgError, ValueError, TypeError) as e:
                # Usually this doesn't happen. If it does, it happens when
                # there are redundant constraints or when approaching the
                # solution. If so, change solver.
                if cholesky:
                    cholesky = False
                    warn(
                        "Solving system with option 'cholesky':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'cholesky' to False.")
                elif sym_pos:
                    sym_pos = False
                    warn(
                        "Solving system with option 'sym_pos':True "
                        "failed. It is normal for this to happen "
                        "occasionally, especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "consider setting option 'sym_pos' to False.")
                elif not lstsq:
                    lstsq = True
                    warn(
                        "Solving system with option 'sym_pos':False "
                        "failed. This may happen occasionally, "
                        "especially as the solution is "
                        "approached. However, if you see this frequently, "
                        "your problem may be numerically challenging. "
                        "If you cannot improve the formulation, consider "
                        "setting 'lstsq' to True. Consider also setting "
                        "`presolve` to True, if it is not already.")
                else:
                    raise e
                solve = _get_solver(M, sparse, lstsq, sym_pos,
                                    cholesky, permc_spec)
        # [4] Results after 8.29
        d_tau = ((rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) /
                 (1 / tau * kappa + (-c.dot(p) + b.dot(q))))
        d_x = u + p * d_tau
        d_y = v + q * d_tau

        # [4] Relations between  after 8.25 and 8.26
        d_z = (1 / x) * (rhatxs - z * d_x)
        d_kappa = 1 / tau * (rhattk - kappa * d_tau)

        # [4] 8.12 and "Let alpha be the maximal possible step..." before 8.23
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
        if ip:  # initial point - see [4] 4.4
            gamma = 10
        else:  # predictor-corrector, [4] definition after 8.12
            beta1 = 0.1  # [4] pg. 220 (Table 8.1)
            gamma = (1 - alpha)**2 * min(beta1, (1 - alpha))
        i += 1

    return d_x, d_y, d_z, d_tau, d_kappa


def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    """
    An implementation of Equation 8.9
    References
    ----------
         Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa


def _get_blind_start(shape):
    """
    Return the starting point from 4.4
    References
    ----------
         Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    m, n = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [4] used as indicators of
    the status of optimization.
    References
    ----------
         Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """

    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, z, tau):
        return c * tau - A.T.dot(y) - z

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, z, kappa):
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)

    obj = c.dot(x / tau) + c0

    def norm(a):
        return np.linalg.norm(a)

    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj







def _sym_solve(Dinv, A, r1, r2, solve):
    """
    An implementation of [4] equation 8.31 and 8.32
    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    """
    # [4] 8.31
    r = r2 + A.dot(Dinv * r1)
    v = solve(r)
    # [4] 8.32
    u = Dinv * (A.T.dot(v) - r1)
    return u, v





# A, b, c, c0, alpha0, beta, maxiter, disp, tol, sparse, lstsq,
#             sym_pos, cholesky, pc, ip, permc_spec, callback, postsolve_args


# c, c0, A, b, callback, postsolve_args, maxiter=1000, tol=1e-8,
#                 disp=False, alpha0=.99995, beta=0.1, sparse=False, lstsq=False,
#                 sym_pos=True, cholesky=None, pc=True, ip=False,
#                 permc_spec='MMD_AT_PLUS_A', **unknown_options


def solveLP(c,A,b,thr, tol=1e-6, 
            maxiter=1000, alpha0=.99995, beta=0.1, sparse=False, lstsq=False,
                sym_pos=True, cholesky=None, pc=True, ip=False, permc_spec='MMD_AT_PLUS_A'):
    '''
    c : 1D numpy array
        The coefficients of the linear objective function to be minimized.
    G : 2D numpy array, optional
        The inequality constraint matrix. 
    h : 1D numpy array, optional
        The inequality constraint vector.
    A : 2D numpy array, optional
        The equality constraint matrix. 
    b : 1D numpy array, optional
        The equality constraint vector. .
    '''
    iteration = 0
    c0 = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    # ip = ip if pc else False


    # [4] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol  # we might get lucky : )

    # if disp:
    #     _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)
    # if callback is not None:
    #     x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
    #     res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
    #                             'con': con, 'nit': iteration, 'phase': 1,
    #                             'complete': False, 'status': 0,
    #                             'message': "", 'success': False})
    #     callback(res)

    status = 0
    message = "Optimization terminated successfully."

    # if sparse:
    #     A = sps.csc_matrix(A)
    #     A.T = A.transpose()  # A.T is defined for sparse matrices but is slow
    #     # Redefine it to avoid calculating again
    #     # This is fine as long as A doesn't change

    while go:

        iteration += 1

        if ip:  # initial point
            # [4] Section 4.4
            gamma = 1

            def eta(g):
                return 1
        else:
            # gamma = 0 in predictor step according to [4] 4.1
            # if predictor/corrector is off, use mean of complementarity [6]
            # 5.1 / [4] Below Figure 10-4
            gamma = 0 if pc else beta * np.mean(z * x)
            # [4] Section 4.1

            def eta(g=gamma):
                return 1 - g

        try:
            # Solve [4] 8.6 and 8.7/8.13/8.23
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
                A, b, c, x, y, z, tau, kappa, gamma, eta,
                sparse, lstsq, sym_pos, cholesky, pc, ip, permc_spec)

            if ip:  # initial point
                # [4] 4.4
                # Formula after 8.23 takes a full step regardless if this will
                # take it negative
                alpha = 1.0
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y,
                    d_z, d_tau, d_kappa, alpha)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False  # done with initial point
            else:
                # [4] Section 4.3
                alpha = _get_step(x, d_x, z, d_z, tau,
                                    d_tau, kappa, d_kappa, alpha0)
                # [4] Equation 8.9
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        except (LinAlgError, FloatingPointError,
                ValueError, ZeroDivisionError) as exp:
            # this can happen when sparse solver is used and presolve
            # is turned off. Also observed ValueError in AppVeyor Python 3.6
            # Win32 build (PR #8676). I've never seen it otherwise.
            status = 4
            print ("Error:", exp)
            print("Probable cause: sparse solver is used and presolve",
            "is turned off. Also observed ValueError in AppVeyor Python 3.6 Win32 build (PR #8676).")
            raise 

        # [4] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
            A, b, c, c0, x, y, z, tau, kappa)
        n_x = len(x)
        mu = (x.dot(z) + tau * kappa) / (n_x + 1)
        go = (rho_p > tol or rho_d > tol or rho_A > tol) and (mu > thr)


        # if callback is not None:
        #     x_o, fun, slack, con = _postsolve(x/tau, postsolve_args)
        #     res = OptimizeResult({'x': x_o, 'fun': fun, 'slack': slack,
        #                             'con': con, 'nit': iteration, 'phase': 1,
        #                             'complete': False, 'status': 0,
        #                             'message': "", 'success': False})
        #     callback(res)

        # [4] 4.5
        inf1 = (rho_p < tol and rho_d < tol and rho_g < tol and tau < tol *
                max(1, kappa))
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            # [4] Lemma 8.4 / Theorem 8.3
            if b.transpose().dot(y) > tol:
                status = 2
            else:  # elif c.T.dot(x) < tol: ? Probably not necessary.
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break

    
    # [4] Statement after Theorem 8.2
    return x, y, z, tau, kappa, mu
