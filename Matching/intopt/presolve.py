import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.linalg import LinAlgError
from warnings import warn
from scipy.optimize._remove_redundancy import (
    _remove_redundancy_svd, _remove_redundancy_pivot_sparse,
    _remove_redundancy_pivot_dense, _remove_redundancy_id
    )
class presolve:
    def __init__(self, A_ub, b_ub, A_eq, b_eq, tol=1e-9,  rr= True, rr_method=None) -> None:
        '''
        A_ub : 2D array, optional
            The inequality constraint matrix. Each row of ``A_ub`` specifies the
            coefficients of a linear inequality constraint on ``x``.
        b_ub : 1D array, optional
            The inequality constraint vector. Each element represents an
            upper bound on the corresponding value of ``A_ub @ x``.
        A_eq : 2D array, optional
            The equality constraint matrix. Each row of ``A_eq`` specifies the
            coefficients of a linear equality constraint on ``x``.
        b_eq : 1D array, optional
            The equality constraint vector. Each element of ``A_eq @ x`` must equal
            the corresponding element of ``b_eq``.
        
        rr : bool
            If ``True`` attempts to eliminate any redundant rows in ``A_eq``.
            Set False if ``A_eq`` is known to be of full row rank, or if you are
            looking for a potential speedup (at the expense of reliability).
        rr_method : string
            Method used to identify and remove redundant rows from the
            equality constraint matrix after presolve.
            
        '''
        self.A_ub, self.b_ub, self.A_eq, self.b_eq = A_ub, b_ub, A_eq, b_eq
        self.tol = tol
        self.rr, self.rr_method = rr, rr_method
        
    def transform(self):
        A_ub, b_ub, A_eq, b_eq = self.A_ub, self.b_ub, self.A_eq, self.b_eq
        tol = self.tol

        if sps.issparse(A_eq):
            A_eq = A_eq.tocsr()
            A_ub = A_ub.tocsr()

            def where(A):
                return A.nonzero()

            vstack = sps.vstack
        else:
            where = np.where
            vstack = np.vstack
        

        # zero row in equality constraints
        zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
        if np.any(zero_row):
            if np.any(
                np.logical_and(
                    zero_row,
                    np.abs(b_eq) > tol)):  # test_zero_row_1
                # infeasible if RHS is not zero

                raise LinAlgError("The problem is (trivially) infeasible due to a row "
                        "of zeros in the equality constraint matrix with a "
                        "nonzero corresponding constraint value.")
            else:  # test_zero_row_2
                # if RHS is zero, we can eliminate this equation entirely
                A_eq = A_eq[np.logical_not(zero_row), :]
                b_eq = b_eq[np.logical_not(zero_row)]

        # zero row in inequality constraints
        zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
        if np.any(zero_row):
            if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
                # infeasible if RHS is less than zero (because LHS is zero)
                raise LinAlgError("The problem is (trivially) infeasible due to a row "
                        "of zeros in the equality constraint matrix with a "
                        "nonzero corresponding  constraint value.")

            else:  # test_zero_row_2
                # if LHS is >= 0, we can eliminate this constraint entirely
                A_ub = A_ub[np.logical_not(zero_row), :]
                b_ub = b_ub[np.logical_not(zero_row)]   

        # zero column in (both) constraints
        # this indicates that a variable isn't constrained and can be removed
        A = vstack((A_eq, A_ub))
        if A.shape[0] > 0:
            zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
            if np.any(zero_col):
                raise LinAlgError("The problem is (trivially) unbounded "
                        "due  to a zero column in the constraint matrices.")
        
        # no constraints indicates that problem is trivial
        if A_eq.size == 0 and A_ub.size == 0:
            raise LinAlgError("The problem is (trivially) unbounded "
                        "because there are no non-trivial constraints and "
                        "a) at least one decision variable is unbounded "
                        "above and its corresponding cost is negative, or "
                        "b) at least one decision variable is unbounded below "
                        "and its corresponding cost is positive. ")



        # remove redundant (linearly dependent) rows from equality constraints
        rr, rr_method = self.rr, self.rr_method
        n_rows_A = A_eq.shape[0]
        redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                            "improve performance, check the problem formulation "
                            "for redundant equality constraints.")
        if (sps.issparse(A_eq)):
            if rr and A_eq.size > 0:  # TODO: Fast sparse rank check?
                rr_res = _remove_redundancy_pivot_sparse(A_eq, b_eq)
                A_eq, b_eq, status, message = rr_res
                if A_eq.shape[0] < n_rows_A:
                    warn(redundancy_warning)
                # if status != 0:
                #     complete = True


        # This is a wild guess for which redundancy removal algorithm will be
        # faster. More testing would be good.
        small_nullspace = 5
        if rr and A_eq.size > 0:
            try:  # TODO: use results of first SVD in _remove_redundancy_svd
                rank = np.linalg.matrix_rank(A_eq)
            # oh well, we'll have to go with _remove_redundancy_pivot_dense
            except Exception:
                rank = 0
        if rr and A_eq.size > 0 and rank < A_eq.shape[0]:
            warn(redundancy_warning)
            dim_row_nullspace = A_eq.shape[0]-rank
            if rr_method is None:
                if dim_row_nullspace <= small_nullspace:
                    rr_res = _remove_redundancy_svd(A_eq, b_eq)
                    A_eq, b_eq, status, message = rr_res
                if dim_row_nullspace > small_nullspace or status == 4:
                    rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                    A_eq, b_eq, status, message = rr_res

            else:
                rr_method = rr_method.lower()
                if rr_method == "svd":
                    rr_res = _remove_redundancy_svd(A_eq, b_eq)
                    A_eq, b_eq, status, message = rr_res
                elif rr_method == "pivot":
                    rr_res = _remove_redundancy_pivot_dense(A_eq, b_eq)
                    A_eq, b_eq, status, message = rr_res
                elif rr_method == "id":
                    rr_res = _remove_redundancy_id(A_eq, b_eq, rank)
                    A_eq, b_eq, status, message = rr_res
                else:  # shouldn't get here; option validity checked above
                    pass
            if A_eq.shape[0] < rank:
                message = ("Due to numerical issues, redundant equality "
                        "constraints could not be removed automatically. "
                        "Try providing your constraint matrices as sparse "
                        "matrices to activate sparse presolve, try turning "
                        "off redundancy removal, or try turning off presolve "
                        "altogether.")
                status = 4
            if status != 0:
                complete = True
        return ( A_ub, b_ub, A_eq, b_eq )
