import torch


def spsolve_triangular(crowIdx, colIdx, val, b, lower=True):
    """
    Solve the equation ``A x = b`` for `x`, assuming A is a triangular matrix.
    Parameters
    A sparse square triangular matrix of dimension m X n. Should be in CSR format.
    ----------
    crowIdx : a vector of length m + 1
        The cumulative row indices
    colIdx : a vector of length nnz
        The column indices
    val : a vector of length nnz
        The nonzero values in A
    b : (m,) or (m, n) array_like
        Right-hand side matrix in ``A x = b``
    lower : bool, optional
        Whether `A` is a lower or upper triangular matrix.
        Default is lower triangular matrix.
    unit_diagonal : bool, optional
        If True, diagonal elements of `a` are assumed to be 1 and will not be
        referenced.
        .. versionadded:: 1.4.0
    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system ``A x = b``. Shape of return matches shape
        of `b`.
    """

    # Init x as (a copy of) b.
    x = b.clone()

    # Choose forward or backward order.
    if lower:
        row_indices = range(len(b))
    else:
        row_indices = range(len(b) - 1, -1, -1)

    # Fill x iteratively.
    for i in row_indices:

        # Get indices for i-th row.
        indptr_start = crowIdx[i]
        indptr_stop = crowIdx[i + 1]

        if lower:
            A_diagonal_index_row_i = indptr_stop - 1
            A_off_diagonal_indices_row_i = slice(indptr_start, indptr_stop - 1)
        else:
            A_diagonal_index_row_i = indptr_start
            A_off_diagonal_indices_row_i = slice(indptr_start + 1, indptr_stop)

        # Incorporate off-diagonal entries.
        A_column_indices_in_row_i = colIdx[A_off_diagonal_indices_row_i]
        A_values_in_row_i = val[A_off_diagonal_indices_row_i]
        x[i] -= torch.matmul(x[A_column_indices_in_row_i].t(),
                             A_values_in_row_i)

        # Compute i-th entry of x.
        x[i] /= val[A_diagonal_index_row_i]

    return x
