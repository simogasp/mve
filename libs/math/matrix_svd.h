/*
 * Copyright (C) 2015, Daniel Thuerck, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * The matrix formats for this implementation are exemplary visualized:
 *
 * A A   U U
 * A A = U U * S S * V V
 * A A   U U   S S   V V
 *
 *                 S S S   V V V
 * A A A = U U U * S S S * V V V
 * A A A   U U U   S S S   V V V
 */

#ifndef MATH_MATRIX_SVD_HEADER
#define MATH_MATRIX_SVD_HEADER

#include <vector>

#include "math/defines.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "math/matrix_qr.h"

MATH_NAMESPACE_BEGIN

/**
 * SVD for dynamic-size matrices A of size MxN (M rows, N columns).
 * The function decomposes input matrix A such that A = USV^T where
 * A is MxN, U is MxN, S is a N-vector and V is NxN.
 * Any of U, S or V can be null, however, this does not save operations.
 *
 * Usually, M >= N, i.e. the input matrix has more rows than columns.
 * If M > 5/3 N, QR decomposition is used to do an economy SVD after Chan
 * that saves some operations. This SVD also handles the case where M < N. In
 * this case, zero rows are internally added to A until A is a square matrix.
 *
 * References:
 * - "Matrix Computations" by Gloub and Loan (page 455, algo 8.6.2, [GK-SVD])
 * - "An Improved Algorithm for Computing the SVD" by Chan (1987) [R-SVD].
 */
template <typename T>
void
matrix_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon = T(1e-12));

/**
 * SVD for compile-time fixed-size matrices. The implementation of this
 * function uses the dynamic-size matrices interface in the background.
 * Any of the results can be null, however, this does not save operations.
 */
template <typename T, int M, int N>
void
matrix_svd (Matrix<T, M, N> const& mat_a, Matrix<T, M, N>* mat_u,
    Matrix<T, N, N>* mat_s, Matrix<T, N, N>* mat_v,
    T const& epsilon = T(1e-12));

/**
 * Computes the Moore–Penrose pseudoinverse of matrix A using the SVD.
 * Let the SVD of A be A = USV*, then the pseudoinverse is A' = VS'U*.
 * The inverse S' of S is obtained by taking the reciprocal of non-zero
 * diagonal elements, leaving zeros (up to the epsilon) in place.
 */
template <typename T, int M, int N>
void
matrix_pseudo_inverse (Matrix<T, M, N> const& A,
    Matrix<T, N, M>* result, T const& epsilon = T(1e-12));

MATH_NAMESPACE_END

/* ------------------------ SVD Internals ------------------------- */

MATH_NAMESPACE_BEGIN
MATH_INTERNAL_NAMESPACE_BEGIN

/**
 * Checks whether the lower-right square sub-matrix of size KxK is enclosed by
 * zeros (up to some epsilon) within a square matrix of size MxM. This check
 * is SVD specific and probably not very useful for other code.
 * Note: K must be smaller than or equal to M.
 */
template <typename T>
bool
matrix_is_submatrix_zero_enclosed (T const* mat, int m, int k, T const& epsilon)
{
    int const j = m - k - 1;
    if (j < 0)
        return true;
    for (int i = m - k; i < m; ++i)
        if (!MATH_EPSILON_EQ(mat[j * m + i], T(0), epsilon)
            || !MATH_EPSILON_EQ(mat[i * m + j], T(0), epsilon))
            return false;
    return true;
}

/**
 * Checks whether the super-diagonal (above the diagonal) of a MxN matrix
 * does not contain zeros up to some epsilon.
 */
template <typename T>
bool
matrix_is_superdiagonal_nonzero (T const* mat,
    int rows, int cols, T const& epsilon)
{
    int const n = std::min(rows, cols) - 1;
    for (int i = 0; i < n; ++i)
        if (MATH_EPSILON_EQ(T(0), mat[i * cols + i + 1], epsilon))
            return false;
    return true;
}

/**
 * Returns the larger eigenvalue of the given 2x2 matrix. The eigenvalues of
 * the matrix are assumed to be non-complex and a negative root set to zero.
 */
template <typename T>
void
matrix_2x2_eigenvalues (T const* mat, T* smaller_ev, T* larger_ev)
{
    /* For matrix [a b; c d] solve (a+d) / 2 + sqrt((a+d)^2 / 4 - ad + bc). */
    T const& a = mat[0];
    T const& b = mat[1];
    T const& c = mat[2];
    T const& d = mat[3];

    T x = MATH_POW2(a + d) / T(4) - a * d + b * c;
    x = (x > T(0) ? std::sqrt(x) : T(0));
    *smaller_ev = (a + d) / T(2) - x;
    *larger_ev = (a + d) / T(2) + x;
}

/**
 * Calculates the Givens rotation coefficients c and s by solving
 * [alpha beta] [c s; -c s] = [sqrt(alpha^2 + beta^2) 0].
 */
template <typename T>
void
matrix_givens_rotation (T const& alpha, T const& beta,
    T* givens_c, T* givens_s, T const& epsilon)
{
    if (MATH_EPSILON_EQ(beta, T(0), epsilon))
    {
        *givens_c = T(1);
        *givens_s = T(0);
        return;
    }

    if (std::abs(beta) > std::abs(alpha))
    {
        T tao = -alpha / beta;
        *givens_s = T(1) / std::sqrt(T(1) + tao * tao);
        *givens_c = *givens_s * tao;
    }
    else
    {
        T tao = -beta / alpha;
        *givens_c = T(1) / std::sqrt(T(1) + tao * tao);
        *givens_s = *givens_c * tao;
    }
}

/**
 * Applies a Givens rotation for columns (givens_i, givens_k)
 * by only rotating the required set of columns in-place.
 */
template <typename T>
void
matrix_apply_givens_column (T* mat, int rows, int cols, int givens_i,
    int givens_k, T const& givens_c, T const& givens_s)
{
    for (int j = 0; j < rows; ++j)
    {
        T const tao1 = mat[j * cols + givens_i];
        T const tao2 = mat[j * cols + givens_k];
        mat[j * cols + givens_i] = givens_c * tao1 - givens_s * tao2;
        mat[j * cols + givens_k] = givens_s * tao1 + givens_c * tao2;
    }
}

/**
 * Applies a transposed Givens rotation for rows (givens_i, givens_k)
 * by only rotating the required set of rows in-place.
 */
template <typename T>
void
matrix_apply_givens_row (T* mat, int /*rows*/, int cols, int givens_i,
    int givens_k, T const& givens_c, T const& givens_s)
{
    for (int j = 0; j < cols; ++j)
    {
        T const tao1 = mat[givens_i * cols + j];
        T const tao2 = mat[givens_k * cols + j];
        mat[givens_i * cols + j] = givens_c * tao1 - givens_s * tao2;
        mat[givens_k * cols + j] = givens_s * tao1 + givens_c * tao2;
    }
}

/**
 * Bidiagonalizes a given MxN matrix, resulting in a MxN matrix U,
 * a bidiagonal MxN matrix B and a NxN matrix V.
 *
 * Reference: "Matrix Computations" by Golub and Loan, 3rd edition,
 * from page 252 (algorithm 5.4.2).
 */
template <typename T>
void
matrix_bidiagonalize (T const* mat_a, int rows, int cols, T* mat_u,
    T* mat_b, T* mat_v, T const& epsilon)
{
    /* Initialize U and V with identity matrices. */
    matrix_set_identity(mat_u, rows);
    matrix_set_identity(mat_v, cols);

    /* Copy mat_a into mat_b. */
    std::copy(mat_a, mat_a + rows * cols, mat_b);

    std::vector<T> buffer(4 * std::max(rows, cols));
    T* input_vec = &buffer[0];
    T* house_vec = input_vec + std::max(rows, cols);

    int const steps = (rows == cols) ? (cols - 1) : cols;
    for (int k = 0; k < steps; ++k)
    {
        int const sub_length = rows - k;
        T house_beta;

        for (int i = 0; i < sub_length; ++i)
            input_vec[i] = mat_b[(k + i) * cols + k];

        matrix_householder_vector(input_vec, sub_length, house_vec,
            &house_beta, epsilon, T(1));
        matrix_apply_householder_vector_left(mat_b, rows, cols,
            house_vec, house_beta, k);
        matrix_apply_householder_vector_right(mat_u, rows, rows,
            house_vec, house_beta, k);

        if (k < cols - 2)
        {
            /* Normalization constant for numerical stability. */
            T norm(0);
            for (int i = k + 1; i < cols; ++i)
                norm += mat_b[k * cols + i];
            if (MATH_EPSILON_EQ(norm, T(0), epsilon))
                norm = T(1);

            int const inner_sub_length = cols - (k + 1);
            T* inner_input_vec = house_vec + std::max(rows, cols);
            T* inner_house_vec = inner_input_vec + std::max(rows, cols);

            for (int i = 0; i < inner_sub_length; ++i)
                inner_input_vec[i] = mat_b[k * cols + (k + 1 + i)];

            T inner_house_beta;
            matrix_householder_vector(inner_input_vec, inner_sub_length,
                inner_house_vec, &inner_house_beta, epsilon, norm);
            matrix_apply_householder_vector_right(mat_b, rows, cols,
                inner_house_vec, inner_house_beta, k + 1);
            matrix_apply_householder_vector_right(mat_v, cols, cols,
                inner_house_vec, inner_house_beta, k + 1);
        }
    }
}

/**
 * Single step in the [GK-SVD] method.
 */
template <typename T>
void
matrix_gk_svd_step (int rows, int cols, T* mat_b, T* mat_q, T* mat_p,
    int p, int q, T const& epsilon)
{
    int const slice_length = cols - q - p;
    int const mat_sizes = slice_length * slice_length;
    std::vector<T> buffer(3 * mat_sizes);
    T* mat_b22 = &buffer[0];
    T* mat_b22_t = mat_b22 + mat_sizes;
    T* mat_tmp = mat_b22_t + mat_sizes;

    for (int i = 0; i < slice_length; ++i)
        for (int j = 0; j < slice_length; ++j)
            mat_b22[i * slice_length + j] = mat_b[(p + i) *  cols + (p + j)];
    for (int i = 0; i < slice_length; ++i)
        for (int j = 0; j < slice_length; ++j)
            mat_b22_t[i * slice_length + j] = mat_b22[j * slice_length + i];

    /* Slice outer product gives covariance matrix. */
    matrix_multiply(mat_b22, slice_length, slice_length, mat_b22_t,
        slice_length, mat_tmp);

    T mat_c[2 * 2];
    mat_c[0] = mat_tmp[(slice_length - 2) * slice_length + (slice_length - 2)];
    mat_c[1] = mat_tmp[(slice_length - 2) * slice_length + (slice_length - 1)];
    mat_c[2] = mat_tmp[(slice_length - 1) * slice_length + (slice_length - 2)];
    mat_c[3] = mat_tmp[(slice_length - 1) * slice_length + (slice_length - 1)];

    /* Use eigenvalue that is closer to the lower right entry of the slice. */
    T eig_1, eig_2;
    matrix_2x2_eigenvalues(mat_c, &eig_1, &eig_2);

    T diff1 = std::abs(mat_c[3] - eig_1);
    T diff2 = std::abs(mat_c[3] - eig_2);
    T mu = (diff1 < diff2) ? eig_1 : eig_2;

    /* Zero another entry by applying givens rotations. */
    int k = p;
    T alpha = mat_b[k * cols + k] * mat_b[k * cols + k] - mu;
    T beta = mat_b[k * cols + k] * mat_b[k * cols + (k + 1)];

    for (int k = p; k < cols - q - 1; ++k)
    {
        T givens_c, givens_s;
        matrix_givens_rotation(alpha, beta, &givens_c, &givens_s, epsilon);
        matrix_apply_givens_column(mat_b, cols, cols, k, k + 1, givens_c,
            givens_s);
        matrix_apply_givens_column(mat_p, cols, cols, k, k + 1, givens_c,
            givens_s);

        alpha = mat_b[k * cols + k];
        beta = mat_b[(k + 1) * cols + k];
        matrix_givens_rotation(alpha, beta, &givens_c, &givens_s, epsilon);
        internal::matrix_apply_givens_row(mat_b, cols, cols, k, k + 1,
            givens_c, givens_s);
        internal::matrix_apply_givens_column(mat_q, rows, cols, k, k + 1,
            givens_c, givens_s);

        if (k < (cols - q - 2))
        {
            alpha = mat_b[k * cols + (k + 1)];
            beta = mat_b[k * cols + (k + 2)];
        }
    }
}

template <typename T>
void
matrix_svd_clear_super_entry(int rows, int cols, T* mat_b, T* mat_q,
    int row_index, T const& epsilon)
{
    for (int i = row_index + 1; i < cols; ++i)
    {
        if (MATH_EPSILON_EQ(mat_b[row_index * cols + i], T(0), epsilon))
        {
            mat_b[row_index * cols + i] = T(0);
            break;
        }

        T norm = MATH_POW2(mat_b[row_index * cols + i])
            + MATH_POW2(mat_b[i * cols + i]);
        norm = std::sqrt(norm) * MATH_SIGN(mat_b[i * cols + i]);

        T givens_c = mat_b[i * cols + i] / norm;
        T givens_s = mat_b[row_index * cols + i] / norm;
        matrix_apply_givens_row(mat_b, cols, cols, row_index,
            i, givens_c, givens_s);
        matrix_apply_givens_column(mat_q, rows, cols, row_index,
            i, givens_c, givens_s);
    }
}

/**
 * Implementation of the [GK-SVD] method.
 */
template <typename T>
void
matrix_gk_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allocate memory for temp matrices. */
    int const mat_q_full_size = rows * rows;
    int const mat_b_full_size = rows * cols;
    int const mat_p_size = cols * cols;
    int const mat_q_size = rows * cols;
    int const mat_b_size = cols * cols;

    std::vector<T> buffer(mat_q_full_size + mat_b_full_size
        + mat_p_size + mat_q_size + mat_b_size);
    T* mat_q_full = &buffer[0];
    T* mat_b_full = mat_q_full + mat_q_full_size;
    T* mat_p = mat_b_full + mat_b_full_size;
    T* mat_q = mat_p + mat_p_size;
    T* mat_b = mat_q + mat_q_size;

    matrix_bidiagonalize(mat_a, rows, cols,
        mat_q_full, mat_b_full, mat_p, epsilon);

    /* Extract smaller matrices. */
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat_q[i * cols + j] = mat_q_full[i * rows + j];
        }
    }
    std::copy(mat_b_full, mat_b_full + cols * cols, mat_b);

    /* Avoid infinite loops and exit after maximum number of iterations. */
    int const max_iterations = rows * rows;
    int iteration = 0;

    while (iteration < max_iterations)
    {
        iteration += 1;

        /* Enforce exact zeros for numerical stability. */
        for (int i = 0; i < (cols - 1); ++i)
            if (MATH_EPSILON_EQ(mat_b[i * cols + i + 1], T(0), epsilon))
                mat_b[i * cols + i + 1] = T(0);

        /* GK 2a. */
        for (int i = 0; i < (cols - 1); ++i)
        {
            if (std::abs(mat_b[i * cols + (i + 1)]) <= epsilon *
                std::abs(mat_b[i * cols + i] + mat_b[(i + 1) * cols + (i + 1)]))
            {
                mat_b[i * cols + (i + 1)] = T(0);
            }
        }

        /* GK 2b. */
        /* Select q such that b33 is diagonal and blocked by zeros. */
        int q = 0;
        for (int k = 0; k < cols; ++k)
        {
            int const slice_len = k + 1;
            std::vector<T> mat_b33(slice_len * slice_len);
            for (int i = 0; i < slice_len; ++i)
            {
                for (int j = 0; j < slice_len; ++j)
                {
                    mat_b33[i * slice_len + j]
                        = mat_b[(cols - k - 1 + i) * cols + (cols - k - 1 + j)];
                }
            }

            if (matrix_is_diagonal(&mat_b33[0], slice_len, slice_len, epsilon))
            {
                if (k < cols - 1)
                {
                    if (matrix_is_submatrix_zero_enclosed(
                        mat_b, cols, k + 1, epsilon))
                    {
                        q = k + 1;
                    }
                }
                else
                {
                    q = k + 1;
                }
            }
        }

        /* Select z := n-p-q such that B22 has no zero superdiagonal entry. */
        int z = 0;
        std::vector<T> mat_b22_tmp((cols - q) * (cols - q));
        for (int k = 0; k < (cols - q); ++k)
        {
            int const slice_len = k + 1;
            for (int i = 0; i < slice_len; ++i)
            {
                for (int j = 0; j < slice_len; ++j)
                {
                    mat_b22_tmp[i * slice_len + j]
                        = mat_b[(cols - q - k - 1 + i)
                        * cols + (cols - q - k - 1 + j)];
                }
            }
            if (matrix_is_superdiagonal_nonzero(
                &mat_b22_tmp[0], slice_len, slice_len, epsilon))
            {
                z = k + 1;
            }
        }

        int const p = cols - q - z;

        /* GK 2c. */
        if (q == cols)
            break;

        bool diagonal_non_zero = true;
        int nz = 0;
        for (nz = p; nz < (cols - q - 1); ++nz)
        {
            if (MATH_EPSILON_EQ(mat_b[nz * cols + nz], T(0), epsilon))
            {
                diagonal_non_zero = false;
                mat_b[nz * cols + nz] = T(0);
                break;
            }
        }

        if (diagonal_non_zero)
            matrix_gk_svd_step(rows, cols, mat_b, mat_q, mat_p, p, q, epsilon);
        else
            matrix_svd_clear_super_entry(rows, cols, mat_b, mat_q, nz, epsilon);
    }

    /* Create resulting matrices and vector from temporary entities. */
    std::copy(mat_q, mat_q + rows * cols, mat_u);
    std::copy(mat_p, mat_p + cols * cols, mat_v);
    for (int i = 0; i < cols; ++i)
        vec_s[i] = mat_b[i * cols + i];

    /* Correct signs. */
    for (int i = 0; i < cols; ++i)
    {
        if (vec_s[i] < epsilon)
        {
            vec_s[i] = -vec_s[i];
            for (int j = 0; j < rows; ++j)
            {
                int index = j * cols + i;
                mat_u[index] = -mat_u[index];
            }
        }
    }
}

/**
 * Implementation of the [R-SVD] method, uses [GK-SVD] as solver
 * for the reduced problem.
 */
template <typename T>
void
matrix_r_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allocate memory for temp matrices. */
    int const mat_w_size = rows * cols;
    int const mat_y_size = rows * cols;
    int const mat_r_size = rows * cols;
    int const mat_u_tmp_size = rows * cols;
    int const buf_size = mat_w_size + mat_y_size + mat_r_size + mat_u_tmp_size;
    std::vector<T> buffer(buf_size);
    T* mat_w = &buffer[0];
    T* mat_y = mat_w + mat_w_size;
    T* mat_r = mat_y + mat_y_size;
    T* mat_u_tmp = mat_r + mat_r_size;

    /* Use a QR-variant that represents Q as 3 matrices W, Y. */
    matrix_blocked_qr(mat_a, rows, cols, mat_w, mat_y, mat_r, epsilon);

    /* Apply SVD on R. */
    matrix_gk_svd(mat_r, cols, cols, mat_u_tmp, vec_s, mat_v, epsilon);
    std::fill(mat_u_tmp + cols * cols, mat_u_tmp + rows * cols, T(0));

    /* Adapt U for big matrices. */
    matrix_apply_blocked_qr(mat_u_tmp, rows, cols, mat_w, mat_y, mat_u);
}

/**
 * Returns the index of the largest eigenvalue. If all eigenvalues are
 * zero, -1 is returned.
 */
template <typename T>
int
find_largest_ev_index (T const* values, int length)
{
    T largest = T(0);
    int index = -1;
    for (int i = 0; i < length; ++i)
        if (values[i] > largest)
        {
            largest = values[i];
            index = i;
        }
    return index;
}

MATH_INTERNAL_NAMESPACE_END
MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T>
void
matrix_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allow for null result matrices. */
    std::vector<T> mat_u_tmp;
    std::vector<T> vec_s_tmp;
    if (vec_s == nullptr)
    {
        vec_s_tmp.resize(cols);
        vec_s = &vec_s_tmp[0];
    }
    std::vector<T> mat_v_tmp;
    if (mat_v == nullptr)
    {
        mat_v_tmp.resize(cols * cols);
        mat_v = &mat_v_tmp[0];
    }

    /*
     * Handle two cases: The regular case, where M >= N (rows >= cols), and
     * the irregular case, where M < N (rows < cols). In the latter one,
     * zero rows are appended to A until A is a square matrix.
     */

    if (rows >= cols)
    {
        /* Allow for null result U matrix. */
        if (mat_u == nullptr)
        {
            mat_u_tmp.resize(rows * cols);
            mat_u = &mat_u_tmp[0];
        }

        /* Perform economy SVD if rows > 5/3 cols to save some operations. */
        if (rows > 5 * cols / 3)
        {
            internal::matrix_r_svd(mat_a, rows, cols,
                mat_u, vec_s, mat_v, epsilon);
        }
        else
        {
            internal::matrix_gk_svd(mat_a, rows, cols,
                mat_u, vec_s, mat_v, epsilon);
        }
    }
    else
    {
        /* Append zero rows to A until A is square. */
        std::vector<T> mat_a_tmp(cols * cols, T(0));
        std::copy(mat_a, mat_a + cols * rows, &mat_a_tmp[0]);

        /* Temporarily resize U, allow for null result matrices. */
        mat_u_tmp.resize(cols * cols);
        internal::matrix_gk_svd(&mat_a_tmp[0], cols, cols,
            &mat_u_tmp[0], vec_s, mat_v, epsilon);

        /* Copy the result back to U leaving out the last rows. */
        if (mat_u != nullptr)
            std::copy(&mat_u_tmp[0], &mat_u_tmp[0] + rows * cols, mat_u);
        else
            mat_u = &mat_u_tmp[0];
    }

    /* Sort the eigenvalues in S and adapt the columns of U and V. */
    for (int i = 0; i < cols; ++i)
    {
        int pos = internal::find_largest_ev_index(vec_s + i, cols - i);
        if (pos < 0)
            break;
        if (pos == 0)
            continue;
        std::swap(vec_s[i], vec_s[i + pos]);
        matrix_swap_columns(mat_u, rows, cols, i, i + pos);
        matrix_swap_columns(mat_v, cols, cols, i, i + pos);
    }
}

template <typename T, int M, int N>
void
matrix_svd (Matrix<T, M, N> const& mat_a, Matrix<T, M, N>* mat_u,
    Matrix<T, N, N>* mat_s, Matrix<T, N, N>* mat_v, T const& epsilon)
{
    T* mat_u_ptr = mat_u ? mat_u->begin() : nullptr;
    T* mat_s_ptr = mat_s ? mat_s->begin() : nullptr;
    T* mat_v_ptr = mat_v ? mat_v->begin() : nullptr;

    matrix_svd<T>(mat_a.begin(), M, N,
        mat_u_ptr, mat_s_ptr, mat_v_ptr, epsilon);

    /* Swap elements of S into place. */
    if (mat_s_ptr)
    {
        std::fill(mat_s_ptr + N, mat_s_ptr + N * N, T(0));
        for (int x = 1, i = N + 1; x < N; ++x, i += N + 1)
            std::swap(mat_s_ptr[x], mat_s_ptr[i]);
    }
}

template <typename T, int M, int N>
void
matrix_pseudo_inverse (Matrix<T, M, N> const& A,
    Matrix<T, N, M>* result, T const& epsilon)
{
    Matrix<T, M, N> U;
    Matrix<T, N, N> S;
    Matrix<T, N, N> V;
    matrix_svd(A, &U, &S, &V, epsilon);

    /* Invert diagonal of S. */
    for (int i = 0; i < N; ++i)
        if (MATH_EPSILON_EQ(S(i, i), T(0), epsilon))
            S(i, i) = T(0);
        else
            S(i, i) = T(1) / S(i, i);

    *result = V * S * U.transposed();
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_SVD_HEADER */
