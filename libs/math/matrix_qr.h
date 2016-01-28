/*
 * Copyright (C) 2015, Daniel Thuerck, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_MATRIX_QR_HEADER
#define MATH_MATRIX_QR_HEADER

#include <vector>
#include <algorithm>
#include <iostream>

#include "math/defines.h"
#include "math/matrix.h"

MATH_NAMESPACE_BEGIN

/**
 * Matrix QR decomposition for compile-time fixed-size matrices.
 * Uses the dynamic-size matrices interface in the background.
 */
template <typename T, int M, int N>
void
matrix_qr (Matrix<T, M, N> const& mat_a, Matrix<T, M, M>* mat_q,
    Matrix<T, M, N>* mat_r, T const& epsilon = T(1e-12));

/**
 * Calculates a QR decomposition for a given matrix A. A is MxN,
 * Q is MxM and R is MxN. Uses Givens algorithm for computation.
 *
 * Reference:
 * - "Matrix Computations" by Gloub and Loan, 3rd Edition, page 227.
 */
template <typename T>
void
matrix_qr (T const* mat_a, int rows, int cols,
    T* mat_q, T* mat_r, T const& epsilon = T(1e-12));

MATH_NAMESPACE_END

/* ------------------------- QR Internals ------------------------- */

MATH_NAMESPACE_BEGIN
MATH_INTERNAL_NAMESPACE_BEGIN

/**
 * Creates a householder transformation vector and the coefficient for
 * householder matrix creation. As input, the function uses a column-frame
 * in a given matrix, i.e., mat(subset_row_start:subset_row_end, subset_col).
 */
template <typename T>
void
matrix_householder_vector (T const* input, int length,
    T* vector, T* beta, T const& epsilon, T const& norm_factor)
{  // TODO Swap epsilon and norm_factor
    T sigma(0);
    for (int i = 1; i < length; ++i)
        sigma += MATH_POW2(input[i] / norm_factor);

    vector[0] = T(1);
    for (int i = 1; i < length; ++i)
        vector[i] = input[i] / norm_factor;

    if (MATH_EPSILON_EQ(sigma, T(0), epsilon))
    {
        *beta = T(0);
        return;
    }

    T first = input[0] / norm_factor;
    T mu = std::sqrt(MATH_POW2(first) + sigma);
    if (first < epsilon)
        vector[0] = first - mu;
    else
        vector[0] = -sigma / (first + mu);

    first = vector[0];
    *beta = T(2) * MATH_POW2(first) / (sigma + MATH_POW2(first));
    for (int i = 0; i < length; ++i)
        vector[i] /= first;
}

/**
 * Applies a given householder matrix to a frame in a given matrix with
 * offset (offset_rows, offset_cols) from the left side.
 */
template <typename T>
void
matrix_apply_householder_vector_left (T* mat_a, int rows, int cols,
    T const* house_vector, const T& beta, int offset)
{
    /* Calculate the blocked [reusable!] part of the Householder matrix. */
    std::vector<T> vA(cols, 0);
    for (int i = 0; i < (rows - offset); ++i)
        for (int j = 0; j < cols; ++j)
            vA[j] += house_vector[i] * mat_a[(offset + i) * cols + j];

    /* Apply householder matrix. */
    mat_a[offset * cols] -= beta * house_vector[0] * vA[0];
    for (int i = 1; i < (rows - offset); ++i)
        mat_a[(offset + i) * cols] = 0;
    for (int i = 0; i < (rows - offset); ++i)
        for (int j = 1; j < cols; ++j)
            mat_a[(offset + i) * cols + j] -= beta * house_vector[i] * vA[j];
}

/**
 * Applies a given householder matrix to a frame in a given matrix with
 * offset (offset_rows, offset_cols) from the right side.
 */
template <typename T>
void
matrix_apply_householder_vector_right (T* mat_a, int rows, int cols,
    T const* house_vector, const T& beta, int offset)
{
    /* Calculate the blocked [reusable!] part of the Householder matrix. */
    std::vector<T> Av(rows, 0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < (cols - offset); ++j)
            Av[i] += mat_a[i * cols + (offset + j)] * house_vector[j];

    /* Apply householder matrix. */
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < (cols - offset); ++j)
            mat_a[i * cols + (offset + j)] -= beta * Av[i] * house_vector[j];
}

MATH_INTERNAL_NAMESPACE_END
MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T, int M, int N>
void
matrix_qr (Matrix<T, M, N> const& mat_a, Matrix<T, M, M>* mat_q,
    Matrix<T, M, N>* mat_r, T const& epsilon)
{
    matrix_qr(mat_a.begin(), M, N, mat_q->begin(), mat_r->begin(), epsilon);
}

template <typename T>
void
matrix_qr (T const* mat_a, int rows, int cols,
    T* mat_q, T* mat_r, T const& epsilon)
{
    /* Prepare Q and R: Copy A to R. */
    std::copy(mat_a, mat_a + rows * cols, mat_r);
    std::fill(mat_q, mat_q + rows * rows, T(0));
    for (int i = 0; i < rows; ++i)
        mat_q[i * rows + i] = T(1);

    T beta;
    std::vector<T> input(rows);
    std::vector<T> house(rows);

    /* Use successive Householder transformations. */
    for (int j = 0; j < cols; ++j)
    {
        for (int i = j; i < rows; ++i)
            input[i - j] = mat_r[i * cols + j];

        math::internal::matrix_householder_vector(&input[0], rows - j,
            &house[0], &beta, epsilon, T(1));
        math::internal::matrix_apply_householder_vector_right(mat_q,
            rows, rows, &house[0], beta, j);
        math::internal::matrix_apply_householder_vector_left(mat_r,
            rows, cols, &house[0], beta, j);
    }
}

template <typename T>
void
matrix_blocked_qr (T const* mat_a, int rows, int cols,
    T* mat_w, T* mat_y, T* mat_r, T const& epsilon)
{
    /* Prepare Q and R: Copy A to R. */
    std::copy(mat_a, mat_a + rows * cols, mat_r);

    T beta;
    std::vector<T> input(rows);
    std::vector<T> house(rows);
    std::vector<T> Ytv(cols);

    /* Use successive Householder transformations. */
    for (int j = 0; j < cols; ++j)
    {
        for (int i = j; i < rows; ++i)
            input[i - j] = mat_r[i * cols + j];

        math::internal::matrix_householder_vector(&input[0], rows - j,
            &house[0], &beta, epsilon, T(1));
        math::internal::matrix_apply_householder_vector_left(mat_r,
            rows, cols, &house[0], beta, j);

        /* Save Q in a blocked format (Q = I + WY^t). */
        for (int i = j; i < rows; ++i)
            mat_y[i * cols + j] = house[i - j];

        if (j == 0)
        {
            for (int i = j; i < rows; ++i)
                mat_w[i * cols + j] = -beta * house[i - j];
        }
        else
        {
            int const r = j;

            /* Compute Y^T * v. */
            for (int i = 0; i < r; ++i)
            {
                Ytv[i] = T(0);
                for (int k = j; k < rows; ++k)
                    Ytv[i] += mat_y[k * cols + i] * house[k - j];
            }

            for (int i = 0; i < rows; ++i)
            {
                mat_w[i * cols + j] = (i < j ? T(0) : house[i - j]);
                for (int k = 0; k < j; ++k)
                    mat_w[i * cols + j] += mat_w[i * cols + k] * Ytv[k];
                mat_w[i * cols + j] *= -beta;
            }
        }
    }
}

template <typename T>
void
matrix_apply_blocked_qr(T const* mat_r, int rows, int cols,
    T const* mat_w, T const* mat_y, T* result)
{
    std::vector<T> YtR(cols * cols);

    /* Precompute Y^t * R. */
    for (int i = 0; i < cols; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            YtR[i * cols + j] = T(0);
            for (int k = 0; k < rows; ++k)
                YtR[i * cols + j] += mat_y[k * cols + i] * mat_r[k * cols + j];
        }
    }

    /* Compute (I + WY^t) * R = R + W * (Y^t * R). */
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result[i * cols + j] = mat_r[i * cols + j];
            for (int k = 0; k < cols; ++k)
                result[i * cols + j] += mat_w[i * cols + k] * YtR[k * cols + j];
        }
    }
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_QR_HEADER */
