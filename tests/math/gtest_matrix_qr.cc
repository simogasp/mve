/*
 * Test cases for matrix singular value decomposition (SVD).
 * Written by Simon Fuhrmann and Daniel Thuerck.
 */

#include <iostream>
#include <gtest/gtest.h>

#include "math/matrix_tools.h"
#include "math/matrix_qr.h"

TEST(MatrixQRTest, MatrixApplyHHR_Full)
{
    math::Matrix<double, 20, 10> A;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 10; ++j)
            A(i, j) = i * 20 + j;

    std::vector<double> input(10);
    for (int i = 0; i < 10; ++i)
        input[i] = A(0, i);

    std::vector<double> house(10);
    double beta;
    math::internal::matrix_householder_vector(&input[0], 10, &house[0],
        &beta, 1e-6, 1.0);
    math::internal::matrix_apply_householder_vector_right(A.begin(),
        20, 10, &house[0], beta, 0);

    for (int i = 1; i < 10; ++i)
    {
        EXPECT_NEAR(A(0, i), 0, 1e-6);
    }
}

TEST(MatrixQRTest, MatrixApplyHHR_Part)
{
    math::Matrix<double, 20, 10> A;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 10; ++j)
            A(i, j) = i * 20 + j;

    std::vector<double> input(4);
    for (int i = 0; i < 4; ++i)
        input[i] = A(6, 6 + i);

    std::vector<double> house(4);
    double beta;
    math::internal::matrix_householder_vector(&input[0], 4, &house[0],
        &beta, 1e-6, 1.0);
    math::internal::matrix_apply_householder_vector_right(A.begin(),
        20, 10, &house[0], beta, 6);

    for (int i = 7; i < 10; ++i)
    {
        EXPECT_NEAR(A(6, i), 0, 1e-6);
    }
}

TEST(MatrixQRTest, MatrixApplyHHL_Full)
{
    math::Matrix<double, 20, 10> A;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 10; ++j)
            A(i, j) = i * 20 + j;

    std::vector<double> input(20);
    for (int i = 0; i < 20; ++i)
        input[i] = A(i, 0);

    std::vector<double> house(20);
    double beta;
    math::internal::matrix_householder_vector(&input[0], 20, &house[0],
        &beta, 1e-6, 1.0);

    math::internal::matrix_apply_householder_vector_left(A.begin(),
        20, 10, &house[0], beta, 0);

    for (int i = 1; i < 20; ++i)
    {
        EXPECT_NEAR(A(i, 0), 0, 1e-6);
    }
}

TEST(MatrixQRTest, MatrixApplyHHL_Part)
{
    math::Matrix<double, 20, 10> A;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 10; ++j)
            A(i, j) = i * 20 + j;

    std::vector<double> input(12);
    for (int i = 0; i < 12; ++i)
        input[i] = A(8 + i, 8);

    std::vector<double> house(12);
    double beta;
    math::internal::matrix_householder_vector(&input[0], 12, &house[0],
        &beta, 1e-6, 1.0);
    math::internal::matrix_apply_householder_vector_left(A.begin(),
        20, 10, &house[0], beta, 8);

    for (int i = 9; i < 20; ++i)
    {
        EXPECT_NEAR(A(i, 8), 0, 1e-6);
    }
}

TEST(MatrixSVDTest, MatrixHouseholderOnZeroTest)
{
    double vec_zero[2];
    vec_zero[0] = 0;
    vec_zero[1] = 0;

    double house_vector[2];
    double house_beta;

    math::internal::matrix_householder_vector(vec_zero, 2, house_vector,
        &house_beta, 1e-13, 1.0);

    EXPECT_NEAR(1.0, house_vector[0], 1e-13);
    EXPECT_NEAR(0.0, house_vector[1], 1e-13);
    EXPECT_NEAR(0.0, house_beta, 1e-13);
}

TEST(MatrixSVDTest, MatrixHouseholderNormalTest)
{
    double vec[3];
    vec[0] = 1;
    vec[1] = 4;
    vec[2] = 7;

    double house_vector[3];
    double house_beta;

    math::internal::matrix_householder_vector(vec, 3, house_vector, &house_beta,
        1e-13, 1.0);

    EXPECT_NEAR(1.0, house_vector[0], 1e-13);
    EXPECT_NEAR(-0.561479286439136, house_vector[1], 1e-13);
    EXPECT_NEAR(-0.982588751268488, house_vector[2], 1e-13);
    EXPECT_NEAR(0.876908509020667, house_beta, 1e-13);
}

TEST(MatrixQRTest, MatrixQRQuadraticTest)
{
    // create simple test matrices
    double mat[3 * 3];
    for (int i = 0; i < 9; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
    }

    double mat_q[3 * 3];
    double mat_r[3 * 3];

    math::matrix_qr(mat, 3, 3, mat_q, mat_r, 1e-14);

    double result[3 * 3];
    math::matrix_multiply(mat_q, 3, 3, mat_r, 3, result);

    // compare results up to EPSILON
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(result[i], mat[i], 1e-14);
    }
}

TEST(MatrixQRTest, MatrixQRRectangularTest)
{
    // create test matrix
    double mat[5 * 4];
    for (int i = 0; i < 20; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
    }

    double mat_q[5 * 5];
    double mat_r[5 * 4];
    math::matrix_qr(mat, 5, 4, mat_q, mat_r, 1e-14);

    double result[5 * 4];
    math::matrix_multiply(mat_q, 5, 5, mat_r, 4, result);

    for (int i = 0; i < 20; ++i)
    {
        EXPECT_NEAR(result[i], mat[i], 1e-14);
    }
}

TEST(MatrixQRTest, MatrixQRLargeTest)
{
    // create test matrix
    const int M = 200;
    const int N = 100;

    double * mat = new double[M * N];
    for (int i = 0; i < M * N; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
    }

    double * mat_q = new double[M * M];
    double * mat_r = new double[M * N];
    math::matrix_qr(mat, M, N, mat_q, mat_r, 1e-14);

    double * result = new double[M * N];
    math::matrix_multiply(mat_q, M, M, mat_r, N, result);

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(result[i], mat[i], 1e-8);
    }

    delete[] mat;
    delete[] mat_q;
    delete[] mat_r;
    delete[] result;
}

TEST(MatrixQRTest, MatrixQRScalarTest)
{
    double mat[1];
    mat[0] = 1;

    double mat_q[1];
    double mat_r[1];

    math::matrix_qr(mat, 1, 1, mat_q, mat_r, 1e-14);

    EXPECT_NEAR(mat_q[0] * mat_r[0], mat[0], 1e-14);
}

TEST(MatrixQRTest, MatrixQRVectorTest)
{
    double mat[2];
    mat[0] = 1;
    mat[1] = 2;

    double mat_q[4];
    double mat_r[2];

    math::matrix_qr(mat, 2, 1, mat_q, mat_r, 1e-14);

    double result[2];
    math::matrix_multiply(mat_q, 2, 2, mat_r, 1, result);

    // compare results up to EPSILON
    EXPECT_NEAR(result[0], mat[0], 1e-14);
    EXPECT_NEAR(result[1], mat[1], 1e-14);
}

TEST(MatrixQRTest, TestMatrixInterface)
{
    double A_values[] = { 1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12 };
    math::Matrix<double, 4, 3> A(A_values);
    math::Matrix<double, 4, 4> Q;
    math::Matrix<double, 4, 3> R;
    math::matrix_qr(A, &Q, &R, 1e-16);

    /* Check if Q is orthogonal. */
    math::Matrix<double, 4, 4> QQ1 = Q * Q.transposed();
    EXPECT_TRUE(math::matrix_is_identity(QQ1, 1e-14));
    math::Matrix<double, 4, 4> QQ2 = Q.transposed() * Q;
    EXPECT_TRUE(math::matrix_is_identity(QQ2, 1e-14));

    /* Check if R is upper diagonal. */
    for (int y = 1; y < 4; ++y)
        for (int x = 0; x < y; ++x)
            EXPECT_NEAR(0.0, R(y, x), 1e-14);

    /* Check if A can be reproduced. */
    math::Matrix<double, 4, 3> newA = Q * R;
    for (int i = 0; i < 12; ++i)
        EXPECT_NEAR(newA[i], A[i], 1e-14);
}

TEST(MatrixQRTest, BeforeAfter1)
{
    math::Matrix<double, 2, 2> A;
    A(0,0) = 1.0f; A(0,1) = 2.0f;
    A(1,0) = -3.0f; A(1,1) = 4.0f;

    math::Matrix<double, 2, 2> Q, R;
    math::matrix_qr(A, &Q, &R);
    EXPECT_TRUE(A.is_similar(Q * R, 1e-14));
    EXPECT_NEAR(0.0, R(1,0), 1e-14);
    EXPECT_NEAR(0.0, Q.col(0).dot(Q.col(1)), 1e-14);
}

TEST(MatrixQRTest, BeforeAfter2)
{
    math::Matrix<double, 3, 3> A;
    A(0,0) = 1.0f; A(0,1) = 2.0f; A(0,2) = 8.0f;
    A(1,0) = 2.0f; A(1,1) = -3.0f; A(1,2) = 18.0f;
    A(2,0) = -4.0f; A(2,1) = 5.0f; A(2,2) = -2.0f;

    math::Matrix<double, 3, 3> Q, R;
    math::matrix_qr(A, &Q, &R);
    EXPECT_TRUE(A.is_similar(Q * R, 1e-12));
    EXPECT_NEAR(0.0, R(1,0), 1e-12);
    EXPECT_NEAR(0.0, R(2,0), 1e-12);
    EXPECT_NEAR(0.0, R(2,1), 1e-12);
    EXPECT_NEAR(0.0, Q.col(0).dot(Q.col(1)), 1e-12);
    EXPECT_NEAR(0.0, Q.col(1).dot(Q.col(2)), 1e-12);
    EXPECT_NEAR(0.0, Q.col(0).dot(Q.col(2)), 1e-12);
}

TEST(MatrixQRTest, BlockedQRTest)
{
    // create test matrix
    double mat[5 * 4];
    for (int i = 0; i < 20; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
    }

    double mat_w[5 * 4];
    double mat_y[5 * 4];
    double mat_r[5 * 4];
    math::matrix_blocked_qr(mat, 5, 4, mat_w, mat_y, mat_r, 1e-14);

    // apply YW instead of Q
    double result[5 * 4];
    math::matrix_apply_blocked_qr(mat_r, 5, 4, mat_w, mat_y, result);

    for (int i = 0; i < 20; ++i)
    {
        EXPECT_NEAR(result[i], mat[i], 1e-14);
    }
}

TEST(MatrixQRTest, MatrixBlockedQRLargeTest)
{
    // create test matrix
    int const M = 200;
    int const N = 100;

    double* mat = new double[M * N];
    for (int i = 0; i < M * N; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
    }

    double* mat_w = new double[M * N];
    double* mat_y = new double[M * N];
    double* mat_r = new double[M * N];
    math::matrix_blocked_qr(mat, M, N, mat_w, mat_y, mat_r, 1e-14);

    double* result = new double[M * N];
    math::matrix_apply_blocked_qr(mat_r, M, N, mat_w, mat_y, result);

    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(result[i], mat[i], 1e-7);
    }

    delete[] mat;
    delete[] mat_w;
    delete[] mat_y;
    delete[] mat_r;
    delete[] result;
}
