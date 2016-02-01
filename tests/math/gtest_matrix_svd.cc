/*
 * Test cases for matrix singular value decomposition (SVD).
 * Written by Simon Fuhrmann and Daniel Thuerck.
 */

#include <limits>
#include <gtest/gtest.h>

#include "math/matrix_svd.h"

TEST(MatrixSVDTest, MatrixSimpleTest1)
{
    math::Matrix<double, 3, 2> A;
    A(0, 0) = 1.0;  A(0, 1) = 4.0;
    A(1, 0) = 2.0;  A(1, 1) = 5.0;
    A(2, 0) = 3.0;  A(2, 1) = 6.0;

    math::Matrix<double, 3, 2> U;
    math::Matrix<double, 2, 2> S;
    math::Matrix<double, 2, 2> V;
    math::matrix_svd(A, &U, &S, &V, 1e-10);

    math::Matrix<double, 3, 2> A_svd = U * S * V.transposed();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(A_svd[i], A[i], 1e-13);
}

TEST(MatrixSVDTest, MatrixSimpleTest2)
{
    math::Matrix<double, 2, 3> A;
    A(0, 0) = 1.0;  A(0, 1) = 2.0;  A(0, 2) = 3.0;
    A(1, 0) = 4.0;  A(1, 1) = 5.0;  A(1, 2) = 6.0;

    math::Matrix<double, 2, 3> U;
    math::Matrix<double, 3, 3> S;
    math::Matrix<double, 3, 3> V;
    math::matrix_svd(A, &U, &S, &V, 1e-10);

    math::Matrix<double, 2, 3> A_svd = U * S * V.transposed();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(A_svd[i], A[i], 1e-13);
}

TEST(MatrixSVDTest, MatrixIsSubmatrixZeroEnclosed)
{
    float mat[4 * 4];
    std::fill(mat, mat + 4 * 4, 0.0f);

    // Everything is zero.
    EXPECT_TRUE(math::internal::matrix_is_submatrix_zero_enclosed(mat, 4, 2, 0.0f));

    // Doesn't check diagonally.
    mat[1 * 4 + 1] = 1.0f;
    EXPECT_TRUE(math::internal::matrix_is_submatrix_zero_enclosed(mat, 4, 2, 0.0f));

    // Damage the upper row.
    mat[1 * 4 + 2] = 2.0f;
    EXPECT_FALSE(math::internal::matrix_is_submatrix_zero_enclosed(mat, 4, 2, 0.0f));
    mat[1 * 4 + 2] = 0.0f;

    // Damage the left column.
    mat[2 * 4 + 1] = 3.0f;
    EXPECT_FALSE(math::internal::matrix_is_submatrix_zero_enclosed(mat, 4, 2, 0.0f));
    mat[2 * 4 + 1] = 0.0f;

    // Check with submatrix as large as full matrix.
    EXPECT_TRUE(math::internal::matrix_is_submatrix_zero_enclosed(mat, 4, 4, 0.0f));
}

TEST(MatrixSVDTest, MatrixIsSuperdiagonalNonzero)
{
    float mat[3 * 10];
    std::fill(mat, mat + 3 * 10, 1.0f);
    EXPECT_TRUE(math::internal::matrix_is_superdiagonal_nonzero(mat, 3, 10, 0.0f));
    EXPECT_TRUE(math::internal::matrix_is_superdiagonal_nonzero(mat, 10, 3, 0.0f));

    // Interpret as 3x10. Inject a zero.
    mat[1 * 10 + 2] = 0.0f;
    EXPECT_FALSE(math::internal::matrix_is_superdiagonal_nonzero(mat, 3, 10, 0.0f));
}

TEST(MatrixSVDTest, Matrix2x2Eigenvalues)
{
    math::Matrix2d mat;
    mat(0,0) = 0.318765239858981; mat(0,1) = -0.433592022305684;
    mat(1,0) = -1.307688296305273; mat(1,1) = 0.342624466538650;

    double smaller_ev, larger_ev;
    math::internal::matrix_2x2_eigenvalues(mat.begin(), &smaller_ev, &larger_ev);

    EXPECT_NEAR(-0.422395797795416, smaller_ev, 1e-13);
    EXPECT_NEAR(1.083785504193047, larger_ev, 1e-13);
}

TEST(MatrixSVDTest, MatrixBidiagonalizationStandardTest)
{
    const int M = 5;
    const int N = 4;
    double test_matrix[M * N];
    for (int i = 0; i < M*N; ++i)
    {
        test_matrix[i] = static_cast<double>(i + 1);
    }

    math::Matrix<double, M, M> mat_u;
    math::Matrix<double, M, N> mat_b; 
    math::Matrix<double, N, N> mat_v;

    math::internal::matrix_bidiagonalize(test_matrix, M, N, 
        mat_u.begin(), mat_b.begin(), mat_v.begin(), 1e-13);

    math::Matrix<double, M, N> X = mat_u * mat_b * mat_v.transposed();
    
    for (int i = 0; i < M * N; ++i)
        EXPECT_NEAR(test_matrix[i], X[i], 1e-13);
}

TEST(MatrixSVDTest, MatrixBidiagonalizationLargeTest)
{
    const int M = 50;
    const int N = 100;
    double* test_matrix = new double[M * N];
    for (int i = 0; i < M * N; ++i)
    {
        test_matrix[i] = static_cast<double>(i + 1);
    }

    double* mat_u = new double[M * M];
    double* mat_b = new double[M * N];
    double* mat_v = new double[N * N];

    math::internal::matrix_bidiagonalize(test_matrix, M, N,
        mat_u, mat_b, mat_v, 1e-13);
    math::matrix_transpose(mat_v, N, N);

    double* res_1 = new double[M * N];
    double* res_2 = new double[M * N];

    math::matrix_multiply(mat_u, M, M, mat_b, N, res_1);
    math::matrix_multiply(res_1, M, N, mat_v, N, res_2);

    for (int i = 0; i < M * N; ++i)
        EXPECT_NEAR(test_matrix[i], res_2[i], 1e-10);

    delete[] test_matrix;
    delete[] mat_u;
    delete[] mat_b;
    delete[] mat_v;
    delete[] res_1;
    delete[] res_2;
}

TEST(MatrixSVDTest, MatrixBidiagonalizationQuadraticTest)
{
    const int M = 5;
    double test_matrix[M * M];
    for (int i = 0; i < M*M; ++i)
    {
        test_matrix[i] = static_cast<double>(i + 1);
    }

    math::Matrix<double, M, M> mat_u, mat_b, mat_v;

    math::internal::matrix_bidiagonalize(test_matrix, M, M,
        mat_u.begin(), mat_b.begin(), mat_v.begin(), 1e-13);

    math::Matrix<double, M, M> X = mat_u * mat_b * mat_v.transposed();

    for (int i = 0; i < M * M; ++i)
        EXPECT_NEAR(test_matrix[i], X[i], 1e-13);
}

TEST(MatrixSVDTest, MatrixBidiagonalizationScalarTest)
{
    double mat_a = 2;
    double mat_u, mat_b, mat_v;

    math::internal::matrix_bidiagonalize(&mat_a, 1, 1, &mat_u, &mat_b, &mat_v, 1e-13);
    EXPECT_NEAR(1, mat_u, 1e-13);
    EXPECT_NEAR(2, mat_b, 1e-13);
    EXPECT_NEAR(1, mat_v, 1e-13);
}

TEST(MatrixQRTest, MatrixApplyGivensRotation)
{
    // create simple enumeration matrices
    double mat[5 * 4];
    double mat2[5 * 4];
    for (int i = 0; i < 20; ++i)
    {
        mat[i] = static_cast<double>(i + 1);
        mat2[i] = static_cast<double>(i + 1);
    }

    // ground truth for column rotation
    double groundtruth_mat_col[5 * 4];
    for (int i = 0; i < 20; ++i)
        groundtruth_mat_col[i] = static_cast<double>(i + 1);

    groundtruth_mat_col[0 * 4 + 1] = -3.577708763999663;
    groundtruth_mat_col[1 * 4 + 1] = -8.944271909999159;
    groundtruth_mat_col[2 * 4 + 1] = -14.310835055998654;
    groundtruth_mat_col[3 * 4 + 1] = -19.677398201998148;
    groundtruth_mat_col[4 * 4 + 1] = -25.043961347997644;

    groundtruth_mat_col[0 * 4 + 2] = 0.447213595499958;
    groundtruth_mat_col[1 * 4 + 2] = 2.236067977499790;
    groundtruth_mat_col[2 * 4 + 2] = 4.024922359499622;
    groundtruth_mat_col[3 * 4 + 2] = 5.813776741499454;
    groundtruth_mat_col[4 * 4 + 2] = 7.602631123499284;

    // ground truth for row rotation
    double groundtruth_mat_row[5 * 4];
    for (int i = 0; i < 20; ++i)
        groundtruth_mat_row[i] = static_cast<double>(i + 1);

    groundtruth_mat_row[1 * 4 + 0] = -10.285912696499032;
    groundtruth_mat_row[1 * 4 + 1] = -11.627553482998907;
    groundtruth_mat_row[1 * 4 + 2] = -12.969194269498779;
    groundtruth_mat_row[1 * 4 + 3] = -14.310835055998654;

    groundtruth_mat_row[2 * 4 + 0] = 0.447213595499958;
    groundtruth_mat_row[2 * 4 + 1] = 0.894427190999916;
    groundtruth_mat_row[2 * 4 + 2] = 1.341640786499874;
    groundtruth_mat_row[2 * 4 + 3] = 1.788854381999831;

    double givens_c, givens_s;
    math::internal::matrix_givens_rotation(1.0, 2.0, &givens_c, &givens_s, 1e-14);
    math::internal::matrix_apply_givens_column(mat, 5, 4, 1, 2, givens_c, givens_s);
    math::internal::matrix_apply_givens_row(mat2, 5, 4, 1, 2, givens_c, givens_s);

    // compare results
    for (int i = 0; i < 5 * 4; ++i)
    {
        EXPECT_NEAR(groundtruth_mat_col[i], mat[i], 1e-14);
        EXPECT_NEAR(groundtruth_mat_row[i], mat2[i], 1e-14);
    }
}

TEST(MatrixSVDTest, MatrixSVDQuadraticSTest)
{
    double mat_a[9];
    for (int i = 0; i < 9; ++i)
    {
        mat_a[i] = static_cast<double>(i + 1);
    }

    double mat_u[9];
    double mat_s[3];
    double mat_v[9];

    math::internal::matrix_gk_svd(mat_a, 3, 3, mat_u, mat_s, mat_v, 1e-6);

    double groundtruth_mat_s[3];
    groundtruth_mat_s[0] = 16.848103352614210;
    groundtruth_mat_s[1] = 1.068369514554709;
    groundtruth_mat_s[2] = 0;

    EXPECT_NEAR(groundtruth_mat_s[0], mat_s[0], 1e-13);
    EXPECT_NEAR(groundtruth_mat_s[1], mat_s[1], 1e-13);
    EXPECT_NEAR(groundtruth_mat_s[2], mat_s[2], 1e-13);
}

TEST(MatrixSVDTest, MatrixSVDNonQuadraticFullTest)
{
    double mat_a[20];
    for (int i = 0; i < 20; ++i)
    {
        mat_a[i] = static_cast<double>(i + 1);
    }

    double mat_u[20];
    double mat_s[4];
    double mat_v[20];

    math::internal::matrix_gk_svd(mat_a, 5, 4, mat_u, mat_s, mat_v, 1e-13);

    double groundtruth_mat_s[4];
    groundtruth_mat_s[0] = 53.520222492850067;
    groundtruth_mat_s[1] = 2.363426393147627;
    groundtruth_mat_s[2] = 0;
    groundtruth_mat_s[3] = 0;

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(groundtruth_mat_s[i], mat_s[i], 1e-13);
    }

}

TEST(MatrixSVDTest, MatrixSVDNonQuadraticEconomyTest)
{
    double mat_a[20];
    for (int i = 0; i < 20; ++i)
    {
        mat_a[i] = static_cast<double>(i + 1);
    }

    double mat_u[20];
    double mat_s[4];
    double mat_v[20];

    math::internal::matrix_r_svd(mat_a, 5, 4, mat_u, mat_s, mat_v, 1e-10);

    double groundtruth_mat_s[4];
    groundtruth_mat_s[0] = 53.520222492850067;
    groundtruth_mat_s[1] = 2.363426393147627;
    groundtruth_mat_s[2] = 0;
    groundtruth_mat_s[3] = 0;

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(groundtruth_mat_s[i], mat_s[i], 1e-10);
    }
}

TEST(MatrixSVDTest, MatrixTransposeTest)
{
    int mat_a[6];
    mat_a[0] = 1;
    mat_a[1] = 3;
    mat_a[2] = 5;
    mat_a[3] = 2;
    mat_a[4] = 4;
    mat_a[5] = 6;

    math::matrix_transpose(mat_a, 2, 3);

    int groundtruth_mat_a_t[6];
    groundtruth_mat_a_t[0] = 1;
    groundtruth_mat_a_t[1] = 2;
    groundtruth_mat_a_t[2] = 3;
    groundtruth_mat_a_t[3] = 4;
    groundtruth_mat_a_t[4] = 5;
    groundtruth_mat_a_t[5] = 6;

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(groundtruth_mat_a_t[i], mat_a[i], 1e-13);
    }
}

TEST(MatrixSVDTest, MatrixSVDUnderdeterminedTest)
{
    double mat_a[2 * 3];
    mat_a[0] = 1.0;
    mat_a[1] = 3.0;
    mat_a[2] = 5.0;
    mat_a[3] = 2.0;
    mat_a[4] = 4.0;
    mat_a[5] = 6.0;

    double mat_u[2 * 3];
    double mat_s[3];
    double mat_v[3 * 3];
    math::matrix_svd(mat_a, 2, 3, mat_u, mat_s, mat_v, 1e-13);

    double groundtruth_mat_s[3];
    groundtruth_mat_s[0] = 9.525518091565104;
    groundtruth_mat_s[1] = 0.514300580658644;
    groundtruth_mat_s[2] = 0.0;

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(groundtruth_mat_s[i], mat_s[i], 1e-13);
}

TEST(MatrixSVDTest, TestLargeBeforeAfter)
{
    const int rows = 1000;
    const int cols = 50;
    double* mat_a = new double[rows * cols];
    for (int i = 0; i < rows * cols; ++i)
        mat_a[i] = static_cast<double>(i + 1);

    // Allocate results on heap to allow SVD of very big matrices.
    double* mat_u = new double[rows * cols];
    double* vec_s = new double[cols];
    double* mat_s = new double[cols * cols];
    double* mat_v = new double[cols * cols];

    math::matrix_svd(mat_a, rows, cols, mat_u, vec_s, mat_v, 1e-8);
    math::matrix_transpose(mat_v, cols, cols);
    std::fill(mat_s, mat_s + cols * cols, 0.0);
    for (int i = 0; i < cols; ++i)
        mat_s[i * cols + i] = vec_s[i];

    double* multiply_temp = new double[rows * cols];
    double* multiply_result = new double[rows * cols];
    math::matrix_multiply(mat_u, rows, cols, mat_s, cols,
        multiply_temp);
    math::matrix_multiply(multiply_temp, rows, cols, mat_v, cols, multiply_result);

    for (int i = 0; i < rows * cols; ++i)
        EXPECT_NEAR(mat_a[i], multiply_result[i], 1e-7);

    delete[] mat_a;
    delete[] mat_u;
    delete[] vec_s;
    delete[] mat_s;
    delete[] mat_v;
    delete[] multiply_temp;
    delete[] multiply_result;
}

TEST(MatrixSVDTest, BeforeAfter1)
{
    math::Matrix<double, 2, 2> A;
    A(0,0) = 1.0f; A(0,1) = 2.0f;
    A(1,0) = 3.0f; A(1,1) = 4.0f;

    math::Matrix<double, 2, 2> U;
    math::Matrix<double, 2, 2> S;
    math::Matrix<double, 2, 2> V;
    math::matrix_svd(A, &U, &S, &V, 1e-12);

    EXPECT_TRUE(A.is_similar(U * S * V.transposed(), 1e-12));
    EXPECT_GT(S(0,0), S(1,1));
}

TEST(MatrixSVDTest, BeforeAfter2)
{
    double values[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    math::Matrix<double, 3, 2> A(values);

    math::Matrix<double, 3, 2> U;
    math::Matrix<double, 2, 2> S;
    math::Matrix<double, 2, 2> V;
    math::matrix_svd(A, &U, &S, &V, 1e-12);

    EXPECT_TRUE(A.is_similar(U * S * V.transposed(), 1e-12));
    EXPECT_GT(S(0,0), S(1,1));
}

TEST(MatrixSVDTest, BeforeAfter3)
{
    double values[] = {1.0, 2.0, 3.0, 4.0 };
    math::Matrix<double, 2, 2> A(values);

    math::Matrix<double, 2, 2> U;
    math::Matrix<double, 2, 2> S(0.0);
    math::Matrix<double, 2, 2> V;
    math::matrix_svd<double>(*A, 2, 2, *U, *S, *V, 1e-12);
    std::swap(S(0,1), S(1,1)); // Swap the eigenvalues into place.

    EXPECT_TRUE(A.is_similar(U * S * V.transposed(), 1e-12));
    EXPECT_GT(S(0,0), S(1,1));
}

TEST(MatrixSVDTest, BeforeAfter4)
{
    double values[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    math::Matrix<double, 3, 2> A(values);

    math::Matrix<double, 3, 2> U;
    math::Matrix<double, 2, 2> S(0.0);
    math::Matrix<double, 2, 2> V;
    math::matrix_svd(*A, 3, 2, *U, *S, *V, 1e-12);
    std::swap(S(0,1), S(1,1)); // Swap the eigenvalues into place.

    EXPECT_TRUE(A.is_similar(U * S * V.transposed(), 1e-12));
    EXPECT_GT(S(0,0), S(1,1));
}

TEST(MatrixSVDTest, BeforeAfter5)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 1.0; A[1] = 0.0; A[2] = 1.0;
    A[3] = 1.0; A[4] = 0.0; A[5] = 1.0;
    A[6] = 1.0; A[7] = 0.0; A[8] = 1.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, BeforeAfter6)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 1.0; A[1] = 1.0; A[2] = 0.0;
    A[3] = 1.0; A[4] = 1.0; A[5] = 0.0;
    A[6] = 1.0; A[7] = 1.0; A[8] = 0.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, BeforeAfter7)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 0.0; A[1] = 1.0; A[2] = 1.0;
    A[3] = 0.0; A[4] = 1.0; A[5] = 1.0;
    A[6] = 0.0; A[7] = 1.0; A[8] = 1.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, BeforeAfter8)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 0.0; A[1] = 0.0; A[2] = 0.0;
    A[3] = 1.0; A[4] = 1.0; A[5] = 1.0;
    A[6] = 1.0; A[7] = 1.0; A[8] = 1.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, BeforeAfter9)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 1.0; A[1] = 1.0; A[2] = 1.0;
    A[3] = 0.0; A[4] = 0.0; A[5] = 0.0;
    A[6] = 1.0; A[7] = 1.0; A[8] = 1.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, BeforeAfter10)
{
    math::Matrix<double, 3, 3> A;
    A[0] = 1.0; A[1] = 1.0; A[2] = 1.0;
    A[3] = 1.0; A[4] = 1.0; A[5] = 1.0;
    A[6] = 0.0; A[7] = 0.0; A[8] = 0.0;

    math::Matrix<double, 3, 3> U, S, V;
    math::matrix_svd<double>(A, &U, &S, &V);
    math::Matrix<double, 3, 3> X = U * S * V.transposed();
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(A[i], X[i], 1e-6) << " at " << i;
}

TEST(MatrixSVDTest, SortedEigenvalues)
{
    double values[] = { 0.0697553, 0.949327, 0.525995, 0.0860558, 0.192214, 0.663227 };
    math::Matrix<double, 3, 2> U, mat(values);
    math::Matrix<double, 2, 2> S, V;
    math::matrix_svd<double, 3, 2>(mat, &U, &S, &V, 1e-12);
    EXPECT_GT(S(0,0), S(1,1));
}

TEST(MatrixSVDTest, TestPseudoInverse)
{
    math::Matrix<double, 3, 2> mat;
    mat[0] = 1.0;  mat[1] = 2.0;
    mat[2] = 3.0;  mat[3] = 4.0;
    mat[4] = 5.0;  mat[5] = 6.0;

    math::Matrix<double, 2, 3> pinv;
    math::matrix_pseudo_inverse(mat, &pinv, 1e-10);

    math::Matrix<double, 3, 2> mat2 = mat * pinv * mat;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(mat2[i], mat[i], 1e-13);

    math::Matrix<double, 2, 3> pinv2 = pinv * mat * pinv;
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(pinv2[i], pinv[i], 1e-13);
}

TEST(MatrixSVDTest, TestPseudoInverseGoldenData1)
{
    double a_values[] = { 2, -4,  5, 6, 0, 3, 2, -4, 5, 6, 0, 3 };
    double a_inv_values[] = { -2, 6, -2, 6, -5, 3, -5, 3, 4, 0, 4, 0 };
    math::Matrix<double, 4, 3> A(a_values);
    math::Matrix<double, 3, 4> Ainv(a_inv_values);
    Ainv /= 72.0;

    math::Matrix<double, 3, 4> result;
    math::matrix_pseudo_inverse(A, &result, 1e-10);

    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_NEAR(Ainv(r, c), result(r, c), 1e-16);
}

TEST(MatrixSVDTest, TestPseudoInverseGoldenData2)
{
    double a_values[] = { 1, 1, 1, 1, 5, 7, 7, 9 };
    double a_inv_values[] = { 2, -0.25, 0.25, 0, 0.25, 0, -1.5, 0.25 };
    math::Matrix<double, 2, 4> A(a_values);
    math::Matrix<double, 4, 2> Ainv(a_inv_values);

    math::Matrix<double, 4, 2> result;
    math::matrix_pseudo_inverse(A, &result, 1e-10);

    math::Matrix<double, 2, 4> U;
    math::Matrix<double, 4, 4> S;
    math::Matrix<double, 4, 4> V;

    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 2; ++c)
            EXPECT_NEAR(Ainv(r, c), result(r, c), 1e-13);
}
