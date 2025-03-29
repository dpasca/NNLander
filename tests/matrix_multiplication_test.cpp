#include <Eigen/Dense>
#include <gtest/gtest.h>

TEST(matrix_multiplication, comma_init)
{
    Eigen::Matrix<float, 3, 4> mat;
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

    Eigen::Vector<float, 4> vec;
    vec << 1, 2, 3, 4;

    Eigen::Vector<float, 3> res = mat * vec;

    EXPECT_FLOAT_EQ(res(0), 30);
    EXPECT_FLOAT_EQ(res(1), 70);
    EXPECT_FLOAT_EQ(res(2), 110);
}

TEST(matrix_multiplication, index_fill)
{
    Eigen::Matrix<float, 3, 4> mat;
    mat(0, 0) = 1; mat(0, 1) =  2; mat(0, 2) =  3; mat(0, 3) =  4;
    mat(1, 0) = 5; mat(1, 1) =  6; mat(1, 2) =  7; mat(1, 3) =  8;
    mat(2, 0) = 9; mat(2, 1) = 10; mat(2, 2) = 11; mat(2, 3) = 12;

    Eigen::Vector<float, 4> vec;
    vec(0) = 1;
    vec(1) = 2;
    vec(2) = 3;
    vec(3) = 4;

    Eigen::Vector<float, 3> res = mat * vec;

    EXPECT_FLOAT_EQ(res(0), 30);
    EXPECT_FLOAT_EQ(res(1), 70);
    EXPECT_FLOAT_EQ(res(2), 110);
}
