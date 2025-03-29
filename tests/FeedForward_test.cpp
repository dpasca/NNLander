#define TEST_FIXITURE

#include "fixiture.hpp"

#define TEST_BODY                                                                                             \
    {                                                                                                         \
        net.FeedForward(params1.data(), inputs1, outputs1);                                                   \
        std::apply([&](const auto&... matrices) { ::FeedForward(inputs2, outputs2, matrices...); }, params2); \
        for (int i = 0; i < outputs2.rows(); i++)                                                             \
            EXPECT_FLOAT_EQ(outputs1[i], outputs2(i));                                                        \
    }

class FeedForwardTest10x3 : public FeedForwardTest<std::array<int, 2>{10, 3}> {};
TEST_F(FeedForwardTest10x3, workingTest) TEST_BODY

class FeedForwardTest10x12x12x3 : public FeedForwardTest<std::array<int, 4>{10, 12, 12, 3}> {};
TEST_F(FeedForwardTest10x12x12x3, workingTest) TEST_BODY

class FeedForwardTest10x20x30x20x10 : public FeedForwardTest<std::array<int, 5>{10, 20, 30, 20, 10}> {};
TEST_F(FeedForwardTest10x20x30x20x10, workingTest) TEST_BODY
