#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Eigen/Dense>
#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

template<typename T, typename Scalar>
concept EigenMatrix = requires
{
    requires std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;
    requires std::is_same_v<typename T::Scalar, Scalar>;
};

template<typename T, typename Scalar, int Cols>
concept EigenMatrixC = requires
{
    requires EigenMatrix<T, Scalar>;
    requires std::decay_t<T>::ColsAtCompileTime == Cols;
};

template<typename T>
concept NetArch = requires(T t)
{
    { t.size() } -> std::convertible_to<size_t>;
    { t.front() } -> std::convertible_to<int>;
    { t.back() } -> std::convertible_to<int>;
    { t[0] } -> std::convertible_to<int>;
};

template<typename T, typename Y>
concept GetValFunc = requires(T t)
{
    { t(0, 0, 0) } -> std::convertible_to<Y>;
};

template<typename T, NetArch auto netArch>
using NetParam = decltype([]<size_t... Is>(std::index_sequence<Is...>) {
    return std::tuple<Eigen::Matrix<T, netArch[Is+1], netArch[Is]+1>...>{};
}(std::make_index_sequence<netArch.size()-1>{}));

template<typename T, NetArch auto netArch>
void fillNetParams(const GetValFunc<T> auto& func, NetParam<T, netArch>& netParam)
{
    auto fillLayer = [&]<size_t Idx>() {
        EigenMatrix<T> auto& layer = std::get<Idx>(netParam);
        for (int r = 0; r < std::remove_cvref_t<decltype(layer)>::RowsAtCompileTime; r++) {
            for (int c = 0; c < std::remove_cvref_t<decltype(layer)>::ColsAtCompileTime; c++) {
                layer(r, c) = func(Idx, r, c);
            }
        }    
    };

    [&]<size_t... Idxs>(std::index_sequence<Idxs...>) {
        (fillLayer.template operator()<Idxs>(), ...);
    }(std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<decltype(netParam)>>>{});
}

template<std::floating_point T>
T Activate(T x)
{
    return x > T(0.0) ? x : T(0.0); 
}

template<std::floating_point T, int I, int O>
void FeedForward(const Eigen::Vector<T, I>& pInputs, Eigen::Vector<T, O>& pOutputs, const Eigen::Matrix<T, O, I+1>& pParameters)
{
    Eigen::Vector<T, I + 1> extendedInputs;
    extendedInputs.template head<I>() = pInputs;
    extendedInputs(I) = T(1.0);

    pOutputs = pParameters * extendedInputs;
    pOutputs = pOutputs.unaryExpr([](T x) { return Activate(x); });
}

template<std::floating_point T, int I, int O>
void FeedForward(const Eigen::Vector<T, I>& pInputs, Eigen::Vector<T, O>& pOutputs, const EigenMatrixC<T, I+1> auto& pParameters,
                 const EigenMatrix<T> auto&  pRemaingParams, const EigenMatrix<T> auto& ... pRemaingParamsPack)
{
    Eigen::Vector<T, I + 1> extendedInputs;
    extendedInputs.template head<I>() = pInputs;
    extendedInputs(I) = T(1.0);

    Eigen::Vector<T, std::remove_cvref_t<decltype(pParameters)>::RowsAtCompileTime> outputs = pParameters * extendedInputs;
    outputs = outputs.unaryExpr([](T x) { return Activate(x); });

    FeedForward(outputs, pOutputs, pRemaingParams, pRemaingParamsPack...);
}

template<typename T, NetArch auto netArch>
void FeedForward(const Eigen::Vector<T, netArch.begin()>& pInputs, Eigen::Vector<T, netArch.end()>& pOutputs, const NetParam<T, netArch>& pParameters)
{
    std::apply([&](const auto&... params) { ::FeedForward(pInputs, pOutputs, params...); }, pParameters);
}

#endif 
