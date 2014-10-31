#include <vector>

template<typename T>
unsigned long fnFindInterval(T x, std::vector<T> v);


std::vector<double> fnCubicHermesInterpolation(
												std::vector<double> TargetX,
												std::vector<double> X,
												std::vector<double> Y,
												double ySlopeLeft = 0.0,
												double ySlopeRight = 0.0
												);

