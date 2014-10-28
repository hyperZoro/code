#include "myMathsFunction.h"


std::vector<double> fnCubicHermesInterpolation(
	std::vector<double> TargetX,
	std::vector<double> X,
	std::vector<double> Y
	)
{
	//data validation
	if (X.size() != Y.size())
		throw 501;

	std::vector<double> m(X.size() - 1, 0.0);
	for (int i = 0; i < X.size(); ++i)
	{
		m[i] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i]);
	}

	return std::vector<double>(4,100.0);
}