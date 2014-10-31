#include "myMathsFunction.h"
#include <algorithm>
#include <cmath>

template<typename T>
unsigned long fnFindInterval(	T x,
								std::vector<T> v)
{
	if ((x < v[0]) || (x > v[v.size()-1]))
		throw 501;

	for(unsigned long i=1; i < v.size(); ++i)
		if(x <= v[i])
				return i-1;
}

double _fnMonoCubicInterp(double x, double x0, double a, double b, double c, double d)
{return a*pow(x - x0, 3) + b*pow(x - x0, 2) + c*(x - x0) + d;}


std::vector<double> fnCubicHermesInterpolation(		std::vector<double> TargetX,
													std::vector<double> X,
													std::vector<double> Y,
													double ySlopeLeft,
													double ySlopeRight)
{
	///data validation

	//size check
	if (X.size() != Y.size())
		throw 501;

	unsigned long nLen = X.size();
	//monotonic X check
	for(std::vector<double>::iterator it = X.begin()+1; it < X.end() ; ++it)
		if(*it < *(it-1))
			throw 502;
	//range check
	for(std::vector<double>::iterator it = TargetX.begin(); it < TargetX.end(); ++it)
		if((*it < X[0]) || (*it > X[nLen-1]))
			throw 503;



	std::vector<double> m(nLen - 1, 0.0);
	std::vector<double> deltaX(nLen - 1, 0.0);
	std::vector<double> yPrime(nLen, 0.0);
	yPrime[0] = ySlopeLeft;
	yPrime[nLen-1] = ySlopeRight;

	for (unsigned int i = 0; i < m.size(); ++i)
	{
		m[i] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i]);
		deltaX[i] = X[i+1] - X[i];
		if(i>0)
		{
			yPrime[i] = (deltaX[i-1]*m[i]+deltaX[i]*m[i-1])/(X[i+1]-X[i-1]);
			if(yPrime[i] > 0)
				yPrime[i] = std::min( std::max(0.0, yPrime[i]) , 3*std::min(abs(m[i-1]), abs(m[i])) );
			else if(yPrime[i] < 0)
				yPrime[i] = std::max( std::min(0.0, yPrime[i]) , -3*std::min(abs(m[i-1]), abs(m[i])) );

		}
	}

	
	std::vector<double> Result(TargetX);
	std::vector<bool> calcFlag(nLen, false);
	std::vector<double> a(X);
	std::vector<double> b(X);
	std::vector<double>& c = yPrime;
	std::vector<double>& d = Y;



	for(unsigned long i = 0; i < TargetX.size(); ++i)
	{
		double x = TargetX[i];
		unsigned long k = fnFindInterval(x, X);
		const double& xx = X[k];
		if (k == X.size() - 1)
			Result[i] = xx;
		else
		{
			if (!calcFlag[k])
			{
				a[k] = 1.0 / (deltaX[k]*deltaX[k]) * (-2*m[k]+yPrime[k]+yPrime[k+1]);
				b[k] = 1.0 / deltaX[k] * (3 * m[k] - 2*yPrime[k] - yPrime[k + 1]);
				calcFlag[k] = true;
			}
			Result[i] = _fnMonoCubicInterp(x, xx, a[k], b[k], c[k], d[k]);
		}

	}



	return Result;
}