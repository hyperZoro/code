#include "myMathsFunction.h"

#include <cmath>

template<typename T>
unsigned long fnFindInterval(	T x,
								std::vector<T> v)
{
	for(unsigned long i; i < v.size(); ++i)
		if(x < v[i])
		{
			if(i==0)
				throw 501;
			else
				return i;
		}
		else if( (i == v.size()-1) and (x == v[i]) )
			return i;
}


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
	//monotonic X check
	for(std::vector<double>::iterator it = X.begin()+1; it < X.end() ; ++it)
		if(*it < *(it-1))
			throw 502;
	//range check
	for(std::vector<double>::iterator it = TargetX.begin(); it < TargetX.end(); ++it)
		if((*it < *X.begin()) || (*it > *X.end()))
			throw 503;

	unsigned long nLen = X.size();

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
	std::vector<double> c(X);

	double fnMonoCubicInterp(double x, double x0, double a, double b, double c, double d){return a*pow(x-x0,3)+b*pow(x-x0,2)+c*(x-x0)+d;}

	for(unsigned long i = 0; i < TargetX.size(); ++i)
	{
		double x = TargetX[i];
		unsigned long k = fnFindInterval(x, X);
		const double& xx = X[k];
		if (k == X.size() - 1)
			Result[i] = xx;
		else
		{
			if(calcFlag[k])

		}

	}



	return Result;
}