// PlayGround.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <string>

#include "myMathsFunction.h"


int main()
{
	std::vector<double> myVector;
	double xArrary[] = { 1.0, 2.0, 3.0, 4.0 };
	std::vector<double> X(xArrary, xArrary + sizeof(xArrary) / sizeof(double));
	double yArrary[] = { 10.0, 5.0, 3.0, 4.0 };
	std::vector<double> Y(yArrary, yArrary + sizeof(yArrary) / sizeof(double));

	int nNumInt = 1000;
	std::vector<double> xx(nNumInt, 0.0);
	xx[0] = X[0];
	for (int i = 1; i < nNumInt; ++i)
		xx[i] = (X[X.size() - 1] - X[0]) / nNumInt + xx[i - 1];


	




	try{
		myVector = fnCubicHermesInterpolation(xx, X, Y);
	}
	catch (int e)
	{
		std::cout << "error: " << e << "\n";
		return 0;
	}


	//display myVector
	std::cout << "myVector:\n";
	for (std::vector<double>::iterator it = myVector.begin(); it < myVector.end(); ++it)
	{
		std::cout << *it;
		if (it < myVector.end()-1)
			std::cout << ", ";
	}
	std::cout << "\n";
		

	
	char c;
	std::cin >> c;
	return c;
}

