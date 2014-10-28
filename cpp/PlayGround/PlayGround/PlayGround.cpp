// PlayGround.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>

#include "myMathsFunction.h"


int main()
{
	std::vector<double> myVector;
	try{
		myVector = fnCubicHermesInterpolation(std::vector<double>(1, 0.0), std::vector<double>(2, 10.0), std::vector<double>(3, 100.0));
	}
	catch (int e)
	{
		std::cout << e << "\n";
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

