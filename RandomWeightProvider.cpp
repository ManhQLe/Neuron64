#include "stdafx.h"
#include "RandomWeightProvider.h"
#include <time.h>
NN64::RandomWeightProvider::RandomWeightProvider()
{
}

double NN64::RandomWeightProvider::GetWeight()
{
	srand(time(0));
	return (double)rand() / RAND_MAX;
}

void NN64::RandomWeightProvider::ProvideWeights(double *W, int N)
{
	srand(time(0));
	double x = 1.0f / RAND_MAX;
	int i = N-1;
	do {		
		W[i] = rand()*x;
	} while (i--);
}


NN64::RandomWeightProvider::~RandomWeightProvider()
{
}
