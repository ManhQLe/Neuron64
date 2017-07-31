#include "stdafx.h"
#include "GoldenWeightProvider.h"


NN64::GoldenWeightProvider::GoldenWeightProvider()
{
}


NN64::GoldenWeightProvider::~GoldenWeightProvider()
{
}

double NN64::GoldenWeightProvider::GetWeight()
{
	return 1.618f;
}

void NN64::GoldenWeightProvider::ProvideWeights(double *X, int N)
{
	int i = N - 1;
	do {
		X[i] = 1.618f;
	} while (i--);
}
