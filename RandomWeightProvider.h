#pragma once
#include "IWeightProvider.h"
namespace NN64 {
	class RandomWeightProvider :public IWeightProvider
	{
	public:
		RandomWeightProvider();
		double GetWeight();
		void ProvideWeights(double *W,int N);
		~RandomWeightProvider();
	};
}

