#pragma once
#include "IWeightProvider.h"
namespace NN64 {
	class GoldenWeightProvider :
		public IWeightProvider
	{
	public:
		GoldenWeightProvider();
		~GoldenWeightProvider();

		// Inherited via IWeightProvider
		virtual double GetWeight() override;
		virtual void ProvideWeights(double *, int N) override;
	};
}

