#pragma once
namespace NN64 {
	class IWeightProvider {
	public:		
		virtual double GetWeight() = 0;
		virtual void ProvideWeights(double *, int N) = 0;
	};
}