#pragma once
#include "NetworkLayer.h"
#include "RandomWeightProvider.h"
using namespace std;
namespace NN64 {
	class Network
	{
		typedef double(*ModFx)(double);			
	public:
		Network();
		vector<NetworkLayer*> Layers;
		ModFx ActFx;
		ModFx DeriActFx;

		Network(int Input, vector<int> Layers, IWeightProvider *WP);
		~Network();

		Matrix Eval(Matrix &I);
		
		double SqrError(Matrix &Exp);

		void Learn(Matrix &I, Matrix &A, double LR = 1.0);

		static double Sigmoid(double x);
		static double DeriSigmoid(double x);

		string ToString();

	};
}

