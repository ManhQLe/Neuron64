#include "stdafx.h"
#include "Network.h"
#include <iostream>
NN64::Network::Network()
{
	ActFx = Sigmoid;
	DeriActFx = DeriSigmoid;
}
NN64::Network::Network(int InputCount, vector<int> LayerCount, IWeightProvider * WP = NULL)
{
	if (!WP) {
		WP = new RandomWeightProvider();
	}
	for (unsigned int i = 0;i < LayerCount.size();i++) {
		int NeuronCount = LayerCount[i];
		int MLen = NeuronCount * InputCount;

		auto n = new NetworkLayer();
		n->W = Matrix(NeuronCount, InputCount);
		WP->ProvideWeights(n->W.Vals, MLen);
		
		Layers.push_back(n);
		InputCount = NeuronCount;
	}
	ActFx = Sigmoid;
	DeriActFx = DeriSigmoid;
}

NN64::Network::~Network()
{
	for (unsigned int i = 0;i < Layers.size();i++) {
		delete Layers[i];
	}
}

NN64::Matrix NN64::Network::Eval(Matrix &In)
{			
	Matrix *I = &In;
	NetworkLayer *L;
	for (unsigned int i = 0;i < Layers.size();i++) {
		L = Layers[i];	
		L->Sum = (L->W**I);
		L->Out =  L->Sum^ ActFx;
		I = &L->Out;
	}
	//So that it does not delete the Inputs val ;	
	return Layers[Layers.size() - 1]->Out;
}

double NN64::Network::SqrError(Matrix &Exp)
{
	auto LastLayer = Layers[Layers.size() - 1];
	if (LastLayer->Out.Vals) {
		auto SqrErr = (Exp - LastLayer->Out) ^ 2;
		return SqrErr.SumElements();
	}
	return FLT_MAX;
}

void NN64::Network::Learn(Matrix &I, Matrix &A, double LR)
{
	int Size =(int) Layers.size();

	auto LastLayer = Layers[Size - 1];
	auto InputSize = Layers[0]->W.Row;
	auto OutputSize = Layers[Size]->W.Row;
		
	Matrix Out = Eval(I).T();

	Matrix Err = Matrix(1,A.Row,A.Vals)- Out; //Row Vector
	
	NetworkLayer *CL =NULL, *PL = NULL;
	Matrix *CI;
	Matrix K, dR;
	Matrix dO;
	int i = Size;
	do {		
		i--;
		CL = Layers[i];

		//Calculate K (It is Row Vector)
		//Err is Row Vector ^ derivative Column vector return RowVector
		
		dO = (CL->Out ^DeriActFx);
		K = (Err ^ dO);

		if (i)
			CI = &Layers[i - 1]->Out;
		else
			CI = &I;		
		
		//Propagate Error first so we dont lose weights;
		Err = K * CL->W;

		//Calculate Rate Of Change by tensor project
		dR = (K & *CI).T();
		//Calculate New Weight;
		CL->W = CL->W + (dR*LR);

	} while (i);
}

double NN64::Network::Sigmoid(double x)
{	
	return 1 / (1 + exp(-x));
}

double NN64::Network::DeriSigmoid(double x)
{
	return x*(1 - x);
}

string NN64::Network::ToString()
{
	using namespace std;
	stringstream ss;
	int S = (int)Layers.size();
	for (int i = 0;i < S;i++) {
		ss << "Layer: " << i << endl;
		ss << Layers[i]->W.ToString();
	}
	return ss.str();
}
