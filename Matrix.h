#pragma once
#include "stdafx.h"
#include <string>
#include <immintrin.h>
namespace NN64 {
	class Matrix
	{
		friend class NetworkLayer;
		friend class Network;			
	public:		
		double *Vals;
		//typedef double(*ModFX)(double v);
		typedef std::function<double(double)> ModFX;
		int Row;
		int Col;

		Matrix();
		Matrix(int row, int col);
		Matrix(double *vals,int row, int col);
		Matrix(int row, int col, double *ValToCopy);
		Matrix(const Matrix&M);
		Matrix& operator=(const Matrix &M);

		Matrix operator+(const Matrix&M);
		Matrix operator-(const Matrix&M);
		Matrix operator^(const Matrix&M);
		Matrix operator^(double x);
		Matrix operator^(const ModFX&fx);

		Matrix operator*(const Matrix&M);
		Matrix operator*(double Scalar);		

		//Tensor Product
		Matrix operator&(const Matrix&M);

		Matrix SumRows();		
		Matrix T();
		
		double SumElements();


		void CopyVals(double *V);
		std::string ToString();
		~Matrix();

		static int MAXTHREAD;
		static std::vector<std::pair<int,int>> ChunkThread(int, int);
				
		static void TransposeThread(double *EV, double *I, int R, int C,int s,int e);
		static void TensorThread(double *EV, double *V1, double *V2, int R1, int C1, int R2, int C2, int s, int e);
		static void DotThread(double *EV, double *V1, double *V2, int Start, int End);
		static void AddThread(double *EV, double *V1, double *V2, int Start, int End);
		static void SubThread(double *EV, double *V1, double *V2, int Start, int End);
		static void MultThread(double *EV, const double *I1, const double *I2, int C1, int C2, int s, int e);		
		static void SumRowsThread(int tidx,double *EV, double*Vals,int Col, int s, int e);

	};
}

