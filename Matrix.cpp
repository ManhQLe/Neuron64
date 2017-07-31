#include "stdafx.h"
#include "Matrix.h"
//#include <iostream>
//#include <sstream>
using namespace std;


NN64::Matrix::Matrix():Row(0),Col(0),Vals(NULL)
{	
}

NN64::Matrix::Matrix(int row, int col):Col(col),Row(row)
{	
	Vals = new double[Row*Col];
}

NN64::Matrix::Matrix(double * vals, int row, int col) : Vals(vals),Row(row), Col(col)
{	
}

NN64::Matrix::Matrix(int row, int col, double * ValToCopy) : Matrix(row, col)
{
	CopyVals(ValToCopy);
}

NN64::Matrix::Matrix(const Matrix & M)
	: Matrix(M.Row, M.Col)
{
	memcpy(Vals, M.Vals, sizeof(double)*Row*Col);
}

NN64::Matrix& NN64::Matrix::operator=(const Matrix & M) {
	if (Row*Col != M.Row*M.Col) {
		delete[]Vals;
		Vals = new double[M.Row*M.Col];
	}	
	Row = M.Row;
	Col = M.Col;
	CopyVals(M.Vals);
	return *this;
}

#pragma region ELEMENTWISE OPS

NN64::Matrix NN64::Matrix::operator-(const Matrix & M)
{
	auto Len = Row * Col;
	if (Len != M.Col * M.Row)
		throw "Mismatched matrix";

	auto Chunks = ChunkThread(Len, MAXTHREAD);
	//Chunks
	auto NOT = Chunks.size();
	auto EV = new double[Len];
	auto *Threads = new thread[NOT];
	unsigned int k = 0;
	do
	{
		auto pair = Chunks[k];
		Threads[k++] = thread(NN64::Matrix::SubThread, EV, Vals, M.Vals, pair.first, pair.second);
	} while (k < NOT);

	for (unsigned int i = 0;i < NOT;i++)
		Threads[i].join();
	delete [] Threads;
	return Matrix(EV, Row, Col);
}

NN64::Matrix NN64::Matrix::operator+(const Matrix & M)
{
	auto Len = Row * Col;
	if (Len != M.Col * M.Row)
		throw "Mismatched matrix";

	auto Chunks = ChunkThread(Len, MAXTHREAD);
	//Chunks
	auto NOT = Chunks.size();
	auto EV = new double[Len];
	auto *Threads = new thread[NOT];
	unsigned int k = 0;
	do
	{
		auto pair = Chunks[k];
		Threads[k++] = thread(NN64::Matrix::AddThread, EV, Vals, M.Vals, pair.first, pair.second);
	} while (k < NOT);

	for (unsigned int i = 0;i < NOT;i++)
		Threads[i].join();
	delete [] Threads;
	return Matrix(EV, Row, Col);
}

NN64::Matrix NN64::Matrix::operator^(const Matrix & M)
{	
	auto Len = Row * Col;
	if (Len != M.Col * M.Row)
		throw "Mismatched matrix";

	auto Chunks = ChunkThread(Len, MAXTHREAD);
	//Chunks
	auto NOT = Chunks.size();
	auto EV = new double[Len];
	auto *Threads = new thread[NOT];
	unsigned int k = 0;	
	for (;k < NOT;k++)
	{
		auto pair = Chunks[k];	
		Threads[k] = thread(NN64::Matrix::DotThread, EV, Vals, M.Vals, pair.first, pair.second);
	}

	for (k = 0;k < NOT;k++)
		Threads[k].join();
	delete [] Threads;
	return Matrix(EV, Row, Col);
}


NN64::Matrix NN64::Matrix::operator^(const ModFX & fx)
{
	auto Len = Row * Col;
	auto Chunk = Len / MAXTHREAD;
	auto Remain = Len % MAXTHREAD;
	int NOT = Chunk > 0 ? MAXTHREAD : Remain;
	auto *Threads = new thread[NOT];
	int i = 0, k = 0;
	auto EV = new double[Len];
	do
	{
		int Start = i;
		i += Chunk + (Remain-- > 0 ? 1 : 0);
		int End = i;
		
		Threads[k++] = thread([&fx](double *EV, double *I, int s, int e) {
			int i = s;
			do {
				EV[i] = fx(I[i]);
			} while (++i < e);
		}, EV, Vals, Start, End);

	} while (i < Len);

	for (i = 0;i < NOT;i++)
		Threads[i].join();
	delete []Threads;
	return Matrix(EV, Row, Col);
}

NN64::Matrix NN64::Matrix::operator^(double x)
{
	auto fx = [&x](double e) {
		return pow(e, x);
	};

	return (*this) ^ fx;
}

NN64::Matrix NN64::Matrix::operator*(double scalar)
{
	auto fx = [&scalar](double e) {
		return e *scalar;
	};

	return (*this) ^ fx;
}

#pragma endregion

NN64::Matrix NN64::Matrix::operator*(const Matrix & M)
{
	if (Col != M.Row)
		throw "Mismatched Matrix";
	auto NRow = this->Row;
	auto NCol = M.Col;
	auto NLen = NRow*NCol;

	auto Chunk = NLen / MAXTHREAD;
	auto Remain = NLen % MAXTHREAD;
	int NOT = Chunk > 0 ? MAXTHREAD : Remain;
	auto *Threads = new thread[NOT];
	int i = 0, k = 0;
	auto EV = new double[NLen];
	do
	{
		int Start = i;
		i += Chunk + (Remain-- > 0 ? 1 : 0);
		int End = i;
		Threads[k++] = thread(NN64::Matrix::MultThread, EV, Vals, M.Vals, Col, NCol, Start, End);
		
	} while (i < NLen);
	for (i = 0;i < NOT;i++)
		Threads[i].join();

	delete[]Threads;

	return Matrix(EV, NRow, NCol);
}

NN64::Matrix NN64::Matrix::operator&(const Matrix & M) {
	auto NRow = Row * M.Row;
	auto NCol = Col * M.Col;
	auto Total = NRow*NCol;
	auto EV = new double[Total];

	auto Chunks = ChunkThread(Total, MAXTHREAD);
	int NOT = (int)Chunks.size();
	auto Threads = new thread[NOT];

	for (int i = 0;i < NOT;i++)
	{
		auto Chunk = Chunks[i];
		Threads[i] = thread(TensorThread, EV, Vals, M.Vals, Row, Col, M.Row, M.Col, Chunk.first, Chunk.second);
	}

	for (int i = 0;i < NOT;i++)
		Threads[i].join();

	delete[]Threads;

	return Matrix(EV, NRow, NCol);
}

NN64::Matrix NN64::Matrix::SumRows()
{
	double *EV = new double[Col];
	std::fill(EV, EV + Col, 0.0f);
	auto Chunks = ChunkThread(Row, MAXTHREAD);
	int NOT = (int)Chunks.size();
	auto Threads = new thread[NOT];
	
	auto SubSum = (double *)_aligned_malloc(sizeof(double)*NOT*Col, 32);
	std::fill(SubSum, SubSum + NOT*Col, 0.0f);
	int k;
	for (k = 0;k < NOT;k++) {
		auto pair = Chunks[k];
		auto Chunkth = k;
		Threads[k] = thread(SumRowsThread, Chunkth, SubSum, Vals, Col, pair.first, pair.second);
	}
	for (k = 0;k < NOT;k++)
		Threads[k].join();

	delete[]Threads;

	for (int i = 0;i < NOT;i++) {
		auto Sub = &SubSum[i*Col];		
		for (int j = 0;j < Col;j++) {
			EV[j] += Sub[j];
		}
	}

	_aligned_free(SubSum);

	return Matrix(EV, 1, Col);
}

NN64::Matrix NN64::Matrix::T()
{	
	auto Len = Row * Col;

	auto Chunks = ChunkThread(Len, MAXTHREAD);
	//Chunks
	auto NOT =(int) Chunks.size();
	auto EV = new double[Len];
	auto *Threads = new thread[NOT];
	int k = 0;

	for (k = 0;k < NOT;k++) {
		auto pair = Chunks[k];
		Threads[k] = thread(TransposeThread, EV, Vals, Row, Col, pair.first, pair.second);
	}
	for (k = 0;k < NOT;k++)
		Threads[k].join();
	delete[]Threads;

	return Matrix(EV, Col, Row);
}

double NN64::Matrix::SumElements()
{
	int i = Row * Col - 1;
	double Sum = 0;	
	do { Sum += Vals[i]; } while (i--);
	return Sum;
}

#pragma region THREAD FUNCTIONS

void NN64::Matrix::MultThread(double *EV, const double *I1, const double *I2, int C1, int C2, int s, int e) {
	int i = s, r, c;
	int BlockOf4 = C1 >> 2;
	int Remain = C1 & 3,R;
	__m256d VSum, V1, V2,Mul;
	double *P2VSum = (double *)&VSum;	
	double Sum;
	do {
		Sum = 0;
		VSum = _mm256_setzero_pd();
		r = (i / C2)*C1;
		c = i%C2;

		for (int j = 0;j < BlockOf4;j++) {

			V1 = _mm256_set_pd(I1[r], I1[r + 1], I1[r + 2], I1[r + 3]);
			r += 4;
			V2 = _mm256_set_pd(I2[c], I2[c + C2], I2[c + (C2 << 1)], I2[c + C2 * 3]);
			c += C2 << 2;
			Mul = _mm256_mul_pd(V1, V2);
			VSum = _mm256_add_pd(VSum, Mul);
		}
		R = Remain;
		while (R-->0) {
			Sum += I1[r++] * I2[c];			
			c += C2;
		}
		VSum = _mm256_hadd_pd(VSum, VSum);
		EV[i] = Sum + P2VSum[0] + P2VSum[2];
	} while (++i < e);
}

void NN64::Matrix::TransposeThread(double * EV, double * I, int R, int C, int s, int e)
{
	int i = s,r,c;
	do {
		r = i / C;
		c = i%C;
		EV[c*R + r] = I[i];
	} while (++i < e);

}

void NN64::Matrix::TensorThread(double * EV, double * V1, double * V2, int R1, int C1, int R2, int C2, int s, int e)
{
	int i = s;
	int C = C1*C2;
	int R = R1*R2;
	int r, c;
	do
	{
		r = i / C;
		c = i % C;		
		EV[i] = V1[r / R2*C1 + c / C2] * V2[(r%R2)*C2 + (c%C2)];
	} while (++i < e);
}

void NN64::Matrix::DotThread(double *EV, double *V1, double *V2, int Start, int End)
{
	int i = Start;
	do {
		EV[i] = V1[i] * V2[i];
	} while (++i < End);
}

void NN64::Matrix::AddThread(double *EV, double *V1, double *V2, int Start, int End)
{
	int i = Start;
	do {
		EV[i] = V1[i] + V2[i];
	} while (++i < End);
}

void NN64::Matrix::SubThread(double *EV, double *V1, double *V2, int Start, int End)
{
	int i = Start;
	do {
		EV[i] = V1[i] - V2[i];
	} while (++i < End);
}

void NN64::Matrix::SumRowsThread(int tidx,double * SubSum, double * Vals, int Col, int sr, int er)
{
	auto MySub = &SubSum[tidx*Col];
	int MultiOf4 = (Col >> 2)<<2;
	int Remain = Col & 3,R;
	__m256d V1;
	__m256d VSum;

	int i = sr;
	int j;
	do {		
		auto SofR = i*Col;
		for (j = 0;j < MultiOf4;j += 4) {
			V1 = _mm256_set_pd(Vals[SofR + 3], Vals[SofR + 2], Vals[SofR + 1], Vals[SofR]);
			VSum = _mm256_load_pd(&MySub[j]);
			VSum = _mm256_add_pd(VSum, V1);
			_mm256_store_pd(&MySub[j], VSum);

			SofR += 4;
		}
		R = Remain;
		while (R-->0)
		{
			MySub[j++] += Vals[SofR++];
		}
	} while (++i < er);
}
#pragma endregion

void NN64::Matrix::CopyVals(double * V)
{	
	memcpy(Vals, V, sizeof(double)*Row*Col);
}

string NN64::Matrix::ToString()
{
	using namespace std;
	stringstream s;
	auto Max = Row*Col;
	auto ColLess = Col - 1;
	for (auto i = 0;i < Max;i++) {
		s << Vals[i] << ((i%Col == ColLess) ? "\r\n" : " ");
	}
	return s.str();
}

vector<pair<int, int>> NN64::Matrix::ChunkThread(int NLen, int THREAD)
{
	auto Chunk = NLen / THREAD;
	auto Remain = NLen % THREAD;
	int i = 0;
	auto Vec = vector<pair<int, int>>();
	do
	{
		int Start = i;
		i += Chunk + (Remain-- > 0 ? 1 : 0);
		int End = i;
		Vec.push_back(make_pair(Start, End));

	} while (i < NLen);

	return Vec;
}

NN64::Matrix::~Matrix()
{
	delete Vals;
}

int NN64::Matrix::MAXTHREAD = 8;