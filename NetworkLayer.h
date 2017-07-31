#pragma once
#include "Matrix.h"
namespace NN64 {
	class NetworkLayer
	{
	public:
		Matrix W, Out,Sum;

	public:
		NetworkLayer();		
		~NetworkLayer();
	};
}