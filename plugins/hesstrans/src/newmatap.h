//$$ newmatap.h           definition file for matrix package applications

// Copyright (C) 1991,2,3,4,8: R B Davies

#ifndef NEWMATAP_LIB
#define NEWMATAP_LIB 0

#include "newmat.h"

#ifdef use_namespace
namespace NEWMAT {
#endif


// ************************** applications *****************************/


void QRZT(Matrix&, LowerTriangularMatrix&);

void QRZT(const Matrix&, Matrix&, Matrix&);

void QRZ(Matrix&, UpperTriangularMatrix&);

void QRZ(const Matrix&, Matrix&, Matrix&);

inline void HHDecompose(Matrix& X, LowerTriangularMatrix& L)
{ QRZT(X,L); }

inline void HHDecompose(const Matrix& X, Matrix& Y, Matrix& M)
{ QRZT(X, Y, M); }

ReturnMatrix Cholesky(const SymmetricMatrix&);

ReturnMatrix Cholesky(const SymmetricBandMatrix&);

void SVD(const Matrix&, DiagonalMatrix&, Matrix&, Matrix&,
    bool=true, bool=true);

void SVD(const Matrix&, DiagonalMatrix&);

inline void SVD(const Matrix& A, DiagonalMatrix& D, Matrix& U,
   bool withU = true) { SVD(A, D, U, U, withU, false); }

void SortSV(DiagonalMatrix& D, Matrix& U, bool ascending = false);

void SortSV(DiagonalMatrix& D, Matrix& U, Matrix& V, bool ascending = false);

void Jacobi(const SymmetricMatrix&, DiagonalMatrix&);

void Jacobi(const SymmetricMatrix&, DiagonalMatrix&, SymmetricMatrix&);

void Jacobi(const SymmetricMatrix&, DiagonalMatrix&, Matrix&);

void Jacobi(const SymmetricMatrix&, DiagonalMatrix&, SymmetricMatrix&,
   Matrix&, bool=true);

void EigenValues(const SymmetricMatrix&, DiagonalMatrix&);

void EigenValues(const SymmetricMatrix&, DiagonalMatrix&, SymmetricMatrix&);

void EigenValues(const SymmetricMatrix&, DiagonalMatrix&, Matrix&);

class SymmetricEigenAnalysis
// not implemented yet
{
public:
   SymmetricEigenAnalysis(const SymmetricMatrix&);
private:
   DiagonalMatrix diag;
   DiagonalMatrix offdiag;
   SymmetricMatrix backtransform;
   FREE_CHECK(SymmetricEigenAnalysis)
};

void SortAscending(GeneralMatrix&);

void SortDescending(GeneralMatrix&);


// class for deciding which fft to use and containing new fft function
class FFT_Controller
{
public:
   static bool OnlyOldFFT;
   static bool ar_1d_ft (int PTS, Real* X, Real *Y);
   static bool CanFactor(int PTS);
};

void FFT(const ColumnVector&, const ColumnVector&,
   ColumnVector&, ColumnVector&);

void FFTI(const ColumnVector&, const ColumnVector&,
   ColumnVector&, ColumnVector&);

void RealFFT(const ColumnVector&, ColumnVector&, ColumnVector&);

void RealFFTI(const ColumnVector&, const ColumnVector&, ColumnVector&);

void DCT_II(const ColumnVector&, ColumnVector&);

void DCT_II_inverse(const ColumnVector&, ColumnVector&);

void DST_II(const ColumnVector&, ColumnVector&);

void DST_II_inverse(const ColumnVector&, ColumnVector&);

void DCT(const ColumnVector&, ColumnVector&);

void DCT_inverse(const ColumnVector&, ColumnVector&);

void DST(const ColumnVector&, ColumnVector&);

void DST_inverse(const ColumnVector&, ColumnVector&);

// This class is used by the new FFT program

// Suppose an integer is expressed as a sequence of digits with each
// digit having a different radix.
// This class supposes we are counting with this multi-radix number
// but also keeps track of the number with the digits (and radices)
// reversed.
// The integer starts at zero
// operator++() increases it by 1
// Counter gives the number of increments
// Reverse() gives the value with the digits in reverse order
// Swap is true if reverse is less than counter
// Finish is true when we have done a complete cycle and are back at zero

class MultiRadixCounter
{
   const SimpleIntArray& Radix;
                              // radix of each digit
                              // n-1 highest order, 0 lowest order
   SimpleIntArray& Value;     // value of each digit
   const int n;               // number of digits
   int reverse;               // value when order of digits is reversed
   int product;               // product of radices
   int counter;               // counter
   bool finish;               // true when we have gone over whole range
public:
   MultiRadixCounter(int nx, const SimpleIntArray& rx,
      SimpleIntArray& vx);
   void operator++();         // increment the multi-radix counter
   bool Swap() const { return reverse < counter; }
   bool Finish() const { return finish; }
   int Reverse() const { return reverse; }
   int Counter() const { return counter; }
};


#ifdef use_namespace
}
#endif



#endif

// body file: cholesky.cpp
// body file: evalue.cpp
// body file: fft.cpp
// body file: hholder.cpp
// body file: jacobi.cpp
// body file: newfft.cpp
// body file: sort.cpp
// body file: svd.cpp





