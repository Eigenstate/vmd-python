/* common definitions for the hesstrans plugin */
#ifndef _HESSTRANS_PLUGIN_H
#define _HESSTRANS_PLUGIN_H


#define WANT_STREAM                  // include.h will get stream fns
#define WANT_MATH                    // include.h will get math fns

#include "newmatap.h"                // need matrix applications

#define DEBUG

#ifdef DEBUG
#include "newmatio.h"                // need matrix output routines
#endif


#ifdef use_namespace
using namespace NEWMAT;              // access NEWMAT namespace
#endif

struct bondCoord{
  int x1;
  int x2;
  //Real val;
  bondCoord(int a1, int a2):
    x1(a1), x2(a2) {}
  bondCoord() {}
};
typedef struct bondCoord bondCoord;


struct angleCoord{
  int x1;
  int x2;
  int x3;
  //Real val;
  angleCoord(int a1, int a2, int a3):
    x1(a1), x2(a2), x3(a3) {}
  angleCoord() {}
};


struct dihedralCoord{
  int x1;
  int x2;
  int x3;
  int x4;
  //Real val;
  dihedralCoord(int a1, int a2, int a3, int a4):
    x1(a1), x2(a2), x3(a3), x4(a4) {}

  dihedralCoord() {}
};


struct improperCoord{
  int x1;
  int x2;
  int x3;
  int x4;
  //Real val;
  improperCoord(int a1, int a2, int a3, int a4):
    x1(a1), x2(a2), x3(a3), x4(a4) {}
  improperCoord() {}
};



void getGeneralizedInverse(Matrix& G, Matrix& Gi);

void getBMatrix(Real** cartCoords, int numCartesians,
		bondCoord** bonds, int numBonds,
		angleCoord** angles, int numAngles,
		dihedralCoord** dihedrals, int numDihedrals,
		improperCoord** impropers, int numImpropers,
		Matrix& B);

extern int getInternalHessian(double** cartCoords, double* carthessian,
		       int* bondlist, int* anglelist, int* dihedlist, int* imprplist,
		       int numCartesians, int numBonds, int numAngles, int numDihedrals,
		       int numImpropers, double* hessianInternal);

extern int getNormalModes(double** cartCoords, double* carthessian,
			  double* masslist, int numCartesians, double* frequencies,
			  double* normalmodes, int &nfreq, int decontaminate);

#endif
