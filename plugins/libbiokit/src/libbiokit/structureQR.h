
#ifndef STRUCTUREQR_H
#define STRUCTUREQR_H

#include "structureAlignment.h"

#include <stdio.h>
#include <math.h>

// XXX - Eventually make child of superclass QR (along with Sequence QR)
class StructureQR{
  
 public:
  StructureQR(StructureAlignment *aln);
  StructureQR() {};
  ~StructureQR();
  int qr();
  int printColumns(FILE* outfile);
 private:
  int householder(int currentColumn);       // Householder transformation
  int permuteColumns(int currentColumn);       // column permutation
  float frobeniusNormSeq(int k, int currentRow);
  float frobeniusNormCoord(int j);
  int scaleGapData();
  
 private:
  /*
    Anytime you access the third index from coordMatrix, go through the
    columnList; So coordMatrix[i][j][k] becomes coordMatrix[i][j][columnList[k]];
    This level of indirection eliminates array copying and saves the sequence
    order.
  */
  float ***coordMatrix;
  int *columnList;         // indices for columns which may be permuted
  int cMi;                 // number of residue columns in coordMatrix (alignment column)
  int cMj;                 // number of coordinates for a coordMatrix residue (x,y,z,gap)
  int cMk;                 // number of sequences in coordMatrix
};

#endif
