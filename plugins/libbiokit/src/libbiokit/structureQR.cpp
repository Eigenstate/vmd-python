
#include "structureQR.h"

// Constructor
StructureQR::StructureQR(StructureAlignment *aln) {

  //printf("=>StructureQR\n");

  //int tempIndex = 0;

  cMi = aln->getNumberPositions();
  cMj = 4;
  cMk = aln->getNumberStructures();

  coordMatrix = new float**[cMi];
  int i,j,k;
  for (i=0; i<cMi; i++) {
    coordMatrix[i] = new float*[cMj];
    for (j=0; j<cMj; j++) {
      coordMatrix[i][j] = new float[cMk];
      for (k=0; k<cMk; k++) {
	//printf("    [%d][%d][%d]\n",i,j,k);
	AlignedStructure* struct1 = aln->getStructure(k);
	if (struct1 == NULL) {
	  printf("Error: StructureQR constructor - structureAlignment is bad\n");
	}
	// if (filling in gap column)
	if (j==3) {
	  if ( struct1->getAlphabet()->isGap(struct1->get(i)) ) {
	    coordMatrix[i][j][k] = 1.0;   // THIS SHOULD BE NORMED
	  }
	  else {
	    coordMatrix[i][j][k] = 0.0;
	  }
	}
	// else if (filling in coord column, but residue is a gap)
	else if ( struct1->getAlphabet()->isGap(struct1->get(i)) ) {
	  //printf("     isGap?\n");	  
	  coordMatrix[i][j][k] = 0.0;
	}
	else {
	  //printf("     here1\n");
	  //printf("     here2\n");
	  //coordMatrix[i][j][k] = aln->sequences[k].structure.caCoordinates[tempIndex][j];
	  switch (j) {
	  case 0:
	    coordMatrix[i][j][k] = struct1->getCoordinate(i).getX();
	    break;
	  case 1:
	    coordMatrix[i][j][k] = struct1->getCoordinate(i).getY();
	    break;
	  case 2:
	    coordMatrix[i][j][k] = struct1->getCoordinate(i).getZ();
	    break;
	  default:
	    printf("Error: StructureQR constructor\n");
	    //printf("WTF, mate?\n");
	  }
	  //printf("  coordMatrix[%d][%d][%d] = %f\n",i,j,k,coordMatrix[i][j][k]);
	  //printf("     here3\n");
	}
      }
    }
  }

  columnList = new int[cMk];
  for (k=0; k<cMk; k++) {
    columnList[k] = k;
  }

  //printf("<=StructureQR\n");

  return;
}


// Destructor
StructureQR::~StructureQR() {

  //printf("=>StructureQR::~StructureQR()\n");
  //printf("cMi = %d, cMj = %d, cMk = %d\n",cMi,cMj,cMk);

  
  int i, j;
/*
  for (i=0; i<cMi; i++) {
    for (j=0; j<cMj; j++) {
      for (k=0; k<cMk; k++) {
	//printf("  coordMatrix[%d][%d][%d] = %f\n",i,j,k,coordMatrix[i][j][k]);
      }
    }
  }
*/
  //printf("   delete coordMatrix[8][3];\n");
  //delete coordMatrix[8][3];

  for (i=cMi-1; i>=0; i--) {
    //printf("   Hey1\n");
    for (j=cMj-1; j>=0; j--) {
      /*
      for (k=0; k<cMk; k++) {
	printf("  coordMatrix[%d][%d][%d] = %f\n",i,j,k,coordMatrix[i][j][k]);
      }
      */
      //printf("   delete coordMatrix[%d][%d]; (%d)\n",i,j,coordMatrix[i][j]);
      delete coordMatrix[i][j];
    }
    //printf("   delete coordMatrix[%d];\n",i);
    delete coordMatrix[i];
  }
  //printf("   delete coordMatrix;\n");
  delete coordMatrix;
  //printf("   Hey3\n");
  delete columnList;
  //printf("<=StructureQR::~StructureQR()\n");

  return;
}


// qrAlgorithm
//   Loop through the sequences, permuting the most linearly independent
//   sequence (n) to the front of the current submatrix, and perform
//   Householder transformations on the submatrices to zero out the
//   contributions of n
int StructureQR::qr() {

  int k;  // current column (corresponds to sequence)

  scaleGapData();

  for (k=0; k<cMk; k++) {
    if (k >= cMi) {
      permuteColumns(k);
    } else {
      permuteColumns(k);
      householder(k);
    }
  }

  return 1;
}


// printColumns
int StructureQR::printColumns(FILE* outfile) {
  int k;
  //printf(">printColumns\n");
  for (k=0; k<cMk; k++) {
    fprintf(outfile, "%d ", columnList[k]);
    //printf("%d ", columnList[k]);
  }
  fprintf(outfile,"\n");
  //printf("\n");

  //printf("<printColumns\n");
  return 1;
}


// householder
//
int StructureQR::householder(int currentColumn) {

  int i,j,k;
  float sign, alpha, beta, gamma;
  float * hhVector;

  // Loop over coordinate dimensions (x,y,z,gap)
  for (j=0; j<cMj; j++) {
    
    // Compute Householder vector for current column
    k = currentColumn;
    alpha = 0;
    for (i=k; i<cMi; i++) {
      alpha += coordMatrix[i][j][columnList[k]] * coordMatrix[i][j][columnList[k]];
    }
    sign = (coordMatrix[k][j][columnList[k]] >= 0) ? 1.0 : -1.0;
    alpha = -sign * sqrt(alpha);
    hhVector = new float[cMi];
    for (i=0; i<k; i++) {
      //hhVector[i] = -alpha;  // REMOVED 8/3
      hhVector[i] = 0;   // ADDED 8/3
    }
    hhVector[k] = coordMatrix[k][j][columnList[k]] - alpha;
    for (i=k+1; i<cMi; i++) {
      //hhVector[i] = coordMatrix[i][j][columnList[k]] - alpha;   // REMOVED 8/3
      // ADDED 8/3 {
      hhVector[i] = coordMatrix[i][j][columnList[k]];
      //if (i==k) {
      //  hhVector[i] -= alpha;
      //}
      // } ADDED 8/3
    }

    // Get inner product of Householder vector with itself
    beta = 0;
    for (i=k; i<cMi; i++) {
      beta += hhVector[i] * hhVector[i];
    }
    
    // Apply transformation to remaining submatrix
    if (beta != 0) {
      //printf("In --- beta: %f\n", beta);
      for (; k<cMk; k++) {
	gamma = 0;
	for (i=0; i<cMi; i++) {
	  gamma += hhVector[i] * coordMatrix[i][j][columnList[k]];
	}
	//printf("gamma: %f, (2*gamma)/beta: %f", gamma, (2*gamma)/beta);
	for (i=currentColumn; i<cMi; i++) {
	  //printf("coordMatrix[%d][%d][%d]: %f\n", i,j,columnList[k], coordMatrix[i][j][columnList[k]]);
	  //printf("((2*gamma)/beta) * hhVector[%d] = %f * %f = %f\n",i, (2*gamma)/beta, hhVector[i], ((2*gamma)/beta) * hhVector[i]);
	  coordMatrix[i][j][columnList[k]] -= ((2*gamma)/beta) * hhVector[i];
	}
	//printf("\n");
      }
    }
  }

  return 0;
}


// permutation - 
//   move the column with the max frobenius norm to the front
//   of the current submatrix (currentColumn)
int StructureQR::permuteColumns(int currentColumn) {
  
  int frontCol = currentColumn;
  int maxCol = 0;
  float *norms = new float[cMk];
  float maxNorm = 0.0;

  int k=0;
  for (k=0; k<cMk; k++) {
    norms[k] = 0.0;
  }

  // Get frobenius norms for remaining matrices
  //for (k=frontCol; k<cMk; k++) {
  for (k=0; k<cMk; k++) {
    norms[k] = frobeniusNormSeq(k, frontCol);
    if (norms[k] > maxNorm) {
      maxCol = k;
      maxNorm = norms[k];
    }
  }

  delete norms;

  //printf("frontCol: %d\n",frontCol);
  //printf("maxCol: %d\n",maxCol);

  int tempMaxCol = columnList[maxCol];
  int tempFrontCol = columnList[frontCol];

  //printf(" tempFrontCol: %d\n",tempFrontCol);
  //printf(" tempMaxCol: %d\n",tempMaxCol);

  columnList[frontCol] = tempMaxCol;
  columnList[maxCol] = tempFrontCol;

  //printColumns();

  return 0;
}


// frobeniusNormSeq
//   Get the frobenius norm for the matrix corresponding
//   to the data for one sequence
//   frobeniusNorm(A) = sqrt( sum( all Aij ) );
float StructureQR::frobeniusNormSeq(int k, int currentRow) {

  float fNorm = 0;
   int i,j;
  for (i=currentRow; i<cMi; i++) {
    for (j=0; j<cMj; j++) {
      //fNorm += pow(abs(coordMatrix[i][j][k]),2);
      fNorm += coordMatrix[i][j][columnList[k]] * coordMatrix[i][j][columnList[k]];
    }
  }

  //printf("%d,%d: %f\n",k,currentRow,sqrt(fNorm));

  return sqrt(fNorm);
}


// frobeniusNormCoord
//
float StructureQR::frobeniusNormCoord(int j) {

  float fNorm = 0;
   int i, k;
  for (i=0; i<cMi; i++) {
    for (k=0; k<cMk; k++) {
      //fNorm += pow(abs(coordMatrix[i][j][k]),2);
      fNorm += coordMatrix[i][j][columnList[k]] * coordMatrix[i][j][columnList[k]];
    }
  }

  return sqrt(fNorm);
}


// scaleGapData
//   Scale the gap matrix elements to appropriate values so that
//   the QR algorithm is not biased towards or against the gaps.
//   scale*fNorm(G) = fNorm(X) + fNorm(Y) + fNorm(Z)
int StructureQR::scaleGapData() {

  //float coordNorm;
  int i, k;
  float scale = 1.0;   // Default for the case where gapNorm==0
  //float scaleConstant = 1.19;   // REMOVED 8/3
  float scaleConstant = 2.0;   // ADDED 8/3
  float gapNorm = frobeniusNormCoord(3);

  if (gapNorm != 0) {
    scale = frobeniusNormCoord(0) + frobeniusNormCoord(1) + frobeniusNormCoord(2);
    scale /= gapNorm;
    scale *= scaleConstant;
    int j=3;
    for (i=0; i<cMi; i++) {
      for (k=0; k<cMk; k++) {
	coordMatrix[i][j][columnList[k]] *= scale;
      }
    }
  }

  return 0;
}
