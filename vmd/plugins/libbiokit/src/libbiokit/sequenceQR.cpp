/*
 * University of Illinois Open Source License
 * Copyright 2007 Luthey-Schulten Group, 
 * All rights reserved.
 * 
 * Developed by: Luthey-Schulten Group
 * 			     University of Illinois at Urbana-Champaign
 * 			     http://www.scs.illinois.edu/~schulten
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with 
 * the Software without restriction, including without limitation the rights to 
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to 
 * do so, subject to the following conditions:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimers.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimers in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): John Eargle, Elijah Roberts
 */

#include <stdio.h>
#include <math.h>
#include "symbol.h"
#include "sequence.h"
#include "ShortIntList.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"
#include "PIDTools.h"
#include "sequenceQR.h"

// Constructor
SequenceQR::SequenceQR(SequenceAlignment* align, int pCount, int pGapScaling, float gapScaleP, float normOrd) {

    alignment = align;
    preserveCount = pCount;
    performGapScaling = pGapScaling;
    gapScaleParameter = gapScaleP;
    normOrder = normOrd;
    cMi = alignment->getNumberPositions();
    cMk = alignment->getNumberSequences();
}
    
// Destructor
SequenceQR::~SequenceQR() {
   int i, j;
    //Free any used memory.
    for (i=0; i<cMi; i++) {
        for (j=0; j<cMj; j++) {
            delete matrix[i][j];
        }
        delete matrix[i];
    }
    delete matrix;
    delete columnList;
}

void SequenceQR::initializeMatrix(){    
   int k;
    //Fill in the initial sequence ordering list.    
    columnList = new int[cMk];
    for (k=0; k<cMk; k++) {
        columnList[k] = k;
    }
    
    //Scale the gap data.
    if (performGapScaling) {
        scaleGapData();
    }
}


// qrAlgorithm
//   Loop through the sequences, permuting the most linearly independent
//   sequence (n) to the front of the current submatrix, and perform
//   Householder transformations on the submatrices to zero out the
//   contributions of n
SequenceAlignment* SequenceQR::qrWithPIDCutoff(float identityCutoff) {
  
    //Perform the QR factorization.
    int k=0;
    for (k=0; k<cMk; k++) {
	    if(k >= cMi) {
		    //Permute the columns
		    if (k >= preserveCount)
			    permuteColumns(k);
		    //break;
	    } else {
		    //Permute the columns and perform the householder transform.
		    if (k >= preserveCount)
			    permuteColumns(k);
		    householder(k);
	    }	
        //If we have exceeded the percent identity cutoff value, we are done.
        if (k >= preserveCount && identityCutoff < 1.0 && isSequenceAboveIdentityCutoff(k, identityCutoff)) break;

    }

    //Copy the profile into a new sequence alignment.
    int kMax = k;
    SequenceAlignment* profile = new SequenceAlignment();
    for (k=0; k<kMax; k++) {
        profile->addSequence(alignment->getSequence(columnList[k]));
    }
    
    return profile;
}

// qrAlgorithm
//   Loop through the sequences, permuting the most linearly independent
//   sequence (n) to the front of the current submatrix, and perform
//   Householder transformations on the submatrices to zero out the
//   contributions of n
//
// Parameters: percent: the percent of sequences to get, out of the total number
//                      of sequences
//
SequenceAlignment* SequenceQR::qrWithPercentCutoff(int percent) {
    //Perform the QR factorization.
    int k=0;
    if(percent < 0) {
	    percent = 0;
    }
    if(percent > 100) {
	    percent = 100;
    }
    int limit = (int)((percent/100.0f)*cMk);
	
    for (k=0; k<limit; k++) {
	    if(k >= cMi) {
		    //	Permute the columns.
		    if (k >= preserveCount) 	
			    permuteColumns(k);
		    //break;	
	    } else {
		//	Permute the columns and perform the householder transform.
		if (k >= preserveCount) 	
			  permuteColumns(k);
		householder(k);
	    }
    }

    //Copy the profile into a new sequence alignment.
    //int kMax = k;
    SequenceAlignment* profile = new SequenceAlignment();
    for (k=0; k<limit; k++) {
        profile->addSequence(alignment->getSequence(columnList[k]));
    }
    
    return profile;
}


// householder
//
void SequenceQR::householder(int currentColumn) {
  int i,j,k;
  float sign, alpha, beta, gamma;
  float * hhVector;

  // Loop over coordinate dimensions (x,y,z,gap)
  for (j=0; j<cMj; j++) {
    
    // Compute Householder vector for current column
    k = currentColumn;
    alpha = 0;
    for (i=k; i<cMi; i++) {
      alpha += matrix[i][j][columnList[k]] * matrix[i][j][columnList[k]];
    }
    sign = (matrix[k][j][columnList[k]] >= 0) ? 1.0 : -1.0;
    alpha = -sign * sqrt(alpha);
    hhVector = new float[cMi];
    for (i=0; i<k; i++) {
      //hhVector[i] = -alpha;  // REMOVED 8/3
      hhVector[i] = 0;   // ADDED 8/3
    }
    hhVector[k] = matrix[k][j][columnList[k]] - alpha;
    for (i=k+1; i<cMi; i++) {
      //hhVector[i] = matrix[i][j][columnList[k]] - alpha;   // REMOVED 8/3
      // ADDED 8/3 {
      hhVector[i] = matrix[i][j][columnList[k]];
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
      for (; k<cMk; k++) {
	gamma = 0;
	for (i=0; i<cMi; i++) {
	  gamma += hhVector[i] * matrix[i][j][columnList[k]];
	}
	//printf("gamma: %f, (2*gamma)/beta: %f", gamma, (2*gamma)/beta);
	for (i=currentColumn; i<cMi; i++) {
	  //printf("((2*gamma)/beta) * hhVector[%d] = %f * %f = %f\n",i, (2*gamma)/beta, hhVector[i], ((2*gamma)/beta) * hhVector[i]);
	  matrix[i][j][columnList[k]] -= ((2*gamma)/beta) * hhVector[i];
	}
	//printf("\n");
      }
    }
  }
}


/**
 * This method moves the column with the max frobenius norm to the front of the current submatrix.
 *
 * @param   currentColumn   The column in the submatrix that should be filled with the max norm.
 */
void SequenceQR::permuteColumns(int currentColumn) {
   int k1, k2;    
    int maxCol = -1;
    
    //See if this is the first column. 
    if ((currentColumn == 0)) {

        // Skip this step for the binary version
	    if(cMj == 2) {
		    return;
	    }
        
        //Switch the first column with the column with least average percent identity.
        float min = -1.0;
        for (k1=0; k1<cMk; k1++) {
            float value = 0.0;
            for (k2=0; k2<cMk; k2++) {
                value += (float)PIDTools::getPercentIdentity(alignment->getSequence(k1), alignment->getSequence(k2));
            }
            
            //If this is the least percent identity we have yet encountered, save it.
            if (min < 0.0 || value < min) {
                min = value;
                maxCol = k1;
            }
        }
    }

    //Otherwise, use the frobenius norm to figure out which column to switch.
    else {
        float *norms = new float[cMk];
        float maxNorm = 0.0;

        int k=0;
        for (k=0; k<cMk; k++) {
            norms[k] = 0.0;
        }
        
        //Get the maxiumum norm.
        for (k=currentColumn; k<cMk; k++) {
            
            //Get frobenius norms for matrix.
            norms[k] = frobeniusNormByK(k, currentColumn);
            
            //If this is the largest norm, select this column.
            if (norms[k] > maxNorm) {
                maxCol = k;
                maxNorm = norms[k];
            }
        }
	delete norms;
    }


    // If we found a column to switch.    
    if (maxCol != -1) {
        //Switch the columns.
        int temp = columnList[maxCol];
        columnList[maxCol] = columnList[currentColumn];    
        columnList[currentColumn] = temp;
    }
}
    
/**
 * This method gets whether the current column exceeds the percent identity threshold with any
 * of the previous columns.
 *
 * @param   currentColumn   The column in the matrix that should be checked.
 * @return  1 if the column does exceed the percent identity threshold, 0 if it does not.
 */
int SequenceQR::isSequenceAboveIdentityCutoff(int currentColumn, float identityCutoff) {
   int k; 
    //See if this column exceeds the percent identity with any of the previously selected columns.
    AlignedSequence* currentSequence = alignment->getSequence(columnList[currentColumn]);
    for (k=0; k<currentColumn; k++) {
        AlignedSequence* sequence = alignment->getSequence(columnList[k]);
        if ((float)PIDTools::getPercentIdentity(currentSequence, sequence) >= identityCutoff) {
            return 1;
        }
    }
    
    return 0;
}


// frobeniusNormSeq
//   Get the frobenius norm for the matrix corresponding
//   to the data for one sequence
//   frobeniusNorm(A) = sqrt( sum( all Aij ) );
float SequenceQR::frobeniusNormByK(int k, int currentRow) {
    float fNorm = 0;
   int i, j; 
    for (i=currentRow; i<cMi; i++) {
        for (j=0; j<cMj; j++) {
            fNorm += pow(matrix[i][j][columnList[k]], normOrder);
        }
    }
    
    return pow(fNorm, 1/normOrder);
}


// frobeniusNormCoord
//
float SequenceQR::frobeniusNormByJ(int j) {
   int i, k;
    float fNorm = 0;

    for (i=0; i<cMi; i++) {
        for (k=0; k<cMk; k++) {
            fNorm += pow(matrix[i][j][columnList[k]], normOrder);
        }
    }
    
    return pow(fNorm, 1/normOrder);
}


// scaleGapData
//   Scale the gap matrix elements to appropriate values so that
//   the QR algorithm is not biased towards or against the gaps.
//   scale*fNorm(G) = fNorm(X) + fNorm(Y) + fNorm(Z)
void SequenceQR::scaleGapData() {
   int i, k;
    //Calculate the gap norm.
    float gapNorm = frobeniusNormByJ(cMj - 1);
    if (gapNorm != 0) {
        
        //Calculate the scaling value.
        float value = 0.0;
		int j=0;
        for (j=0; j<(cMj - 1); j++)
            value += frobeniusNormByJ(j);
        value /= gapNorm*(cMj -1);
        value *= gapScaleParameter;
        
        //Apply the scaling value to all of the gaps.
        j=cMj - 1;
        for (i=0; i<cMi; i++)
            for (k=0; k<cMk; k++)
                matrix[i][j][columnList[k]] *= value;
    }
}

