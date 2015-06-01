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
 * Author(s): Patrick O'Donoghue, Michael Januszyk, John Eargle, Elijah Roberts
 */

#include <stdlib.h>
#include <math.h>
#include "alphabet.h"
//#include "Atom.h"
#include "Residue.h"
#include "structure.h"
#include "Contact.h"
#include "ContactList.h"
#include "structureAlignment.h"
#include "qTools.h"


#define MIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)  ((X) > (Y) ? (X) : (Y))


// Constructor
QTools::QTools(StructureAlignment* sa) 
  : alignment(sa), qScores(0), qPerRes(0), qPower(0.15) {

  return;
}


// Destructor
QTools::~QTools() {
   int i;
  int structCount = alignment->getNumberStructures();

  if (qScores != 0) {
    for (i=structCount-1; i>=0; i--) {
      if (qScores[i] != 0) {
	delete qScores[i];
      }
    }
    
    delete qScores;
  }

  if (qPerRes != 0) {
    for (i=structCount-1; i>=0; i--) {
      if (qPerRes[i] != 0) {
	delete qPerRes[i];
      }
    }
    
    delete qPerRes;
  }

  return;
}


// q
//   Calculates a q-value which takes into account the gapped regions
//   Originally written by Patrick O'Donoghue
//   Translated from fortran to C by Michael Januszyk
//   Completely rewritten in C++ by John Eargle
int QTools::q(int ends, int excludeAln /*=0*/, int excludeGap /*=0*/) {

  //printf("=>QTools::q\n");
// 
  int structCount = alignment->getNumberStructures();
   int i, j;
  if (qScores == 0) {
    qScores = new float* [structCount];
    for (i=0; i<structCount; i++) {
      qScores[i] = new float[structCount];
    }  
  }

  // Don't need to initialize qScores with 0s
  
  float** qAln  = new float* [structCount];
  float** qGap  = new float* [structCount];
  int**   qNorm = new int*   [structCount];

  for (i=0; i<structCount; i++) {
    qAln[i]  = new float[structCount];
    qGap[i]  = new float[structCount];
    qNorm[i] = new int[structCount];
    for (j=0; j<structCount; j++) {
      qAln[i][j] = 0.0;
      qGap[i][j] = 0.0;
      qNorm[i][j] = 0;
    }
  }

  if (!excludeAln) {
      getQAln(qAln,qNorm);
  } else {
      printf("Excluding aln\n");
  }
  if (!excludeGap) {
      getQGap(qGap,qNorm,ends);
  } else {
      printf("Excluding gap\n");
  }
  
  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++) {
      if (qNorm[i][j] > 0) {
          qScores[i][j] = (qAln[i][j] + qGap[i][j]) / qNorm[i][j];
      } else {
          qScores[i][j] = 0;
      }
      if (i==j) {
          qScores[i][j] = 1.0;
      }
    }
  }

  // XXX - DEBUGGING CODE
  /*
    FILE *out = fopen("q.txt","w");
    fprintf(out,"qAln:\n");
    printMatrix(out,qAln);
    fprintf(out,"\n");
    fprintf(out,"qGap:\n");
    printMatrix(out,qGap);
    fprintf(out,"\n");
    fprintf(out,"qNorm:\n");
    printMatrix(out,qdNorm);
    fprintf(out,"\n");
    fprintf(out,"q:\n");
    printQ(out);
    fclose(out);
  */
  // \XXX

  //printf("<=QTools::q\n");
  return 1;
}


// qPerResidue
//   
int QTools::qPerResidue() {

  //printf("=>QTools::qPerResidue\n");

  int len = alignment->getNumberPositions();
  int structCount = alignment->getNumberStructures();
  //int norm = 0;

  float numerator = 0.0;
  float denominator1 = 0.0;
  float denominator2 = 0.0;
  float score1 = 0.0;
  float score2 = 0.0;
  float* distances = new float[structCount];
   int i, j, row, struct1, struct2, col;
  if (qPerRes == 0) {
    qPerRes = new float* [structCount];
    for (i=0; i<structCount; i++) {
      //qPerRes[i] = new float[alignment->getStructure(i)->getLength()];
      qPerRes[i] = new float[len];
    }  
  }

  int **qNorm = new int* [structCount];
  for (i=0; i<structCount; i++) {
    //qNorm[i] = new int[alignment->getStructure(i)->getLength()];
    qNorm[i] = new int[len];
    for (j=0; j<len; j++) {
      qPerRes[i][j] = 0.0;
      qNorm[i][j] = 0;
    }
  }

    // Traverse columns i and j summing up the contributions to
    //   qAln from differences between interior backbone distances;
    //   ignore adjacent columns
  for (i=0; i<len-2; i++) {
    //printf("i: %d\n",i);
    for (j=i+2; j<len; j++) {
      //printf("j: %d\n",j);
      // Get backbone distances for current pair of columns
      getBackboneDistances(distances, i, j);
      
      // Unset distances for structures if the columns correspond
      //   to adjacent residues
      for (row=0; row<structCount; row++) {
	//printf("distances[%d] = %f\n",row,distances[row]);
	if (alignment->getStructure(row)->alignedToUnalignedIndex(i) ==
	    alignment->getStructure(row)->alignedToUnalignedIndex(j)-1) {
	  distances[row] = -1;
	}
      }
      
      // Calculate scores for current pair of columns
      // Traverse rows struct1 and struct2 calculating qPerRes
      //   scores for each set of rows and columns
      for (struct1=0; struct1<structCount-1; struct1++) {
	for (struct2=struct1+1; struct2<structCount; struct2++) {
	  // if (neither structure is gapped) add new score to qAln
	  if ( distances[struct1] >= 0.0 && distances[struct2] >= 0.0 ) {
	    numerator = -1 * pow( distances[struct1] - distances[struct2], 2 );
	    denominator1 = 2 * pow( alignment->getStructure(struct1)->alignedToUnalignedIndex(j) - 
				    alignment->getStructure(struct1)->alignedToUnalignedIndex(i),
				    qPower );
	    denominator2 = 2 * pow( alignment->getStructure(struct2)->alignedToUnalignedIndex(j) - 
				    alignment->getStructure(struct2)->alignedToUnalignedIndex(i),
				    qPower );
	    score1 = exp( numerator/denominator1 );
	    score2 = exp( numerator/denominator2 );
	    //printf("1 (%d,%d) struct1(%d,%d) struct2(%d,%d)\n",i,j,alignment->getStructure(struct1)->alignedToUnalignedIndex(i),alignment->getStructure(struct1)->alignedToUnalignedIndex(j),alignment->getStructure(struct2)->alignedToUnalignedIndex(i),alignment->getStructure(struct2)->alignedToUnalignedIndex(j));
	    //printf("numerator: %f, denominator1: %f, denominator2: %f, score1: %f, score2: %f\n",numerator,denominator1,denominator2,score1,score2);
	    qPerRes[struct1][i] += score1;
	    qPerRes[struct2][i] += score2;
	    qPerRes[struct1][j] += score1;
	    qPerRes[struct2][j] += score2;
	    //printf("qAln[%d][%d] = %f\n",struct1,struct2,qAln[struct1][struct2]);
	    qNorm[struct1][i]++;
	    qNorm[struct2][i]++;
	    qNorm[struct1][j]++;
	    qNorm[struct2][j]++;
	    //printf("1 (%d,%d) distances[%d] = %f\n",i,j,struct1,distances[struct1]);
	    //printf("        distances[%d] = %f\n",struct2,distances[struct2]);
            
	  } else {
	    // Complicated process for getting appropriate norms
	    //   
	    Structure* s1 = alignment->getStructure(struct1);
	    Structure* s2 = alignment->getStructure(struct2);
            
	    if ( distances[struct1] >= 0.0 && s2->getAlphabet()->isGap(s2->get(i)) && !s2->getAlphabet()->isGap(s2->get(j)) ) {
	      qNorm[struct1][i]++;
	      //printf("2 (%d,%d) distances[%d] = %f\n",i,j,struct1,distances[struct1]);
	    //printf("        distances[%d] = %f\n",struct2,distances[struct2]);
	    }
	    else if ( distances[struct1] >= 0.0 && !s2->getAlphabet()->isGap(s2->get(i)) && s2->getAlphabet()->isGap(s2->get(j)) ) {
	      qNorm[struct1][j]++;
	      //printf("3 (%d,%d) distances[%d] = %f\n",i,j,struct1,distances[struct1]);
	    //printf("        distances[%d] = %f\n",struct2,distances[struct2]);
	    }
	    else if (distances[struct2] >= 0.0 && s1->getAlphabet()->isGap(s1->get(i)) && !s1->getAlphabet()->isGap(s1->get(j)) ) {
	      qNorm[struct2][i]++;
	      //printf("4 (%d,%d) distances[%d] = %f\n",i,j,struct1,distances[struct1]);
	    //printf("        distances[%d] = %f\n",struct2,distances[struct2]);
	    }
	    else if (distances[struct2] >= 0.0 && !s1->getAlphabet()->isGap(s1->get(i)) && s1->getAlphabet()->isGap(s1->get(j))) {
	      qNorm[struct2][j]++;
	      //printf("5 (%d,%d) distances[%d] = %f\n",i,j,struct1,distances[struct1]);
	    //printf("        distances[%d] = %f\n",struct2,distances[struct2]);
	    }
	  }
	}
      }
    }
  }
  
  // Normalize qPerRes scores
  for (row=0; row<structCount; row++) {
    for (col=0; col<len; col++) {
      //printf("qPerRes[%d][%d] = %f\n",row,col,qPerRes[row][col]);
      //printf("qNorm[%d][%d]   = %d\n",row,col,qNorm[row][col]);
      if (qNorm[row][col] != 0) {
	qPerRes[row][col] /= qNorm[row][col];
      }
      else {
	qPerRes[row][col] = 0.0;
      }
      //printf("normed qPerRes[%d][%d] = %f\n",row,col,qPerRes[row][col]);
    }
  }
  
  // Delete qNorm
  for (row=0; row<structCount; row++) {
    delete qNorm[row];
  }
  delete qNorm;

  return 1;
} // end of QTools::qPerResidue() {


// printQ
//   
int QTools::printQ(FILE* outfile, int qhValues /*=0*/) {

   int i, j;
  if (qScores == 0) {
    return 0;
  }

  int structCount = alignment->getNumberStructures();

  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++)
    {
        if (qhValues) {
          fprintf(outfile,"%6.4f ",(float)(qScores[i][j]));
        } else {
          fprintf(outfile,"%6.4f ",(float)(1-qScores[i][j]));
        }
    }
    fprintf(outfile,"\n");
  }

  return 1;
}


// printMatrix - DEBUGGING METHOD
//
int QTools::printMatrix(FILE* outfile, float** mat) {
   int i, j;
  int structCount = alignment->getNumberStructures();

  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++)
      fprintf(outfile,"%6.4f ",mat[i][j]);
    fprintf(outfile,"\n");
  }

  return 1;
}


// printMatrix - DEBUGGING METHOD
//
int QTools::printMatrix(FILE* outfile, int** mat) {
   int i, j;
  int structCount = alignment->getNumberStructures();

  for (i=0; i<structCount; i++) {
    for (j=0; j<structCount; j++)
      fprintf(outfile,"%d ",mat[i][j]);
    fprintf(outfile,"\n");
  }

  return 1;
}


// printQPerResidue
//   
int QTools::printQPerResidue(FILE* outfile) {
   int i, j;
  int structCount = alignment->getNumberStructures();
  int alignedRes = 0;

  for (i=0; i<structCount; i++) {
    for (j=0; j<alignment->getStructure(i)->getUnalignedLength(); j++) {
      alignedRes = alignment->getStructure(i)->unalignedToAlignedIndex(j);
      fprintf(outfile,"%f ",qPerRes[i][alignedRes]);
    }
    fprintf(outfile,"\n");
  }

  return 0;
}


// getQAln
//   Calculate QAln scores for each pair of structures
//   QAln = sum((i<j-2), exp [ - ((r(ij) - r(i'j'))^2 / (2 * sigma(ij)^2)) ]
int QTools::getQAln(float** qAln, int** qNorm) {

  //printf("=>QTools::getQAln\n");

  int len = alignment->getNumberPositions();
  int structCount = alignment->getNumberStructures();
  //int norm = 0;

  float numerator = 0.0;
  float denominator = 0.0;
  float score = 0.0;
  float* distances = new float[structCount];
   int struct1, struct2, i, j, row;
  // Zero out qAln
  for (struct1=0; struct1<structCount; struct1++) {
    for (struct2=0; struct2<structCount; struct2++) {
      qAln[struct1][struct2] = 0.0;
    }
  }

  // Traverse columns i and j summing up the contributions to
  //   qAln from differences between interior backbone distances;
  //   ignore adjacent columns
  for (i=0; i<len-2; i++) {
    for (j=i+2; j<len; j++) {
        
      // Get backbone distances for current pair of columns
      getBackboneDistances(distances, i, j);

      // Unset distances for structures if the columns correspond to adjacent residues
      for (row=0; row<structCount; row++) {
          if ( alignment->getStructure(row)->alignedToUnalignedIndex(i) == alignment->getStructure(row)->alignedToUnalignedIndex(j)-1) {
                distances[row] = -1;
            }
      }

      // Calculate scores for current pair of columns
      // Traverse rows struct1 and struct2 calculating qAln
      //   scores for each pair of rows
      for (struct1=0; struct1<structCount-1; struct1++) {
        for (struct2=struct1+1; struct2<structCount; struct2++) {
          // if (neither structure is gapped) add new score to qAln
          if ( distances[struct1] >= 0.0 &&
               distances[struct2] >= 0.0 ) {
            numerator = -1 * pow( distances[struct1] - distances[struct2], 2 );
            // XXX - only takes distance along first structure into account
            //denominator = 2 * pow( j-i, qPower );
            denominator = 2 * pow( alignment->getStructure(struct1)->alignedToUnalignedIndex(j) -
                       alignment->getStructure(struct1)->alignedToUnalignedIndex(i),
                       qPower );
            score = exp( numerator/denominator );
            //printf("numerator: %f, denominator: %f, score: %f\n",numerator,denominator,score);
            qAln[struct1][struct2] += score;
            qAln[struct2][struct1] += score;
            //printf("qAln[%d][%d] = %f\n",struct1,struct2,qAln[struct1][struct2]);
            qNorm[struct1][struct2]++;
            qNorm[struct2][struct1]++;
          }
        }
      }
    }
  }

  //printf("<=QTools::getQAln\n");
  return 1;
}


// getQGap
//   Calculate and return QGap scores for each pair of structures
//   in the alignment
int QTools::getQGap(float** qGap, int** qNorm, int ends) {

  //printf("=>QTools::getQGap\n");

  int structCount = alignment->getNumberStructures();

  float qGapPair1 = 0.0;   //QGap score for a pair of structures
  float qGapPair2 = 0.0;   //QGap score for a pair of structures

  int struct1, struct2;
  for (struct1=0; struct1<structCount; struct1++) {
    for (struct2=0; struct2<structCount; struct2++) {
      qGap[struct1][struct2] = 0.0;
    }
  }
  
  for (struct1=0; struct1<structCount-1; struct1++) {
    for (struct2=struct1+1; struct2<structCount; struct2++) {
      // getQGap(x,y,e) gets score for x with respect to y
      //   e is whether the ends should be counted or not
      qGapPair1 = getQGap(struct1, struct2, qNorm, ends);
      qGapPair2 = getQGap(struct2, struct1, qNorm, ends);
      if (qGapPair1 >= 0.0 && qGapPair2 >= 0.0) {
	//printf("qGapPair1: %f, qGapPair2: %f\n",qGapPair1,qGapPair2);
	qGap[struct1][struct2] = qGapPair1 + qGapPair2;
	qGap[struct2][struct1] = qGapPair1 + qGapPair2;
	//qNorm[struct1][struct2]++;
	//qNorm[struct2][struct1]++;
      }
    }
  }

  //printf("<=QTools::getQGap\n");
  return 1;
}


// getQGap
//   Return the QGap score for a structure with respect
//   to a second structure in the alignment
float QTools::getQGap(int s1, int s2, int** qNorm, int ends) {

  //printf("  =>QTools::getQGap(%d,%d,%d)\n",s1,s2,ends);

  int len = alignment->getNumberPositions();
  int gapHead = -1;
  int gapTail = getGapTail(gapHead, s1, s2);
  //int norm = 0;
  float dist1 = 0.0;
  float dist2 = 0.0;
  float numerator = 0.0;
  float denominator = 0.0;
  float score1 = 0.0;
  float score2 = 0.0;
  float qGap = 0.0;
  AlignedStructure* struct1 = alignment->getStructure(s1);
  AlignedStructure* struct2 = alignment->getStructure(s2);
   int i, j;
  if (struct1 == 0 || struct2 == 0) {
    printf("    dying\n");
    printf("  <=QTools::getQGap\n");
    return -1.0;
  }

  gapHead = -1;
  gapTail = getGapTail(gapHead, s1, s2);
  for (i=0; i<len; i++) {
    //gapHead = -1;
    //gapTail = getGapTail(gapHead, s1, s2);
    if (i == gapTail) {
      gapHead = getGapHead(gapTail, s1, s2);
      gapTail = getGapTail(gapHead, s1, s2);
    }
    // if (struct1 has insertion (gap in struct2))
    if ( !struct1->getAlphabet()->isGap(struct1->get(i)) &&
	  struct2->getAlphabet()->isGap(struct2->get(i)) ) {
      for (j=0; j<len; j++) {
	// if (the structure pair have aligned elements at one end &&
	//     measuring non-adjacent residues)
	if ( !struct1->getAlphabet()->isGap(struct1->get(j)) &&
	     !struct2->getAlphabet()->isGap(struct2->get(j)) &&
	     abs( alignment->getStructure(s1)->alignedToUnalignedIndex(i) -
		  alignment->getStructure(s1)->alignedToUnalignedIndex(j) ) > 1 &&
	     abs( alignment->getStructure(s2)->alignedToUnalignedIndex(i) -
		  alignment->getStructure(s2)->alignedToUnalignedIndex(j) ) > 1) {
	  
	  // Head case - haven't reached aligned elements yet
	  if (gapHead == -1 && ends != 0) {
	    //printf("  Headcase\n");
	    dist1 = getBackboneDistance(s1,i,j);
	    dist2 = getBackboneDistance(s2,gapTail,j);
	    numerator = -1 * pow( dist1 - dist2, 2 );
	    //denominator = 2 * pow( abs(j-i), qPower );
	    denominator = 2 * pow( abs(alignment->getStructure(s1)->alignedToUnalignedIndex(j) -
				       alignment->getStructure(s1)->alignedToUnalignedIndex(i)),
				   qPower );
	    score1 = exp( numerator/denominator );
	    //printf("  dist1: %f, dist2: %f, score1: %f\n",dist1,dist2,score1);
	    qGap +=  score1;
	    qNorm[s1][s2]++;
	    qNorm[s2][s1]++;
	  }
	  // Tail case - no more aligned elements
	  else if (gapTail == -1 && ends != 0) {
	    //printf("  Tailcase\n");
	    dist1 = getBackboneDistance(s1,i,j);
	    dist2 = getBackboneDistance(s2,gapHead,j);
	    numerator = -1 * pow( dist1 - dist2, 2 );
	    //denominator = 2 * pow( abs(j-i), qPower );
	    denominator = 2 * pow( abs(alignment->getStructure(s1)->alignedToUnalignedIndex(j) -
				       alignment->getStructure(s1)->alignedToUnalignedIndex(i)),
				   qPower );
	    score1 = exp( numerator/denominator );
	    //printf("  dist1: %f, dist2: %f, score1: %f\n",dist1,dist2,score1);
	    qGap +=  score1;
	    qNorm[s1][s2]++;
	    qNorm[s2][s1]++;
	  }
	  // Body case - aligned elements abound
	  else if (gapHead >= 0 && gapTail >= 0) {
	    //printf("  Bodycase\n");
	    dist1 = getBackboneDistance(s1,i,j);
	    dist2 = getBackboneDistance(s2,gapHead,j);
	    numerator = -1 * pow( dist1 - dist2, 2 );
	    //denominator = 2 * pow( abs(j-i), qPower );
	    denominator = 2 * pow( abs(alignment->getStructure(s1)->alignedToUnalignedIndex(j) -
				       alignment->getStructure(s1)->alignedToUnalignedIndex(i)),
				   qPower );
	    score1 = exp( numerator/denominator );
	    
	    dist2 = getBackboneDistance(s2,gapTail,j);
	    numerator = -1 * pow( dist1 - dist2, 2 );
	    //denominator = 2 * pow( j-i, qPower );
	    score2 = exp( numerator/denominator );
	    
	    //printf("  dist1: %f, dist2: %f, score1: %f, score2: %f\n",dist1,dist2,score1,score2);
	    qGap +=  MAX(score1,score2);
	    qNorm[s1][s2]++;
	    qNorm[s2][s1]++;
	  }
	  //norm++;
	}
      }
    }
  }

  //printf("  <=QTools::getQGap\n");
  return qGap;
}


// getGapHead
//   Return the column before the next gapped region,
//   -1 if end (gaps at head before the first aligned pair)
int QTools::getGapHead(int gapTail, int s1, int s2) {

  //printf("=>QTools::getGapHead\n");

  if (gapTail < 0) {
    //printf("   Nope1\n");
    //printf("<=QTools::getGapHead\n");
    return -1;
  }

  int len = alignment->getNumberPositions();
  AlignedStructure* struct1 = alignment->getStructure(s1);
  AlignedStructure* struct2 = alignment->getStructure(s2);

  int i;
  for (i = gapTail; i<len-1; i++) {
    if ( struct1->getAlphabet()->isGap(struct1->get(i+1)) ||
	 struct2->getAlphabet()->isGap(struct2->get(i+1)) ) {
      //printf("   Yep1\n");
      //printf("<=QTools::getGapHead\n");
      return i;
    }
  }

  //printf("   Yep2\n");
  //printf("<=QTools::getGapHead\n");
  return i;
}


// getGapTail
//   Return the next column where struct1 and struct2 are aligned,
//   -1 if end (gaps at tail after the last aligned pair)
int QTools::getGapTail(int gapHead, int s1, int s2) {
  
  //printf("=>QTools::getGapTail\n");

  if (gapHead < -1) {
    //printf("   Nope1\n");
    //printf("<=QTools::getGapTail\n");
    return -1;
  }
   int i;
  int len = alignment->getNumberPositions();
  AlignedStructure* struct1 = alignment->getStructure(s1);
  AlignedStructure* struct2 = alignment->getStructure(s2);

  for (i=gapHead+1; i<len; i++) {
    if ( !struct1->getAlphabet()->isGap(struct1->get(i)) &&
	 !struct2->getAlphabet()->isGap(struct2->get(i)) ) {
      //printf("   Yep1\n");
      //printf("<=QTools::getGapTail\n");
      return i;
    }
  }

  //printf("   Nope2\n");
  //printf("<=QTools::getGapTail\n");
  return -1;
}


// getBackboneDistances
//   Calculate distance between the two backbone atoms that
//   correspond to the aligned residues in two columns;  Return
//   the distances in an array
int QTools::getBackboneDistances(float* distances, int bb1, int bb2) {

  //printf("=>QTools::getBackboneDistances\n");
   int i;
  for (i=0; i<alignment->getNumberStructures(); i++) {
    //printf("%d",i);
    distances[i] = getBackboneDistance(i, bb1, bb2);
  }
  //printf("\n");

  //printf("<=QTools::getBackboneDistances\n");
  return 1;
}


// getBackboneDistance
//   Calculate distance between two backbone atoms in a structure;
//   Returns -1 on error
float QTools::getBackboneDistance(int structure, int bb1, int bb2) {

  //printf("=>QTools::getBackboneDistance\n");
  //printf("   bb1: %d, bb2: %d\n",bb1,bb2);

  AlignedStructure* tempStruct = alignment->getStructure(structure);
  if (tempStruct == 0) {
    //printf("   Nope1\n");
    //printf("<=QTools::getBackboneDistance\n");
    return -1;
  }

  Coordinate3D tempCoord1 = tempStruct->getCoordinate(bb1);
  Coordinate3D tempCoord2 = tempStruct->getCoordinate(bb2);
  return tempCoord1.getDistanceTo(tempCoord2);
}


