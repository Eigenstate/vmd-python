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

#ifndef QTOOLS_H
#define QTOOLS_H

#include <stdio.h>

class ContactList;
class Structure;
class StructureAlignment;

class QTools {

 public:
  QTools(StructureAlignment* sa);
  ~QTools();
    
  int q(int ends, int excludeAln=0, int excludeGap=0);
  int qPerResidue();
  int printQ(FILE* outfile, int qhValues=0);
  int printMatrix(FILE* outfile, float** mat);
  int printMatrix(FILE* outfile, int** mat);
  int printQPerResidue(FILE* outfile);
    
 private:
  int getQAln(float** qAln, int** qNorm);             // for q()
  int getQGap(float** qGap, int** qNorm, int ends);   // for q()
  float getQGap(int struct1, int struct2, int** qNorm, int ends);
  int getGapHead(int gapTail, int s1, int s2);
  int getGapTail(int gapHead, int s1, int s2);
  int getBackboneDistances(float* distances, int bb1, int bb2);
  float getBackboneDistance(int structure, int bb1, int bb2);
    
 private:
  StructureAlignment* alignment;
  float **qScores;  // Q scores between each pair of structures
  float **qPerRes;  // Q per residue scores
  float qPower;     // Empirically derived constant for Q calculation
};

#endif
