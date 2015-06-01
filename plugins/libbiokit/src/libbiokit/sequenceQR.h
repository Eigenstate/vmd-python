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

#ifndef SEQUENCEQR_H
#define SEQUENCEQR_H

class SequenceAlignment;

class SequenceQR {
  
 public:
  SequenceQR(SequenceAlignment* align, int pCount, int pGapScaling, float gapScaleP, float normOrd);
  virtual ~SequenceQR();
  virtual void createMatrix() = 0;
  virtual void initializeMatrix();
  virtual SequenceAlignment* qrWithPIDCutoff(float identityCutoff);
  virtual SequenceAlignment* qrWithPercentCutoff(int percent);
  
 private:
  void householder(int currentColumn);
  void permuteColumns(int currentColumn);
  int isSequenceAboveIdentityCutoff(int currentColumn, float identityCutoff);
  float frobeniusNormByK(int k, int currentRow);
  float frobeniusNormByJ(int j);
  void scaleGapData();


  
 protected:
  /*
    Anytime you access the third index from coordMatrix, go through the
    columnList; So coordMatrix[i][j][k] becomes coordMatrix[i][j][columnList[k]];
    This level of indirection eliminates array copying and saves the sequence
    order.
  */
  SequenceAlignment* alignment;
  int preserveCount;
  int performGapScaling;
  float gapScaleParameter;
  float normOrder;
  float*** matrix;
  int* columnList;         // indices for columns which may be permuted
  int cMi;                 // number of residue columns in coordMatrix (alignment column)
  int cMj;                 // number of coordinates for a coordMatrix residue (x,y,z,gap)
  int cMk;                 // number of sequences in coordMatrix
};

#endif
