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
 * Author(s): John Eargle, Michael Bach, Elijah Roberts
 */

#include <stdio.h>
#include <math.h>
#include "symbol.h"
#include "alphabet.h"
#include "sequence.h"
#include "ShortIntList.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"
#include "PIDTools.h"
#include "sequenceQR.h"
#include "aminoAcidQR.h"

// Constructor
AminoAcidQR::AminoAcidQR(SequenceAlignment *alignment, int preserveCount, int performGapScaling, float gapScaleParameter, float normOrder)
: SequenceQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder) {
    cMj = 24;
    createMatrix();
    initializeMatrix();
}

// Destructor
AminoAcidQR::~AminoAcidQR(){}

void AminoAcidQR::createMatrix(){
   int i, j, k;
    //Create a matrix containing the data representing this alignment.
    matrix = new float**[cMi];
    for (i=0; i<cMi; i++) {
    
        matrix[i] = new float*[cMj];
        for (j=0; j<cMj; j++) {
        
            matrix[i][j] = new float[cMk];
            for (k=0; k<cMk; k++) {
            
                //Get the sequence we are working with.
                AlignedSequence* sequence = alignment->getSequence(k);
                
                //Fill in the value for the matrix element.
                switch (j) {
                    case 0:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'A')?1:0;
                        break;
                    case 1:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'B')?1:0;
                        break;
                    case 2:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'C')?1:0;
                        break;
                    case 3:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'D')?1:0;
                        break;
                    case 4:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'E')?1:0;
                        break;
                    case 5:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'F')?1:0;
                        break;
                    case 6:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'G')?1:0;
                        break;
                    case 7:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'H')?1:0;
                        break;
                    case 8:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'I')?1:0;
                        break;
                    case 9:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'K')?1:0;
                        break;
                    case 10:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'L')?1:0;
                        break;
                    case 11:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'M')?1:0;
                        break;
                    case 12:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'N')?1:0;
                        break;
                    case 13:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'P')?1:0;
                        break;
                    case 14:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'Q')?1:0;
                        break;
                    case 15:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'R')?1:0;
                        break;
                    case 16:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'S')?1:0;
                        break;
                    case 17:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'T')?1:0;
                        break;
                    case 18:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'V')?1:0;
                        break;
                    case 19:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'W')?1:0;
                        break;
                    case 20:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'X')?1:0;
                        break;
                    case 21:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'Y')?1:0;
                        break;
                    case 22:
                        matrix[i][j][k] = (sequence->get(i).getOne() == 'Z')?1:0;
                        break;
                    case 23:
                        matrix[i][j][k] = sequence->getAlphabet()->isGap(sequence->get(i))?1:0;
                        break;
                }
            }
        }
    }
}

