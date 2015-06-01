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
#include "binaryQR.h"

// Constructor
BinaryQR::BinaryQR(SequenceAlignment *alignment, int preserveCount, int performGapScaling, float gapScaleParameter, float normOrder)
: SequenceQR(alignment, preserveCount, performGapScaling, gapScaleParameter, normOrder) {
    cMj = 2;
    createMatrix();
    initializeMatrix();
}

// Destructor
BinaryQR::~BinaryQR(){}

void BinaryQR::createMatrix(){
   int i, k;    
    //Create a matrix containing the data representing this alignment.
    matrix = new float**[cMi];
    for (i=0; i<cMi; i++) {
        matrix[i] = new float*[cMj];
        matrix[i][0] = new float[cMk];
        matrix[i][1] = new float[cMk];
        for (k=0; k<cMk; k++) {
            //Get the sequence we are working with.
            AlignedSequence* sequence = alignment->getSequence(k);
                
                //Fill in the value for the matrix element.
            int result = sequence->getAlphabet()->isGap(sequence->get(i));
            matrix[i][0][k] = result?0:1;
            matrix[i][1][k] = result?1:0;
	
        }
    }
}

