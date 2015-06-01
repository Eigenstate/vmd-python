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
 * Author(s): Elijah Roberts
 */

#include <stdlib.h>
#include <math.h>
#include "symbol.h"
#include "alphabet.h"
#include "ShortIntList.h"
#include "alignedSequence.h"
#include "PIDTools.h"

/**
 * This method gets the percentage of positions that two sequences share.
 *
 * @param sequence1 The first sequence.
 * @param sequence2 The second sequence.
 * @return  The percentage of elements that the two sequences have in common or 0.0 on error.
 */
double PIDTools::getPercentIdentity(AlignedSequence *sequence1, AlignedSequence *sequence2) {

    //Make sure the sequences are of the same length.
    if (sequence1->getSize() !=  sequence2->getSize()) return 0.0;

    //Go through the symbols and count the ones that are the same.
    int identityCount = 0;
    int alignmentLength = 0;
    int i;
    for (i=0; i<sequence1->getSize(); i++) {
        
        //Get the symbols.
        Symbol symbol1 = sequence1->get(i);
        Symbol symbol2 = sequence2->get(i);
        
        //If neither one is a gap and they are the same, increment the identity count.
        if (!sequence1->getAlphabet()->isGap(symbol1) && !sequence2->getAlphabet()->isGap(symbol2) && symbol1 == symbol2) {
            identityCount++;
        }
        
        //If both are not gaps, increment the alignment length.
        if (!sequence1->getAlphabet()->isGap(symbol1) || !sequence2->getAlphabet()->isGap(symbol2))
            alignmentLength++;
    }
    
    if (alignmentLength == 0.0)
        return 0.0;
    return ((double)identityCount)/((double)alignmentLength);
}

