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
#include <stdio.h>
#include "alphabet.h"
#include "symbol.h"
#include "ShortIntList.h"
#include "alignedSequence.h"
#include "sequenceAlignment.h"

SequenceAlignment::~SequenceAlignment()
{
   int i;
    // Free the sequences, since that is the contract when they get passed in.
    for (i=0; i<getSize(); i++)
    {
        AlignedSequence* sequence = (AlignedSequence*)get(i);
        if (sequence != NULL)
        {
            delete sequence;
            sequence = NULL;
            set(i, NULL);
        }
    }
}

Alphabet* SequenceAlignment::getAlphabet()
{
    return alphabet;
}

bool SequenceAlignment::addSequence(AlignedSequence* sequence)
{
    if (getSize() == 0)
    {
        alphabet = sequence->getAlphabet();
        numberPositions = sequence->getSize();
        add(sequence);
        return true;
    }
    else
    {
        if (sequence->getAlphabet() == alphabet && sequence->getSize() == numberPositions)
        {
            add(sequence);
            return true;
        }
    }

    return false;
}

AlignedSequence* SequenceAlignment::getSequence(int index)
{
    return (AlignedSequence*)get(index);
}

int SequenceAlignment::getNumberPositions()
{
    return numberPositions;
}

int SequenceAlignment::getNumberSequences()
{
    return getSize();
}

Symbol SequenceAlignment::getPosition(int sequenceIndex, int positionIndex)
{
    if (sequenceIndex >= 0 && sequenceIndex < getSize() && positionIndex >= 0 && positionIndex < numberPositions)
    {
        AlignedSequence* sequence = (AlignedSequence*)get(sequenceIndex);
        if (sequence != NULL)
            return sequence->get(positionIndex);
    }
    return alphabet->getUnknown();
}

