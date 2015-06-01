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
#include "symbol.h"
#include "alphabet.h"
#include "ShortIntList.h"
#include "alignedSequence.h"

// --------------------------------------------------------------------
AlignedSequence::AlignedSequence(const AlignedSequence &as) :
                                            Sequence(as)
{
   residueToPosition = as.residueToPosition;
   positionToResidue = as.positionToResidue;
}


// --------------------------------------------------------------------
int AlignedSequence::getSize()
{
    return getNumberPositions();
}

// --------------------------------------------------------------------
int AlignedSequence::getNumberPositions()
{
    return positionToResidue.getSize();
}

// --------------------------------------------------------------------
int AlignedSequence::getNumberResidues()
{
    return Sequence::getSize();
}

// --------------------------------------------------------------------
Symbol& AlignedSequence::get(int positionIndex)
{
    return getPosition(positionIndex);
}

// --------------------------------------------------------------------
Symbol& AlignedSequence::getPosition(int positionIndex)
{
    if (positionIndex >= 0 && positionIndex < positionToResidue.getSize())
    {
        int residueIndex = positionToResidue.get(positionIndex);
        if (residueIndex != ShortIntList::MAX)
        {
            return Sequence::get(residueIndex);
        }
        return alphabet->getGap();
    }
    return alphabet->getUnknown();
}

// --------------------------------------------------------------------
Symbol& AlignedSequence::getResidue(int residueIndex)
{
    return Sequence::get(residueIndex);
}

// --------------------------------------------------------------------
int AlignedSequence::getResidueForPosition(int positionIndex)
{
    return positionToResidue.get(positionIndex);
}

// --------------------------------------------------------------------
int AlignedSequence::getPositionForResidue(int residueIndex)
{
    return residueToPosition.get(residueIndex);
}

// --------------------------------------------------------------------
void AlignedSequence::addGap()
{
    add(alphabet->getGap());
}

// --------------------------------------------------------------------
void AlignedSequence::optimize()
{
   residueToPosition.optimize();
   positionToResidue.optimize();
}

// --------------------------------------------------------------------
void AlignedSequence::addSymbolIndex(int symbolIndex)
{
//   printf("(symbolIndex:%d,isGap:%d,posSize:%d,resSize%d,resIndex:%d)",
//           symbolIndex, 
//           alphabet->isGap(alphabet->getSymbol(symbolIndex)),
//           positionToResidue.getSize(),
//           residueToPosition.getSize(),
//           Sequence::getSize());
    if (alphabet->isGap(alphabet->getSymbol(symbolIndex)))
    {
        positionToResidue.add(ShortIntList::MAX);
    }
    else
    {
        int positionIndex = positionToResidue.getSize();
        int residueIndex = Sequence::getSize();
        positionToResidue.add(residueIndex);
        residueToPosition.add(positionIndex);
        Sequence::addSymbolIndex(symbolIndex);
    }
}

// --------------------------------------------------------------------

