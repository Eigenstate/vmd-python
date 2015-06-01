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

#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "sequence.h"
#include "coordinate3D.h"
#include "PointerList.h"

class Symbol;
class Alphabet;
class Residue;

class Structure : public Sequence {

    public:
    Structure(Alphabet* alphabet, const char* name=0) : Sequence(alphabet,name) {}
    virtual ~Structure();
    
    virtual void addResidue(char c, float x, float y, float z);
    virtual void addResidue(const char* n, float x, float y, float z);
    virtual void addResidue(Symbol& symbol,  float x, float y, float z);
    virtual void addResidue(char c, Coordinate3D coord, Residue* residue);
    virtual void addResidue(const char* n, Coordinate3D coord, Residue* residue);
    virtual void addResidue(Symbol& symbol, Coordinate3D coord, Residue* residue);
    virtual Coordinate3D getCoordinate(int residueIndex);
    virtual Residue* getResidue(int residueIndex);
    
    private:
    PointerList backboneCoordinates;
    PointerList residues;
    
    friend class StructureTest;
};

#endif
