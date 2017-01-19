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

#include <stdlib.h>
#include "PointerList.h"
#include "Atom.h"
#include "AtomList.h"
#include "Residue.h"
#include "structure.h"

Structure::~Structure()
{
    // Free the backbone coordinates, since these are copies we made ourself.
    int i;
    for (i=0; i<backboneCoordinates.getSize(); i++)
    {
        if (backboneCoordinates.get(i) != NULL)
        {
            delete (Coordinate3D*)backboneCoordinates.get(i);
            backboneCoordinates.set(i, NULL);
        }
    }
    
    // Free the residues, since that is the contract when they get passed in.
    for (i=0; i<residues.getSize(); i++)
    {
        if (residues.get(i) != NULL)
        {
            delete (Residue*)residues.get(i);
            residues.set(i, NULL);
        }
    }
}

void Structure::addResidue(char c, float x, float y, float z)
{
    addResidue(c, Coordinate3D(x,y,z), NULL);
}

void Structure::addResidue(const char* n, float x, float y, float z)
{
    addResidue(n, Coordinate3D(x,y,z), NULL);
}

void Structure::addResidue(Symbol& symbol, float x, float y, float z)
{
    addResidue(symbol, Coordinate3D(x,y,z), NULL);
}

void Structure::addResidue(char c, Coordinate3D coord, Residue* residue)
{
    Sequence::add(c);
    backboneCoordinates.add(new Coordinate3D(coord));
    
    if (residue != NULL)
        residues.add(residue);
    else
        residues.add(new Residue());
}

void Structure::addResidue(const char* n, Coordinate3D coord, Residue* residue)
{
    Sequence::add(n);
    backboneCoordinates.add(new Coordinate3D(coord));
    
    if (residue != NULL)
        residues.add(residue);
    else
        residues.add(new Residue());
}

void Structure::addResidue(Symbol& symbol, Coordinate3D coord, Residue* residue)
{
    Sequence::add(symbol);
    backboneCoordinates.add(new Coordinate3D(coord));
    
    if (residue != NULL)
        residues.add(residue);
    else
        residues.add(new Residue());
}

Coordinate3D Structure::getCoordinate(int residueIndex)
{
    return *(Coordinate3D*)backboneCoordinates.get(residueIndex);
}

Residue* Structure::getResidue(int residueIndex)
{
    return (Residue*)residues.get(residueIndex);
}

