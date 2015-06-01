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
#include "Atom.h"
#include "Residue.h"
#include "structure.h"
#include "Contact.h"

Contact::Contact(Structure* structure, int residue1Index, int atom1Index, int residue2Index, int atom2Index)
: structure(structure), residue1Index(residue1Index), atom1Index(atom1Index), residue2Index(residue2Index), atom2Index(atom2Index) {}

Contact::Contact(const Contact& copyFrom)
{
    structure = copyFrom.structure;
    residue1Index = copyFrom.residue1Index;
    atom1Index = copyFrom.atom1Index;
    residue2Index = copyFrom.residue2Index;
    atom2Index = copyFrom.atom2Index;
}

Residue* Contact::getResidue1()
{
    if (structure != NULL)
        return structure->getResidue(residue1Index);
    return NULL;
}

Residue* Contact::getResidue2()
{
    if (structure != NULL)
        return structure->getResidue(residue2Index);
    return NULL;
}

Atom* Contact::getAtom1()
{
    if (structure != NULL)
        if (structure->getResidue(residue1Index) != NULL)
            return structure->getResidue(residue1Index)->getAtom(atom1Index);
    return NULL;
}

Atom* Contact::getAtom2()
{
    if (structure != NULL)
        if (structure->getResidue(residue2Index) != NULL)
            return structure->getResidue(residue2Index)->getAtom(atom2Index);
    return NULL;
}

double Contact::getContactDistance()
{
    Atom* atom1 = getAtom1();
    Atom* atom2 = getAtom2();
    if (atom1 != NULL && atom2 != NULL)
        return atom1->getDistanceTo(*atom2);
    else
        return -1.0;
}

bool Contact::operator==(const Contact& contact) const
{   
    if (structure != contact.structure || residue1Index != contact.residue1Index || atom1Index != contact.atom1Index || residue2Index != contact.residue2Index || atom2Index != contact.atom2Index)
        return false;
    return true;
}

bool Contact::operator!=(const Contact& contact) const
{   
    return !(*this == contact);
}

