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

#ifndef Contact_H
#define Contact_H

class Atom;
class Residue;
class Structure;

class Contact
{
  
    public:
    Contact(Structure* structure, int residue1Index, int atom1Index, int residue2Index, int atom2Index);
    Contact(const Contact& copyFrom);
    virtual ~Contact() {}
    virtual Structure* getStructure() {return structure;}
    virtual int getResidue1Index() {return residue1Index;}
    virtual int getResidue2Index() {return residue2Index;}
    virtual Residue* getResidue1();
    virtual Residue* getResidue2();
    virtual int getAtom1Index() {return atom1Index;}
    virtual int getAtom2Index() {return atom2Index;}
    virtual Atom* getAtom1();
    virtual Atom* getAtom2();
    virtual double getContactDistance();
    virtual bool operator==(const Contact& contact) const;
    virtual bool operator!=(const Contact& contact) const;
    

    private:
    Structure *structure;
    int residue1Index, atom1Index, residue2Index, atom2Index;

    friend class ContactTest;    
};

#endif
