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
 * Author(s): Patrick O'Donoghue, Michael Januszyk, John Eargle, Elijah Roberts
 */

#include <stdlib.h>
#include <math.h>
#include "Atom.h"
#include "Residue.h"
#include "structure.h"
#include "Contact.h"
#include "ContactList.h"
#include "ContactTools.h"


/**
 * Calculates thecontacts in a structure.
 *
 * @param structure The structure.
 * @param distanceCutoff The maximum distance between two residues from them to be considered in contact.
 * @param minSequenceDistance The minimum separation in the sequence for two residues to be considered contacts.
 * @param maxSequenceDistance The maximum separation in the sequence for two residues to be considered contacts.
 * @return The contacts present in the structure.
 */
ContactList* ContactTools::getContacts(Structure* structure, double distanceCutoff, int minSequenceDistance, int maxSequenceDistance)
{
    // Find all of the contacts within the cutoff distance.
    ContactList* contacts = new ContactList;
    int length = structure->getSize();
    int i, j, k, l;
    for (i=0; i<length-minSequenceDistance; i++)
    {
        int maxJ = length;
        if (maxSequenceDistance >= 0)
        {
            maxJ = i+maxSequenceDistance+1;
            if (maxJ > length) maxJ = length;
        }
            
        for (j=i+minSequenceDistance; j<maxJ; j++)
        {
            Residue* residue1 = structure->getResidue(i);
            Residue* residue2 = structure->getResidue(j);
            
            // If this is the same residue, process each atom pair only once.
            if (i == j)
            {
                for (k=0; k<residue1->getNumberAtoms()-1; k++)
                {
                    Atom* atom1 = residue1->getAtom(k);
                    for (l=k+1; l<residue2->getNumberAtoms(); l++)
                    {
                        Atom* atom2 = residue2->getAtom(l);
                        if (atom1->getDistanceTo(*atom2) <= distanceCutoff)
                        {
                            contacts->addContact(new Contact(structure, i, k, j, l));
                        }
                    }
                }
            }
            
            // Otherwise, process each atom pair.
            else
            {
                for (k=0; k<residue1->getNumberAtoms(); k++)
                {
                    Atom* atom1 = residue1->getAtom(k);
                    for (l=0; l<residue2->getNumberAtoms(); l++)
                    {
                        Atom* atom2 = residue2->getAtom(l);
                        if (atom1->getDistanceTo(*atom2) <= distanceCutoff)
                        {
                            contacts->addContact(new Contact(structure, i, k, j, l));
                        }
                    }
                }
            }
        }
    }
    
    return contacts;
}

ContactList* ContactTools::getFormedNativeContacts(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation)
{
    // Get the sequence distance for each formed native contact.
    ContactList* formedNativeContacts = new ContactList;
    int i;
    for (i=0; i<nativeContacts->getNumberContacts(); i++)
    {
        Contact* nativeContact = nativeContacts->getContact(i);
        double nativeDistance = nativeContact->getContactDistance();
        
        // Find the contact in the comparison structure.
        Contact* comparisonContact = new Contact(comparisonStructure, nativeContact->getResidue1Index(), nativeContact->getAtom1Index(), nativeContact->getResidue2Index(), nativeContact->getAtom2Index());
        double comparisonDistance = comparisonContact->getContactDistance();
        
        if (nativeDistance-maxDistanceDeviation <= comparisonDistance && comparisonDistance <= nativeDistance+maxDistanceDeviation)
        {
            formedNativeContacts->addContact(comparisonContact);
        }
        else
        {
            // Otherwise delete the contact.
            delete comparisonContact;
            comparisonContact = NULL;
        }
    }
    
    return formedNativeContacts;
}

/**
 * Calculates the fraction of native contacts between two structures.
 *
 * @param nativeStructure The native structure.
 * @param comparisonStructure The structure to compare to the native.
 * @param distanceCutoff The maximum distance between two residues from them to be considered in contact.
 * @param minSequenceDistance The minimum separation in the sequence for two residues to be considered native.
 * @param maxSequenceDistance The maximum separation in the sequence for two residues to be considered contacts.
 * @return The fraction of native contacts present in the comparison structure.
 */
double ContactTools::getFractionNativeContacts(Structure* nativeStructure, Structure* comparisonStructure, double distanceCutoff, int minSequenceDistance, double maxDistanceDeviation, int maxSequenceDistance)
{
    ContactList* contacts = getContacts(nativeStructure, distanceCutoff, minSequenceDistance, maxSequenceDistance);
    double d = getFractionNativeContacts(contacts, comparisonStructure, maxDistanceDeviation);
    delete contacts;
    return d;
}

/**
 * Calculates the fraction of native contacts between two structures.
 *
 * @param nativeContacts The native contacts.
 * @param comparisonStructure The structure to compare to the native.
 * @param distanceCutoff The maximum distance between two residues from them to be considered in contact.
 * @param minSequenceDistance The minimum separation in the sequence for two residues to be considered native.
 * @return The fraction of native contacts present in the comparison structure.
 */
double ContactTools::getFractionNativeContacts(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation)
{
    if (nativeContacts->getNumberContacts() == 0) return 0.0;
    
    // Get the formed native contacts.
    ContactList* formedNativeContacts = getFormedNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation);
    
    // Return the fraction.
    return double(formedNativeContacts->getNumberContacts())/double(nativeContacts->getNumberContacts());
}

/**
 * Calculates the contact order of a structure.
 *
 * @param structure The structure.
 * @param distanceCutoff The maximum distance between two residues from them to be considered in contact.
 * @param minSequenceDistance The minimum separation in the sequence for two residues to be considered contacts.
 * @param maxSequenceDistance The maximum separation in the sequence for two residues to be considered contacts.
 * @return The number of contacts present in the structure.
 */
double ContactTools::getContactOrder(Structure* structure, double distanceCutoff, int minSequenceDistance, int maxSequenceDistance)
{
    ContactList* contacts = getContacts(structure, distanceCutoff, minSequenceDistance, maxSequenceDistance);
    double d = getContactOrder(contacts);
    delete contacts;
    return d;
}

double ContactTools::getContactOrder(ContactList* contacts)
{
   int i;
    // Get the sequence distance for each contact.
    int totalSequenceDistance=0;    
    for (i=0; i<contacts->getNumberContacts(); i++)
    {
        Contact* contact = contacts->getContact(i);
        totalSequenceDistance += abs(contact->getResidue1Index()-contact->getResidue2Index());
    }
    
    if (contacts->getNumberContacts() == 0) return 0.0;
    return double(totalSequenceDistance)/double(contacts->getNumberContacts());
}

double ContactTools::getPartialContactOrder(Structure* nativeStructure, Structure* comparisonStructure, double distanceCutoff, int minSequenceDistance, double maxDistanceDeviation, int maxSequenceDistance)
{
    ContactList* nativeContacts = getContacts(nativeStructure, distanceCutoff, minSequenceDistance, maxSequenceDistance);
    double d = getPartialContactOrder(nativeContacts, comparisonStructure, maxDistanceDeviation);
    delete nativeContacts;
    return d;
}

double ContactTools::getPartialContactOrder(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation)
{
    return getContactOrder(getFormedNativeContacts(nativeContacts, comparisonStructure, maxDistanceDeviation));
}
