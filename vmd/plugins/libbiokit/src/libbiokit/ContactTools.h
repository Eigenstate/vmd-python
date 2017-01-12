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

#ifndef CONTACTTOOLS_H
#define CONTACTTOOLS_H

class ContactList;
class Structure;

class ContactTools {

    public:
    static ContactList* getContacts(Structure* structure, double distanceCutoff=5.0, int minSequenceDistance=4, int maxSequenceDistance=-1);
    static ContactList* getFormedNativeContacts(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation=1.0);
    static double getFractionNativeContacts(Structure* nativeStructure, Structure* comparisonStructure, double distanceCutoff=5.0, int minSequenceDistance=4, double maxDistanceDeviation=1.0, int maxSequenceDistance=-1);
    static double getFractionNativeContacts(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation=1.0);
    static double getContactOrder(Structure* structure, double distanceCutoff=5.0, int minSequenceDistance=4, int maxSequenceDistance=-1);
    static double getContactOrder(ContactList* contacts);
    static double getPartialContactOrder(Structure* nativeStructure, Structure* comparisonStructure, double distanceCutoff=5.0, int minSequenceDistance=4, double maxDistanceDeviation=1.0, int maxSequenceDistance=-1);
    static double getPartialContactOrder(ContactList* nativeContacts, Structure* comparisonStructure, double maxDistanceDeviation=1.0);
    
};

#endif
