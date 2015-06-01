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
#include "Contact.h"
#include "ContactList.h"

ContactList::ContactList(Contact* contact)
{
    addContact(contact);
}

void ContactList::addContact(Contact* contact)
{
    if (contact != NULL)
        add(contact);
}

Contact* ContactList::getContact(int index)
{
    return (Contact*)get(index);
}

int ContactList::getNumberContacts()
{
    return getSize();
}

ContactList* ContactList::getSubsetExcluding(ContactList* excludeList)
{
   int i, j;
    ContactList* newList = new ContactList();
    for (i=0; i<getNumberContacts(); i++)
    {
        bool include = true;
        Contact* contact = getContact(i);
        for (j=0; j<excludeList->getNumberContacts(); j++)
        {
            if (*contact == *excludeList->getContact(j))
            {
                include = false;
                break;
            }
        }
        
        // If this contact should be included, add it to the list.
        if (include)
        {
            newList->addContact(new Contact(*contact));
        }
    }
    
    return newList;
}
