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
#include <stdio.h>
#include <string.h>
#include "symbol.h"
#include "alphabet.h"

/* ------------------------------------------------------------------- */
Alphabet::Alphabet(int numSymb, Symbol* symb, int gapSymbolIndex, int unknownSymbolIndex)
: numberSymbols(numSymb), symbols(NULL), gapSymbolIndex(gapSymbolIndex), unknownSymbolIndex(unknownSymbolIndex)
{
   int i;
    symbols = new Symbol[numberSymbols];
    for (i=0; i<numberSymbols; i++)
    {
        symbols[i] = symb[i];
    }
}

/* ------------------------------------------------------------------- */
Alphabet::~Alphabet()
{
    if (symbols != NULL)
    {
        delete[] symbols;
        symbols = NULL;
    }
}

/* ------------------------------------------------------------------- */
int Alphabet::getNumberSymbols()
{
    return numberSymbols;
}

/* ------------------------------------------------------------------- */
int Alphabet::getSymbolIndex(char c)
{
   int i;
    for (i=0; i<numberSymbols; i++)
        if (c == symbols[i].getOne())
            return i;
    
    return unknownSymbolIndex;
}

/* ------------------------------------------------------------------- */
int Alphabet::getSymbolIndex(const char* chars)
{
   int i;
    if (strlen(chars) == 1)
        for (i=0; i<numberSymbols; i++)
            if (chars[0] == symbols[i].getOne())
                return i;
            
    if (strlen(chars) <= 3)
        for (i=0; i<numberSymbols; i++)
            if (strcmp(symbols[i].getThree(), chars) == 0)
                return i;
    
    for (i=0; i<numberSymbols; i++)
        if (strcmp(symbols[i].getFull(), chars) == 0)
            return i;
    
    return unknownSymbolIndex;
}

/* ------------------------------------------------------------------- */
int Alphabet::getSymbolIndex(Symbol& symbol)
{
   int i;
    for (i=0; i<numberSymbols; i++)
        if (symbol == symbols[i])
            return i;

    return unknownSymbolIndex;
}

/* ------------------------------------------------------------------- */
Symbol& Alphabet::getSymbol(int index)
{
    if (index < numberSymbols)
        return symbols[index];
    return getUnknown();
}

/* ------------------------------------------------------------------- */
Symbol& Alphabet::getGap()
{
    return symbols[gapSymbolIndex];
}

/* ------------------------------------------------------------------- */
Symbol& Alphabet::getUnknown()
{
    return symbols[unknownSymbolIndex];
}

/* ------------------------------------------------------------------- */
bool Alphabet::isGap(Symbol& symbol)
{
    return symbol == symbols[gapSymbolIndex];
}

/* ------------------------------------------------------------------- */
bool Alphabet::isUnknown(Symbol& symbol)
{
    return symbol == symbols[unknownSymbolIndex];
}

/* ------------------------------------------------------------------- */
bool Alphabet::hasSymbol(Symbol& symbol)
{
   int i;
    for (i=0; i<numberSymbols; i++)
        if (symbol == symbols[i])
            return true;

    return false;
}

/* ------------------------------------------------------------------- */
char* Alphabet::toString() {
  
  //printf("Number symbols: %d\n",numberSymbols);
  char* str = new char[(numberSymbols*2)+1];
  int i=0;
  for (; i<numberSymbols; i++) {
    str[i*2] = getSymbol(i).getOne();
    str[(i*2)+1] = ' ';
  }
  //i--;
  str[i*2] = '\0';

  return str;
}

/* ------------------------------------------------------------------- */



