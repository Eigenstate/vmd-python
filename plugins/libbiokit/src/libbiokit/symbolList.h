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

#ifndef SYMBOLLIST_H
#define SYMBOLLIST_H

class Symbol;
class Alphabet;

class SymbolList {
  
    public:
    SymbolList(Alphabet* alphabet, int initialSize=64);
    virtual ~SymbolList();
    
    virtual bool operator==(SymbolList* compareTo);
    virtual bool operator!=(SymbolList* compareTo);
    virtual bool nongapSymbolsEqual(SymbolList* compareTo);
    virtual int getSize();
    virtual void add(char c);
    virtual void addAll(const char* chars);
    virtual void add(const char* c);
    virtual void add(Symbol& symbol);
    virtual Symbol& get(int index);
    virtual Alphabet* getAlphabet();
    virtual void optimize();

    SymbolList(const SymbolList &s);


    protected:
    virtual void addSymbolIndex(int symbolIndex);
    Alphabet* alphabet;
    
    private:    
    int size;
    int maxSize;
    unsigned char *data;
    
    friend class SymbolListTest;
};

#endif
