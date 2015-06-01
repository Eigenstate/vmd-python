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

#include "symbol.h"
#include "alphabet.h"
#include "alphabetBuilder.h"

Alphabet* AlphabetBuilder::createDnaAlphabet()
{
    const int length = 11;
    Symbol list[length];
    list[0]  = Symbol('A', "ADE", "Adenine");
    list[1]  = Symbol('C', "CYT", "Cytosine");
    list[2]  = Symbol('G', "GUA", "Guanine");
    list[3]  = Symbol('T', "THY", "Thymine");
    list[4]  = Symbol('X', "PUR", "Purine");
    list[5]  = Symbol('Y', "PYR", "Pyrimidine");
    list[6]  = Symbol('N', "NUC", "Unknown Nucleotide");
    list[7]  = Symbol('-', "-",   "Gap");
    list[8]  = Symbol('~', "~",   "Tilde Gap");
    list[9]  = Symbol('?', "?",   "Unknown");
    list[10] = Symbol('.', ".",   "Gap");
    return new Alphabet(length, list, 7, 9);
}

Alphabet* AlphabetBuilder::createRnaAlphabet()
{
    const int length = 11;
    Symbol list[length];
    list[0]  = Symbol('A', "ADE", "Adenine");
    list[1]  = Symbol('C', "CYT", "Cytosine");
    list[2]  = Symbol('G', "GUA", "Guanine");
    list[3]  = Symbol('U', "URA", "Uracil");
    list[4]  = Symbol('X', "PUR", "Purine");
    list[5]  = Symbol('Y', "PYR", "Pyrimidine");
    list[6]  = Symbol('N', "NUC", "Unknown Nucleotide");
    list[7]  = Symbol('-', "-",   "Gap");
    list[8]  = Symbol('~', "~",   "Tilde Gap");
    list[9]  = Symbol('?', "?",   "Unknown");
    list[10] = Symbol('.', ".",   "Gap");
    return new Alphabet(length, list, 7, 9);
}

Alphabet* AlphabetBuilder::createProteinAlphabet() {
    const int length = 28;
    Symbol list[length];
    list[0]  = Symbol('A', "ALA", "Alanine");
    list[1]  = Symbol('R', "ARG", "Arginine");
    list[2]  = Symbol('N', "ASN", "Aspartine");
    list[3]  = Symbol('D', "ASP", "Aspartate");
    list[4]  = Symbol('C', "CYS", "Cysteine");
    list[5]  = Symbol('Q', "GLN", "Glutamine");
    list[6]  = Symbol('E', "GLU", "Glutamate");
    list[7]  = Symbol('G', "GLY", "Glycine");
    list[8]  = Symbol('H', "HIS", "Histidine");
    list[9]  = Symbol('I', "ILE", "Isoleucine");
    list[10] = Symbol('L', "LEU", "Leucine");
    list[11] = Symbol('K', "LYS", "Lysine");
    list[12] = Symbol('M', "MET", "Methionine");
    list[13] = Symbol('F', "PHE", "Phenylalanine");
    list[14] = Symbol('P', "PRO", "Proline");
    list[15] = Symbol('S', "SER", "Serine");
    list[16] = Symbol('T', "THR", "Threonine");
    list[17] = Symbol('W', "TRP", "Tryptophan");
    list[18] = Symbol('Y', "TYR", "Tyrosine");
    list[19] = Symbol('V', "VAL", "Valine");
    list[20] = Symbol('B', "D/N", "Asp or Asn");
    list[21] = Symbol('Z', "E/Q", "Glu or Gln");
    list[22] = Symbol('X', "???", "Unknown");
    list[23] = Symbol('H', "HSD", "Histidine");
    list[24] = Symbol('H', "HSE", "Histidine");
    list[25] = Symbol('H', "HSP", "Histidine");
    list[26] = Symbol('-', "-",   "Gap");
    list[27] = Symbol('?', "?",   "Unknown");
    return new Alphabet(length, list, 26, 27);
}
