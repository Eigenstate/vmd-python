/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: AtomSel.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.54 $      $Date: 2011/06/14 21:12:21 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for selecting atoms.
 *
 ***************************************************************************/
#ifndef ATOMSEL_H
#define ATOMSEL_H

class MoleculeList;
class DrawMolecule;
class ParseTree;
class SymbolTable;
class Timestep;

extern void atomSelParser_init(SymbolTable *);

/// Context which must be passd to xxx_keyword_info functions in SymbolTable.
struct atomsel_ctxt {
  SymbolTable *table;
  DrawMolecule *atom_sel_mol;
  int which_frame;
  const char *singleword;
  atomsel_ctxt(SymbolTable *s, DrawMolecule *d, int frame, const char *word)
  : table(s), atom_sel_mol(d), which_frame(frame), singleword(word) {}
};
 
/// This interacts with the AtomParser
class AtomSel {
private:
  ParseTree *tree;                 ///< this is the result of a selection

  // prevent use of these methods.
  AtomSel& operator=(const AtomSel &) { return *this; }
  AtomSel(AtomSel &) : ID(-1) {}
  const int ID;
  SymbolTable *table;              ///< presumably atomSelParser

public:
  char *cmdStr;                    ///< string with the selection command
  int *on;                         ///< per-atom 'selected' flags; 1=on, 0=off
  int molid() const { return ID; } ///< molid of "my" molecule
  int num_atoms;                   ///< number of atoms in mol
  int selected;                    ///< number of selected atoms the molecule
  int firstsel;                    ///< index of first selected atom
  int lastsel;                     ///< index of final selected atom
 
  enum {TS_LAST = -2, TS_NOW = -1};
  int which_frame;
  int do_update;

  AtomSel(SymbolTable *, int mymolid);
  ~AtomSel();

  /// return codes for the change() method.
  /// NO_PARSE if the string is not correct
  /// NO_EVAL if there was some problem finding the atoms in the selection
  /// otherwise PARSE_SUCCESS.  NO_EVAL will still allow the atom selection
  /// to be used, although warning messages will be printed; whether this is
  /// the best semantics is questionable but since I don't know how to
  /// produce this kind of error we may as well let it slide.
  enum {NO_PARSE = -1, NO_EVAL=-2, PARSE_SUCCESS = 0};
  
  /// provide new settings; does a 'find' at the end if a mol has
  /// been previously provided and returns the results
  /// for the given molecule, find atoms for the molecule.  Stores the indices
  /// in 'on' for quick retrieval later.
  /// If newcmd is NULL, use existing atom selection text
  /// return one of the above enum's.
  int change(const char *newcmd, /* const */ DrawMolecule *);

  /// get the current coordinates (or NULL if it doesn't exist/ no molecule)
  /// FIXME: This is just a wrapper around timestep() and should be eliminated
  float *coordinates(MoleculeList *) const;

  /// get the current timestep (or NULL if it doesn't exist/ no molecule)
  Timestep *timestep(MoleculeList *) const;

  /// given a string ("first", "last", "now", or a value)
  /// return the timestep value in *val
  /// returns -1 if there was a problem
  /// on error, if val > 0, the value of s wasn't understood
  ///           if val < 0, the value was negative
  static int get_frame_value(const char *s, int *val);
};

/// global function to use for custom singlewords
int atomsel_custom_singleword(void *v, int num, int *flgs);

#endif

