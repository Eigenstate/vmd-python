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
 *	$RCSfile: ParseTree.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.43 $	$Date: 2011/03/12 17:25:54 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Basic data types for the yacc/lex interface and for the parse tree
 *
 ***************************************************************************/
#ifndef PARSETREE_H
#define PARSETREE_H

#include "SymbolTable.h"
#include "AtomParser.h"

/// Simplifies the use of three basic data types in an array situation.
/// It does the conversion as needed and can be told to change size
class symbol_data {
 private:
   void make_space(void); ///< de/reallocate an array of active symbol type
   void free_space(void); ///< deallocate array of active symbol type

   /// flag for whether sval strings need to be freed.  This only happens when
   /// they have been converted from int or double; otherwise they are just
   /// copies of pointers and should not be freed.
   int free_sval;

 public:
   SymbolTableElement::symtype type;
   double *dval; ///< array of floating point numbers
   int *ival;    ///< array of integer numbers
   char **sval;  ///< array of strings
   int num;      ///< number of array elements

   symbol_data(SymbolTableElement::symtype new_type, int new_num);
   ~symbol_data(void);
   void convert(SymbolTableElement::symtype totype);
};


/// An atom selection expression parse tree
class ParseTree {
private:
   SymbolTable *table;
   atomparser_node *tree;
   int *selected_array;    ///< this are returned via evaluate
   int num_selected;
   void *context;

public:
   ParseTree(/*const*/ SymbolTable *, atomparser_node *);
   ~ParseTree(void);
   void use_context(void *ctxt) { context = ctxt; }
   int evaluate(int num_atoms, int *flgs);  // sets an array of flags and returns 1, return 0 if bad
   int find_recursion(const char *head);

private:
   void eval_compare(atomparser_node *node, int num, int *flgs);
   symbol_data *eval_mathop(atomparser_node *node, int num, int *flgs);
   symbol_data *eval_key( atomparser_node *node, int num, int *flgs);
   void eval_stringfctn( atomparser_node *node, int num, int *flgs);
   void eval_within(atomparser_node *node, int num, int *flgs);
   void eval_exwithin(atomparser_node *node, int num, int *flgs);
   void eval_pbwithin(atomparser_node *node, int num, int *flgs);
   void eval_single(atomparser_node *node, int num, int *flgs);
   void eval_same(atomparser_node *node, int num, int *flgs);
   void eval_within_bonds(atomparser_node *node, int num, int *flgs);
   void eval_k_nearest(atomparser_node *node, int num, int *flgs);
   void find_rings(int num, int *flgs, int *others, int minringsize, int maxringsize);
   void eval_maxringsize(atomparser_node *node, int num, int *flgs);
   void eval_ringsize(atomparser_node *node, int num, int *flgs);
   symbol_data *eval(atomparser_node *node, int num, int *flgs);
   void eval_find_recursion(atomparser_node *, int *, hash_t *);
};

#endif

