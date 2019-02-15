/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: AtomParser.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.30 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Basic data types for the yacc/lex interface and for the parse tree
 *
 ***************************************************************************/
#ifndef ATOMPARSER_H
#define ATOMPARSER_H

#include "JString.h"
#include "ptrstack.h"

// the idea about strings is that some can be regexed.
//  "This" is a double-quoted string  -- can do regex
//  'this' is a single-quoted string  -- don't do regex
//   this  is a raw, or unquoted, string -- don't do regex
enum  atomparser_stringtype {DQ_STRING, SQ_STRING, RAW_STRING};

/// stores an atom parser string with its string type enumeration
typedef struct atomparser_string {
  atomparser_stringtype st;
  JString s;
} atomparser_string;

/// Each node of the parse tree contains all data needed for that description
typedef struct atomparser_node {
   int node_type;   ///< these are token types, e.g. 'AND', 'WITHIN', 
                    ///< defined in AtomParser.y/y.tab.h
   int extra_type;  ///< for weird things like distinguishing
                    ///< 'index 5 to 7' from 'index 5 7'
   double dval;     ///< floating point value (if any)
   int ival;        ///< integer value (if any)
   atomparser_string sele;  ///< if this is a string, what kind of string?
   atomparser_node *left;   ///< link to left branch of parse tree
   atomparser_node *right;  ///< link to right branch of parse tree

   /// constructor
   atomparser_node(int nnode_t, int nextra_t = -1) {
      node_type = nnode_t;
      extra_type = nextra_t;
      left = NULL;
      right = NULL;
   }

   /// destructor
#if 1
   /// XXX The original recursive implementation will fail on massive 
   ///     selection strings, e.g. a list of 500,000 residue names 
   ///     can blow the stack:
   ///
   ///     for { set i 0 } { $i < 500000 } { incr i } {
   ///       lappend long $i
   ///     }
   ///     atomselect macro huge "index $long"
   ///
   ~atomparser_node(void) {  // destructor
      if (left)
        delete left;
      left=NULL;

      if (right) 
        delete right;
      right=NULL;
   }
#elif 0
   /// Iterative implementation with self-managed stack,
   /// based on a depth-first traversal of the parse tree child nodes
   ~atomparser_node(void) {  // destructor
      if (left == NULL && right == NULL) 
        return;

      PtrStackHandle s = ptrstack_create(128);
      atomparser_node *tnode = NULL;

      if (left) {
        ptrstack_push(s, left);
        left=NULL;
      }

      if (right) {
        ptrstack_push(s, right);
        right=NULL;
      }

      // depth-first traversal deleting nodes
      while (!ptrstack_pop(s, (void **) &tnode)) {
        if (tnode->left) {
          ptrstack_push(s, (void *) tnode->left);
          tnode->left=NULL;
        }
  
        if (tnode->right) {
          ptrstack_push(s, (void *) tnode->right);
          tnode->right=NULL;
        }

        // delete the node once the child nodes have been recorded
        delete tnode;
      }

      ptrstack_destroy(s);
   }
#elif 0
   /// Iterative implementation with self-managed stack,
   /// based on a breadth-first traversal of the parse tree child nodes
   ~atomparser_node(void) {  // destructor
      if (left == NULL && right == NULL) 
        return;

      PtrStackHandle so = ptrstack_create(128);
      PtrStackHandle sn = ptrstack_create(128);
      atomparser_node *tnode = NULL;

      if (left) {
        ptrstack_push(so, left);
        left=NULL;
      }

      if (right) {
        ptrstack_push(so, right);
        right=NULL;
      }

      // breadth-first traversal deleting nodes
      while (!ptrstack_empty(so)) {
        while (!ptrstack_pop(so, (void **) &tnode)) {
          if (tnode->left) {
            ptrstack_push(sn, (void *) tnode->left);
            tnode->left=NULL;
          }
  
          if (tnode->right) {
            ptrstack_push(sn, (void *) tnode->right);
            tnode->right=NULL;
          }

          // delete the node once the child nodes have been recorded
          delete tnode;
        }

        // swap old and new stacks
        PtrStackHandle stmp = so;
        so = sn;
        sn = stmp; 
      }

      ptrstack_destroy(so);
      ptrstack_destroy(sn);
   }
#endif
} atomparser_node;

/// contains the final parse tree, or NULL if there was an error
extern atomparser_node *atomparser_result;

/// given the string and its length, return the index in the symbol table
/// or -1 if it isn't there
int atomparser_yylookup(const char *s, int len);

/// contains the location of the string to parse
extern char *atomparser_yystring;

/// contains the list of the functions and keywords
extern class SymbolTable *atomparser_symbols;

#endif

