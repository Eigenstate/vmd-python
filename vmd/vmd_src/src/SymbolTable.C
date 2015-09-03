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
 *	$RCSfile: SymbolTable.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.60 $	$Date: 2010/12/16 04:08:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Stores the functions available to get info from a molecule
 *  Calls the atom selection parser to create a parse tree
 *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "SymbolTable.h"
#include "AtomSel.h"
#include "Inform.h"
#include "ParseTree.h"

SymbolTable::~SymbolTable(void) {
  int num, i;

  num = fctns.num();
  for (i=0; i<num; i++)
    delete fctns.data(i);

  num = custom_singlewords.num();
  for (i=0; i<num; i++) 
    delete [] custom_singlewords.data(i);
}


// is the given element changable?  Returns TRUE on yes, FALSE on no
int SymbolTable::is_changeable( int fctnidx) {
  if (!(fctnidx >= 0 && fctnidx < fctns.num()))
    return FALSE;

  SymbolTableElement *fctn=fctns.data(fctnidx);
  if (fctn->set_fctn)
    return TRUE;
  
  return FALSE;
}

int SymbolTable::num_custom_singleword() {
  return custom_singlewords.num();
}

const char *SymbolTable::custom_singleword_name(int i) {
  return custom_singlewords.name(i);
}

int SymbolTable::add_custom_singleword(const char *name, const char *macro) {
  // Check if the macro already exists as a "hardwired" singleword.  If so,
  // return failure.
  if (find_attribute(name) >= 0 && custom_singlewords.typecode(name) < 0) {
    msgErr << "Macro '" << name << "' cannot be redefined." << sendmsg;
    return 0;
  }
  // Check for recursion
  ParseTree *tree = parse(macro);
  if (!tree) {
    msgErr << "Macro '" << macro << "' cannot be parsed." << sendmsg;
    return 0;
  }
  if (tree->find_recursion(name)) {
    msgErr << "Macro definition'" << name << "' => '" << macro << "' contains itself." << sendmsg;
    delete tree;
    return 0;
  }
  delete tree;

  // add the name with the given macro.
  // if the macro already exists, overwrite it with the new one
  int ind = custom_singlewords.typecode(name);
  if (ind < 0) {
    ind = custom_singlewords.add_name(name, stringdup(macro));
  } else {
    delete [] custom_singlewords.data(ind);
    custom_singlewords.set_data(ind, stringdup(macro));
  }

  // get cached copy of the name 
  const char *my_name = custom_singlewords.name(ind);
  add_singleword(my_name, atomsel_custom_singleword, NULL);
 
  return 1;
}

const char *SymbolTable::get_custom_singleword(const char *name) {
  int ind = custom_singlewords.typecode(name);
  if (ind < 0) return NULL;
  return custom_singlewords.data(ind);
}

int SymbolTable::remove_custom_singleword(const char *name) {
  // remove from list of custom singlewords by changing the name to ""
  int ind = custom_singlewords.typecode(name);
  if (ind < 0) return 0;
  custom_singlewords.set_name(ind, "");

  // get the index in the tables of names and functions
  ind = find_attribute(name);
  if (ind < 0) return 0;  // XXX this had better not happen
  fctns.set_name(ind, "");
  return 1;
}

#if defined(_MSC_VER)
extern int yyparse(void);
#else
extern "C" int yyparse();
#endif

ParseTree *SymbolTable::parse(const char *s) {
  char *temps = strdup(s);
  atomparser_yystring = temps;
  atomparser_symbols = this;
  yyparse();
  free(temps);
  if (atomparser_result)
    return new ParseTree(this, atomparser_result);
  return NULL;
}

