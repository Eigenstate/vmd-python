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
 *	$RCSfile: SymbolTable.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.59 $	$Date: 2010/12/16 04:08:41 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Stores the functions available to get info from a molecule
 *  Calls the atom selection parser to create a parse tree
 *
 ***************************************************************************/
#ifndef SYMBOLTABLE_H
#define SYMBOLTABLE_H

#include <stddef.h>
#include "NameList.h"
#include "Command.h"

class ParseTree;

/// create a new atom selection macro 
class CmdAddAtomSelMacro : public Command {
public:
  CmdAddAtomSelMacro(const char *theName, const char *theMacro)
  : Command(ATOMSEL_ADDMACRO) {}
};


/// delete the specified atom selection macro 
class CmdDelAtomSelMacro : public Command {
public:
  CmdDelAtomSelMacro(const char *theName)
  : Command(ATOMSEL_DELMACRO) {}
};


/// Typedef for C functions that take a double and return a double
extern "C" {
  typedef double (*c_ddfunc)(double);
}


/// keeps track of the template-dependent mapping from index to member function
class SymbolTableElement {
public:   
  typedef int (*int_fctn)(void *, int, int *, int *);
  typedef int (*double_fctn)(void *, int, double *, int *);
  typedef int (*string_fctn)(void *, int, const char **, int *);
  typedef int (*single_fctn)(void *, int, int *);
  typedef int (*stringfctn_fctn)(void *, int, const char **, int *, int, int *);

  typedef void (*void_fctn)(void);
  typedef int (*set_int_fctn)(void *, int, int *, int *);
  typedef int (*set_double_fctn)(void *, int, double *, int *);
  typedef int (*set_string_fctn)(void *, int, const char **, int *);
  typedef int (*set_single_fctn)(void *, int, int *, int *);
 
  enum symtype {IS_INT, IS_FLOAT, IS_STRING};
  enum symdesc {NOTHING, KEYWORD, FUNCTION, SINGLEWORD, STRINGFCTN};

  symdesc is_a;
  symtype returns_a;

  /// these acccess (extract) the data
  union {
    c_ddfunc fctn;     
    int_fctn keyword_int;
    double_fctn keyword_double;
    string_fctn keyword_string;
    single_fctn keyword_single;
    stringfctn_fctn keyword_stringfctn;
  };

  /// these set the data -- note functions and string functions are not included
  union {
    void_fctn set_fctn; // set this to NULL if there is nothing else
    set_int_fctn set_keyword_int;
    set_double_fctn set_keyword_double;
    set_string_fctn set_keyword_string;
    set_single_fctn set_keyword_single;
  };

  SymbolTableElement() // need for use in a NameList
  : is_a(NOTHING), fctn(NULL), set_fctn(NULL) {}
   
  SymbolTableElement(c_ddfunc get) 
  : is_a(FUNCTION), returns_a(IS_FLOAT), 
    fctn(get), set_fctn(NULL) {}

  SymbolTableElement(int_fctn get, set_int_fctn set) 
  : is_a(KEYWORD), returns_a(IS_INT), 
    keyword_int(get), set_keyword_int(set) {}

  SymbolTableElement(double_fctn get, set_double_fctn set) 
  : is_a(KEYWORD), returns_a(IS_FLOAT), 
    keyword_double(get), set_keyword_double(set) {}

  SymbolTableElement(string_fctn get, set_string_fctn set)
  : is_a(KEYWORD), returns_a(IS_STRING),
    keyword_string(get), set_keyword_string(set) {}

  SymbolTableElement(stringfctn_fctn get) 
  : is_a(STRINGFCTN), returns_a(IS_STRING),
    keyword_stringfctn(get), set_fctn(NULL) {}

  SymbolTableElement(single_fctn get, set_single_fctn set) 
  : is_a(SINGLEWORD), returns_a(IS_INT),
    keyword_single(get), set_keyword_single(set) {}
};


/// tracks names and functions needed to parse a selection for the given class
class SymbolTable {
private:
  /// list of singlewords that have been added by the user
  NameList<char *> custom_singlewords;

public:
  NameList<SymbolTableElement *> fctns;

  SymbolTable(void) {};
  ~SymbolTable(void);
  
  /// parse selection text, return a new ParseTree on success, NULL on error
  ParseTree *parse(const char *seltext);

  // 
  // add functions and keywords ...
  //
  void add_keyword(const char *visible,
          SymbolTableElement::int_fctn get,
          SymbolTableElement::set_int_fctn set) {
    fctns.add_name(visible, new SymbolTableElement(get, set));
  }
  void add_keyword(const char *visible,
          SymbolTableElement::double_fctn get,
          SymbolTableElement::set_double_fctn set) {
    fctns.add_name(visible, new SymbolTableElement(get, set));
  }
  void add_keyword(const char *visible,
          SymbolTableElement::string_fctn get,
          SymbolTableElement::set_string_fctn set) {
    fctns.add_name(visible, new SymbolTableElement(get, set));
  }
  void add_singleword(const char *visible,
         SymbolTableElement::single_fctn get,
          SymbolTableElement::set_single_fctn set) {
    fctns.add_name(visible, new SymbolTableElement(get, set));
  }
  void add_stringfctn(const char *visible,
          SymbolTableElement::stringfctn_fctn get) { 
    fctns.add_name(visible, new SymbolTableElement(get));
  }

  /// Register C functions that take a double and return a double as
  /// atom selection functions.
  /// Note: These functions must return the same output for given input.
  ///       Functions like rand() will break some optimizations in the 
  ///       atom selection code otherwise.
  void add_cfunction(const char *visible, c_ddfunc fctn) {
    fctns.add_name(visible, new SymbolTableElement(fctn));
  }

  /// find keyword/function matching the name and return function index, or -1
  int find_attribute(const char *attrib) {
    return fctns.typecode(attrib);
  }

  /// returns '1' if the variable can be changed/ modified
  int is_changeable(int fctnidx);

  /// Query what custom singlewords have been defined.  Note: these include
  /// those have been deleted, because the implementation of remove_custom
  /// only sets the string to "".  Text API's will want to only return the
  /// strings that have nonzero length.
  int num_custom_singleword();

  /// Return name with given index, or NULL if not defined.
  const char *custom_singleword_name(int);

  /// add a new singleword macro
  int add_custom_singleword(const char *name, const char *macro);

  /// get the macro for the given singleword
  const char *get_custom_singleword(const char *name);

  /// delete the given singleword macro
  int remove_custom_singleword(const char *name);
};

#endif

