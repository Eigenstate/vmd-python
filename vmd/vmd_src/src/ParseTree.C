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
 *  $RCSfile: ParseTree.C,v $
 *  $Author: johns $		$Locker:  $		$State: Exp $
 *  $Revision: 1.150 $		$Date: 2019/02/07 21:50:48 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Given the parse tree created by a SymbolTable, evaluate it and return
 * the selection
 *
 ***************************************************************************/

#define NOMINMAX 1         // prevent MSVS from defining min/max macros

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>          // for pow, fabs

#include "AtomParser.h"    // for atomparser_node definition
#include "y.tab.h"         // for FLOATVAL, INTVAL, and STRWORD
#include "ParseTree.h"
#include "Inform.h"        // for printing error messages
#include "JRegex.h"        // for regular expression matching
#include "AtomSel.h"       // for atomsel_ctxt definition
#include "Timestep.h"      // for accessing coordinate data
#include "DrawMolecule.h"  // for drawmolecule multiple-frame selections
#include "SpatialSearch.h" // for find_within()

#include <vector>          // for knearest implementation
#include <algorithm>       // for knearest implementation
// #include <limits>         // for knearest implementation

// do the string and numeric compares
#define case_compare_numeric_macro(switchcase, symbol)	\
  case switchcase:					\
    l->convert(SymbolTableElement::IS_FLOAT);		\
    r->convert(SymbolTableElement::IS_FLOAT);		\
    ldval = l->dval;					\
    rdval = r->dval;					\
    flg = flgs;						\
    for (i=num-1; i>=0; i--) {				\
      *flg &= (*ldval symbol *rdval);			\
      ldval += lincr; rdval += rincr; flg++;		\
    }							\
  break;

#define case_compare_string_macro(switchcase, symbol)	\
  case switchcase:					\
    l->convert(SymbolTableElement::IS_STRING);		\
    r->convert(SymbolTableElement::IS_STRING);		\
    lsptr = l->sval;					\
    rsptr = r->sval;					\
    flg = flgs;						\
    for (i=num-1; i>=0; i--) {				\
      if (*flg)						\
        *flg &= (strcmp(*lsptr, *rsptr) symbol 0);	\
      lsptr += lincr; rsptr += rincr; flg++;		\
    }							\
  break;
   

///////////////// the ParseTree
ParseTree::ParseTree(SymbolTable *parser, atomparser_node *parse_tree)
{
  tree = parse_tree;
  table = parser;
  selected_array = NULL;
  num_selected = 0;
  context = NULL;
}

ParseTree::~ParseTree(void) {
  if (selected_array != NULL) 
    delete [] selected_array;
  delete tree;
}

void ParseTree::eval_compare(atomparser_node *node, int num, int *flgs) {
  int i;
  double *ldval, *rdval;
  char **lsptr, **rsptr;
  int lincr, rincr;
  int *flg;

  // get the data on the left and right
  symbol_data *l = eval(node->left, num, flgs);
  symbol_data *r = eval(node->right, num, flgs);

  // If the symbol data contains num elements, we need to check each one.
  // Otherwise, it contains exactly one element and we can just keep
  // reusing it.  
  lincr = l->num == num ? 1 : 0;
  rincr = r->num == num ? 1 : 0;

  switch (node->ival) {
    case_compare_numeric_macro(NLT, <  )
    case_compare_numeric_macro(NLE, <= )
    case_compare_numeric_macro(NEQ, == )
    case_compare_numeric_macro(NGE, >= )
    case_compare_numeric_macro(NGT, >  )
    case_compare_numeric_macro(NNE, != )

    case_compare_string_macro(SLT, <  )
    case_compare_string_macro(SLE, <= )
    case_compare_string_macro(SEQ, == )
    case_compare_string_macro(SGE, >= )
    case_compare_string_macro(SGT, >  )
    case_compare_string_macro(SNE, != )

    case MATCH: {
      l->convert(SymbolTableElement::IS_STRING);
      r->convert(SymbolTableElement::IS_STRING);
      lsptr = l->sval;
      rsptr = r->sval;
      flg = flgs;
      JRegex *rgx = NULL;
      const char *first = *rsptr;

      for (i=0; i<num; i++) {
        if (i==0 || strcmp(*rsptr, first)) {
          if (rgx) 
            delete rgx;
          rgx = new JRegex(*rsptr);
          first = *rsptr;
        }
        if (rgx) {
          if (*flg)
            *flg &= (rgx->match(*lsptr, strlen(*lsptr)) != -1);
        } else {
          *flg = 0;
        }
        lsptr += lincr; rsptr += rincr; flg++;
      }
      if (rgx) {
        delete rgx;
      }
      // done with match search
      break;
    }

    default:
      msgWarn << "ParseTree::eval_compare() missing operator!" << sendmsg;
   }

   delete l;
   delete r;
}


// place to do +, -, *, and /
symbol_data * ParseTree::eval_mathop(atomparser_node *node, int num, int *flgs)
{
  symbol_data *l = eval(node->left, num, flgs);
  symbol_data *r = eval(node->right, num, flgs);
  // since we can only have 1 or num elements, we'll either be using the
  // incrementing index value, or we'll only be using dval[0], so we set
  // the lincr/rincr value to 0 or all 1's and do binary AND against it.
  int lincr = l->num == num ? (~0) : 0;
  int rincr = r->num == num ? (~0) : 0;
  l->convert(SymbolTableElement::IS_FLOAT);
  r->convert(SymbolTableElement::IS_FLOAT);
  symbol_data *tmp = new symbol_data(SymbolTableElement::IS_FLOAT, num);
  int i;
  const double *lval = l->dval;
  const double *rval = r->dval;
  double *tmpval = tmp->dval;

  // XXX does it really pay to have tests on the flgs[] array
  //     in loops that just do addition/subtraction?  If the resulting
  //     values are never referenced, we might get better performance 
  //     by doing the math regardless, for these simple cases.  For fmod()
  //     it is definitely beneficial to performance to test before calling...
  int firstsel = 0;
  int lastsel = -1;
  if (!find_first_selection_aligned(num, flgs, &firstsel) &&
      !find_last_selection_aligned(num, flgs, &lastsel)) {
    // XXX we should trim the loop ranges to the last selection prior to
    //     entering the switch cases...
    switch (node->node_type) {
      case ADD:
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = lval[lincr & i] + rval[rincr & i];
        }
        break;
      case SUB:
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = lval[lincr & i] - rval[rincr & i];
        }
        break;
      case MULT:
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = lval[lincr & i] * rval[rincr & i];
        }
        break;
      case DIV:
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = lval[lincr & i] / rval[rincr & i];
        }
        break;
      case MOD:  // fake mod
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = fmod(lval[lincr & i], rval[rincr & i]);
        }
        break;
      case EXP:
        for (i=firstsel; i<=lastsel; i++) {
          if (flgs[i]) tmpval[i] = pow(lval[lincr & i], rval[rincr & i]);
        }
        break;
    }
  }
     
  delete l;
  delete r;
  return tmp;
}


// This puts the task of doing the selection inside the function
// For example: sequence APW "T.*A"
// The function converts the linked list into an array of const char *
// and of fields,  0 == raw, 1 == single quote, 2 == double quote
// if this is the start of a "to" then the fields are
// and of fields,  3 == raw, 4 == single quote, 5 == double quote
// the function modifies the selection as it pleases, 
// so it _can_ override the current flags.  Please be careful.
void ParseTree::eval_stringfctn(atomparser_node *node, int num, int *flgs) {
  int count = 0;
  atomparser_node *left;

  // first count the number of elements in the linked list
  for (left = node->left; left != NULL; left = left->left) {
    count++;
  }
  if (count == 0) 
    return;

  // now populate the char ** pointers
  char **argv= (char **) calloc(1, count * sizeof(char *));
  int *types = new int[count];
  int i=0;
  for (left = node->left; left != NULL; left = left -> left, i++) {
    // get the string type (single quote, double quote, raw)
    switch (left->sele.st) {
      case RAW_STRING:
        types[i] = 0;
        argv[i] = (char *) ((const char *) left->sele.s);
        break;

      case SQ_STRING:
        types[i] = 1;
        argv[i] = (char *) ((const char *) left->sele.s);
        break;

      case DQ_STRING: 
        types[i] = 2;
        argv[i] = (char *) ((const char *) left->sele.s);
        break;
    }

    if (left->extra_type != -1) { // then it is a "through" search
      types[i] += 3;
    }
  }

  // Call the function. Functions can override flags, so they are copied first
  int *tmp_flgs = new int[num];
  memcpy(tmp_flgs, flgs, num * sizeof(int));
  SymbolTableElement *elem = table->fctns.data(node->extra_type);
  elem->keyword_stringfctn(context, count, (const char **)argv, types, num, tmp_flgs);

  // XXX candidate for a nice SSE loop
  for (i = num-1; i>=0; i--) {
    if (flgs[i]) flgs[i] = tmp_flgs[i];
  }
  delete [] tmp_flgs;
  delete [] types;
  free(argv);
}

static void same_string(symbol_data *tmp, symbol_data *tmp2, int num, 
                        int *subselect, int *flgs) {
  hash_t hash;
  hash_init(&hash, num);

  // Hash all entries in the sublist
  int i;
  for (i=0; i<num; i++)
    if (subselect[i])
      hash_insert(&hash, tmp2->sval[i], 0);

  // Turn on flgs only if it's already on and its value is in the table.
  // Note: We cannot access string data for items that aren't on.
  //       This is also much faster than calling hash_lookup() unnecessarily.
  for (i=0; i<num; i++) 
    if (flgs[i])
      flgs[i] = (hash_lookup(&hash, tmp->sval[i]) != HASH_FAIL);

  hash_destroy(&hash);
}

static void same_int(symbol_data *tmp, symbol_data *tmp2, int num, 
                        int *subselect, int *flgs) {
  int firstsubsel = -1;
  if (find_first_selection_aligned(num, subselect, &firstsubsel)) {
    // subselection is empty, so set all flags to zero.
    memset(flgs, 0, num*sizeof(int));
    return;
  }

  int lastsubsel = -1;
  if (find_last_selection_aligned(num, subselect, &lastsubsel)) {
    // subselection is empty, but this should never happen since we
    // already checked for a first selection!
    msgErr << "internal inconsistency in first/last selection search results"
           << sendmsg;
    //  set all flags to zero and abort.
    memset(flgs, 0, num*sizeof(int));
    return;
  }

  // Create a table of values found in subselect
  // XXX This could get to be very large, and therefore slow.
  //     We should consider changing this to a hash table implementation
  //     so that we don't soak up massive amounts of memory for cases where
  //     we have an extremely sparse array of values
  int *int_table = NULL, int_min, int_max;

  // XXX Could use a minmax_1iv_aligned() SSE vectorized helper routine here...
  int_min = int_max = tmp2->ival[firstsubsel];
  int i;
  for (i=firstsubsel; i<=lastsubsel; i++) {
    if (subselect[i]) {
      int ival = tmp2->ival[i];
      if (ival > int_max)
        int_max = ival; 
      if (ival < int_min)
        int_min = ival; 
    }
  }

  int_table = (int *) calloc(1+int_max-int_min, sizeof(int)); 
  for (i=firstsubsel; i<=lastsubsel; i++) {
    if (subselect[i]) {
      int_table[tmp2->ival[i]-int_min] = 1;
    }
  }

  // Turn on flgs only if it's already on and its value is in the table.
  for (i=0; i<num; i++) {
    if (flgs[i]) {
      int ival = tmp->ival[i];
      if (ival >= int_min && ival <= int_max)
        flgs[i] = int_table[ival-int_min];
      else
        flgs[i] = 0;
    }
  }
  free(int_table);
}

static void same_double(symbol_data *tmp, symbol_data *tmp2, int num, 
                        int *subselect, int *flgs) {
  int firstsubsel, lastsubsel, subselselected;
  if (analyze_selection_aligned(num, subselect,
                                &firstsubsel, &lastsubsel, &subselselected)) {
    return;
  }

  // Hash all the entries in the sublist, then check each flag against the
  // table.  I have to convert doubles to strings.
  hash_t hash;
  // XXX doubles can't be longer than 25 chars, can they?
  char *doublestring = new char[25L*subselselected];
  char *istring = doublestring;
  hash_init(&hash, subselselected);
  int i;
  for (i=firstsubsel; i<=lastsubsel; i++) {
    if (subselect[i]) {
      sprintf(istring,"%f", (double) tmp2->dval[i]); 
      hash_insert(&hash, istring, 0);
      istring += 25;
    }
  }

  char tmpstring[25];
  for (i=0; i<num; i++) {
    sprintf(tmpstring,"%f", (double) tmp->dval[i]);  
    flgs[i] &= (hash_lookup(&hash, tmpstring) != HASH_FAIL);
  }
  hash_destroy(&hash);

  delete [] doublestring;
}


// this does things like: same resname as name CA 
// 1) evalute the expression (m atoms)
// 2) get the keyword information (n atoms)
// 3) do an n*m search for the 'same' values
void ParseTree::eval_same(atomparser_node *node, int num, int *flgs) {
   int i;
   int *subselect = new int[num];
   for (i=0; i<num; i++)
     subselect[i]=1; 

   // 1) evaluate the sub-selection
   if (eval(node->left, num, subselect)) {
     delete [] subselect;
     msgErr << "eval of a 'same' returned data when it shouldn't have" 
            << sendmsg;
     return;
   }

   // at this point, only the sub selection is defined
   // 2) get the keyword information
   // 2a) make space for the return type
   SymbolTableElement *elem = table->fctns.data(node->extra_type);
   SymbolTableElement::symtype has_type = elem->returns_a;
   symbol_data *tmp, *tmp2;
   tmp = new symbol_data(has_type, num);
   tmp2 = new symbol_data(has_type, num);
   
   // 2b) get the data (masked by the info passed by flgs)
   //     and find the 'same' value
   switch (has_type) {
    case SymbolTableElement::IS_INT:   
      elem->keyword_int(context, num, tmp->ival, flgs);
      elem->keyword_int(context, num, tmp2->ival, subselect);
      same_int(tmp, tmp2, num, subselect, flgs);
      break;

    case SymbolTableElement::IS_FLOAT: 
      elem->keyword_double(context, num, tmp->dval, flgs);
      elem->keyword_double(context, num, tmp2->dval, subselect);
      same_double(tmp, tmp2, num, subselect, flgs);
      break;

    case SymbolTableElement::IS_STRING:
      elem->keyword_string(context, num, (const char **)tmp->sval, flgs);
      elem->keyword_string(context, num, (const char **)tmp2->sval, subselect);
      same_string(tmp, tmp2, num, subselect, flgs); 
      break;
   }
   
   delete tmp;
   delete tmp2;
   delete [] subselect;
}


// here's where I get things like: name CA N C O
// and: mass
symbol_data *ParseTree::eval_key(atomparser_node *node, int num, int *flgs) {
  // make space for the return type
  SymbolTableElement *elem = table->fctns.data(node->extra_type);
  SymbolTableElement::symtype has_type = elem->returns_a;
  symbol_data *tmp;
  tmp = new symbol_data(has_type, num);

  switch (has_type) {
    case SymbolTableElement::IS_INT:
      elem->keyword_int(context, num, tmp->ival, flgs);
      break;
    case SymbolTableElement::IS_FLOAT:
      elem->keyword_double(context, num, tmp->dval, flgs);
      break;
    case SymbolTableElement::IS_STRING:
      elem->keyword_string(context, num, (const char **)tmp->sval, flgs);
      break;
  }

  // If we're doing int's, set up a table to store all the values we find
  // in the list of values.  
  int *int_table = NULL;

  // XXX it should be possible for us to move the first/last selection
  // evaluation prior to the elem->keyword_xxx() calls so that we can
  // accelerate the inner loops within those operations too...
  int firstsel = 0;
  int lastsel = -1;
  if (find_first_selection_aligned(num, flgs, &firstsel)) {
    // XXX what do we do if there was no selection?
    //     is there any point in going on from here?
    firstsel=0; // loops that test i<=lastsel will early-exit as they should
  } else {
    if (find_last_selection_aligned(num, flgs, &lastsel)) {
      msgErr << "internal inconsistency in first/last selection search results"
             << sendmsg;
    }
  }
 
  int int_min=0, int_max=0; 
  if (has_type == SymbolTableElement::IS_INT) {
    if (lastsel != -1) {
      int_min = int_max = tmp->ival[firstsel];
    } 
    // XXX what do we do if there was no selection?

    // find min/max values
    // XXX Could use a minmax_1iv_aligned() SSE vectorized helper routine here...
    for (int i=firstsel; i<=lastsel; i++) {
      if (flgs[i]) {
        const int ival = tmp->ival[i];
        if (ival > int_max)
          int_max = ival; 
        if (ival < int_min)
          int_min = ival; 
      }
    }
    int_table = (int *) calloc(1+int_max-int_min, sizeof(int)); 
  }
 
  // Now that I have the data, I can do one of two things
  // Either it is a list, in which case there is data off the
  // left, or it returns the data itself

  // if there is a list coming off the left, then I have
  // name CA N     ===> (name='CA' and name='N')
  // chain 1 to 3  ===> (name>='1' and name<='3'

  // XXX call calloc() instead, and avoid extra clear operation
  int *newflgs = new int[num];
  // have to do this since selection parameters are 'OR'ed together
  memset(newflgs, 0, num*sizeof(int));

  if (node->left) {
    atomparser_node *left = node->left;
    while (left) {
      if (left->extra_type == -1) { 
        // then it is normal
        switch(has_type) {
          case SymbolTableElement::IS_INT:
            {
              int ival = atoi(left->sele.s);
              if (ival >= int_min && ival <= int_max) 
                int_table[ival-int_min] = 1;
            }
            break;

          case SymbolTableElement::IS_FLOAT:
            {
              // select atoms that are within .1% of dval
              double dval = atof(left->sele.s);
              double delta = fabs(dval / 1000);
              double maxval = dval+delta;
              double minval = dval-delta;
              for (int i=firstsel; i<=lastsel; i++) {
                if (flgs[i]) 
                  newflgs[i] |= (minval <= tmp->dval[i] && maxval >= tmp->dval[i]);
              }
            }
            break;

          case SymbolTableElement::IS_STRING:
            {
              switch (left->sele.st) {
                case SQ_STRING: // doing string as single quotes
                case RAW_STRING:
                  {
                    for (int i=firstsel; i<=lastsel; i++) {
                      // XXX we get NULL tmp->sval[i] when only coords
                      // are loaded, without any structure/names, so
                      // checking this prevents crashes
                      if (flgs[i] && (tmp->sval[i] != NULL)) {
                        newflgs[i] |= !strcmp(left->sele.s, tmp->sval[i]);
                      }
                    }
                  }
                  break;

                case DQ_STRING:
                default:
                  {
                    // A regex like "H" would match 'H', 'H21',
                    // 'OH2', etc.  I force the match to be
                    // complete with the ^ and $.  The parenthesis \(\)
                    // are to avoid turning C\|O into ^C\|O$
                    // and the double \ is to escape the string escape
                    // mechanism.  Ain't this grand?
                    JString temps = "^("+left->sele.s+")$";
                    JRegex r(temps, 1);  // 1 for fast compile
                    for (int i=firstsel; i<=lastsel; i++) {
                      if (flgs[i]) {
                        newflgs[i] |= (r.match(tmp->sval[i], strlen(tmp->sval[i])) != -1);
                      }
                    } // end loop
                  } // end check for DQ_STRING
                  break;
              } // end based on string type
            } // end of IS_STRING
        } // end switch based on keyword type
      } else {  // do a 'through' search
        switch(has_type) {
          case SymbolTableElement::IS_INT:
            {
              int ltval = atoi(left->sele.s);
              int gtval = atoi(left->left->sele.s);
              if (ltval < int_min) ltval = int_min;
              if (gtval > int_max) gtval = int_max;
              for (int i=ltval-int_min; i<= gtval-int_min; i++)
                int_table[i] = 1;
            }
            break;
          case SymbolTableElement::IS_FLOAT:
            {
              double ltval = atof(left->sele.s);
              double gtval = atof(left->left->sele.s);
              for (int i=firstsel; i<=lastsel; i++) {
                if (flgs[i])
                   newflgs[i] |= ((ltval <= tmp->dval[i]) && (gtval >= tmp->dval[i]));
              }
            }
            break;
          default:
            {
              // no way to do regex with < or >, so do exact
              for (int i=firstsel; i<=lastsel; i++) {
                if (flgs[i]) 
                  newflgs[i] |= (flgs[i] && strcmp(left->sele.s, tmp->sval[i]) <= 0
                                 && strcmp(left->left->sele.s, tmp->sval[i]) >= 0);

              }
            }
        } // end switch checking type
        left = left->left;  // need to bypass that 2nd one
      } // end both possible ways
      left = left->left;
    } // end while loop going down the left side

    // get the flgs info back together
    if (has_type == SymbolTableElement::IS_INT) {
      for (int i=firstsel; i<=lastsel; i++) {
        if (flgs[i])
          flgs[i] = int_table[tmp->ival[i]-int_min]; 
      }
      free(int_table);
    } else {
      for (int i=firstsel; i<=lastsel; i++) {
        if (flgs[i]) 
          flgs[i] = newflgs[i];
      }
    }
    // first and last selections are now invalidated by the merge op
    firstsel=0;
    lastsel=num-1;

    delete [] newflgs;
    delete tmp;
    return NULL;
  } else {
    // if there isn't a list, then I have something like
    // mass + 5 < 7
    // so just return the data
    delete [] newflgs;
    if (int_table) free(int_table);
    return tmp;
  }
}


void ParseTree::eval_single(atomparser_node *node, int num, int *flgs) {
  // XXX Cast to atomsel_ctxt since we _know_ that only atom selections
  // use singlewords.
  atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
  ctxt->singleword = table->fctns.name(node->extra_type);
  table->fctns.data(node->extra_type)->keyword_single(context, num, flgs);
}


void ParseTree::eval_exwithin(atomparser_node *node, int num, int *flgs) {
  eval_within(node, num, flgs);

  // add "and not others"
  int *others = new int[num];
  int i;
  for (i=0; i<num; others[i++] = 1);
  
  // XXX evaluates node->left twice
  if (eval(node->left, num, others)) {
    delete [] others;
    msgErr << "eval of a 'within' returned data when it shouldn't have." << sendmsg;
    return;
  }
  for (i=0; i<num; i++) {
    if (others[i]) flgs[i] = 0;
  }
  delete [] others;
}

 
// XXX copied from AtomSel
static Timestep *selframe(DrawMolecule *atom_sel_mol, int which_frame) {
  switch (which_frame) {
   case AtomSel::TS_LAST: return atom_sel_mol->get_last_frame(); 
   case AtomSel::TS_NOW : return atom_sel_mol->current(); 
   default: {
     if (!atom_sel_mol->get_frame(which_frame)) {
       return atom_sel_mol->get_last_frame();

     } else {
       return atom_sel_mol->get_frame(which_frame);
     }
   }
  }
  return NULL;
}

void ParseTree::eval_pbwithin(atomparser_node *node, int num, int *flgs) {
  // for a zero valued distance, just return the "others" part
  // with no additional atoms selected.
  if ((float) node->dval <= 0.0f) {
    eval(node->left, num, flgs);
    return; // early exit
  }

  //
  // if we have a non-zero distance criteria, do the computation
  //
  int i;

  // coords holds original coordinates in first 3N entries
  ResizeArray<float> coords(3L*2L*num);

  // others holds the flags for others in the first N entries, and will
  // be padded with zeros, one for each replicated flg atom
  ResizeArray<int> others(2L*num);
  
  // repflgs holds a copy of flgs in the first N entries, and will be
  // extended with ones for each replicated flag atom.  We store the index
  // of the replicated atom in repindexes.
  ResizeArray<int> repflgs(2L*num);

  // repindexes holds the indexes of the replicated flg atoms.
  ResizeArray<int> repindexes(num);

  // fetch coordinates
  atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
  const Timestep *ts = selframe(ctxt->atom_sel_mol, ctxt->which_frame);
  if (!ts) {
    msgErr << "No timestep available for 'within' search!" << sendmsg;
    return;
  }
  others.appendN(1, num);
  if (eval(node->left, num, &others[0])) {
    msgErr << "eval of a 'within' returned data when it shouldn't have." << sendmsg;
    return;
  }

  // fill in start of coords and repflgs.
  const float * pos=ts->pos;
  for (i=0; i<num; i++) {
    coords.append3(pos);
    pos += 3;
    repflgs.append(flgs[i]);
  }

  // find bounding box on others
  float min[3], max[3];
  if (!find_minmax_selected(num, &others[0], ts->pos, 
        min[0], min[1], min[2], max[0], max[1], max[2])) {
    memset(flgs, 0, num*sizeof(int));
    return;
  }

  // extend bounding box by the cutoff distance.
  const float cutoff = (float)node->dval;
  for (i=0; i<3; i++) {
    min[i] -= cutoff;
    max[i] += cutoff;
  }

  // replicate flgs atoms as needed.  
  float A[3], B[3], C[3];
  ts->get_transform_vectors(A, B, C);
  for (i=-1; i<=1; i++) {
    float v1[3];
    vec_scale(v1, (float) i, A);
    for (int j=-1; j<=1; j++) {
      float v2[3];
      vec_scale(v2, (float) j, B);
      for (int k=-1; k<=1; k++) {
        // don't replicate the home cell
        if (!i && !j && !k) continue;
        float v3[3];
        vec_scale(v3, (float) k, C);
        float vx = v1[0] + v2[0] + v3[0];
        float vy = v1[1] + v2[1] + v3[1];
        float vz = v1[2] + v2[2] + v3[2];
        pos = ts->pos;
        for (int ind=0; ind<num; ind++) {
          if (flgs[ind]) {
            const float x = pos[0] + vx;
            const float y = pos[1] + vy;
            const float z = pos[2] + vz;
            if (x>min[0] && x<=max[0] &&
                y>min[1] && y<=max[1] &&
                z>min[2] && z<=max[2]) {
              repindexes.append(ind);
              coords.append3(x, y, z);
              repflgs.append(1);
              others.append(0);
            }
          }
          pos += 3;
        }
      }
    }
  }

  find_within(&coords[0], &repflgs[0], &others[0], others.num(), cutoff);

  // copy the flags for the unreplicated coordinates into the final result
  memcpy(flgs, &repflgs[0], num*sizeof(int));

  // OR the replicated atom flags into the unreplicated set.
  for (i=0; i<repindexes.num(); i++) {
    if (repflgs[num+i]) {
      flgs[repindexes[i]] = 1;
    }
  }
}


void ParseTree::eval_within(atomparser_node *node, int num, int *flgs) {
  // if we have a non-zero distance criteria, do the computation
  if ((float) node->dval > 0.0f) {
    // find the atoms in the rest of the selection
    int *others = new int[num];
    int i;
    for (i=0; i<num; ++i) 
      others[i] = 1;

    if (eval(node->left, num, others)) {
      delete [] others;
      msgErr << "eval of a 'within' returned data when it shouldn't have." << sendmsg;
      return;
    }

    // get the coordinates directly from the molecule.
    atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
    Timestep *ts = selframe(ctxt->atom_sel_mol, ctxt->which_frame);
    if (!ts) {
      msgErr << "No timestep available for 'within' search!" << sendmsg;
      return;
    }

    find_within(ts->pos, flgs, others, num, (float) node->dval);

    delete [] others;
  } else {
    // for a zero valued distance, just return the "others" part
    // with no additional atoms selected.
    eval(node->left, num, flgs);
  }
}


void ParseTree::eval_within_bonds(atomparser_node *node, int num, int *flgs) {
  atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
  int *others = new int[num];

  int i;
  for (i=0; i<num; ++i) 
    others[i] = 1;

  if (eval(node->left, num, others)) {
    delete [] others;
    msgErr << "eval of a 'within' returned data when it shouldn't have." << sendmsg;
    return;
  }

  // copy others to bondedsel
  int *bondedsel = new int[num];
  memcpy(bondedsel, others, num*sizeof(int));

  // grow selection by traversing N bonds...
  int bonddepth;
  for (bonddepth=0; bonddepth<node->ival; bonddepth++) {
    // if an atom is selected, select all of its bond partners
    for (i=0; i<num; i++) {
      if (bondedsel[i]) {
        MolAtom *atom = ctxt->atom_sel_mol->atom(i);
        int j;
        for (j=0; j<atom->bonds; j++) {
          others[atom->bondTo[j]] = 1;
        }
      }        
    }        

    // copy others to bondedsel 
    memcpy(bondedsel, others, num*sizeof(int));
  } 

  for (i=0; i<num; ++i) 
    if (bondedsel[i] && flgs[i]) 
      flgs[i] = 1;
    else
      flgs[i] = 0;

  delete [] bondedsel;
  delete [] others;
}


//
// Find K atoms nearest to a selection
//
// XXX This uses the brute force approach rather than building a K-d tree.
//     This should be fine for small selections, but will not do very well 
//     on large structures.  In reality, the small selections are the common
//     case so this is a low priority item for now.
// 
namespace {
  struct PointDistance {
    float o;
    int i;
    PointDistance() {}
    PointDistance(float o_, int i_) : o(o_), i(i_) {}
    bool operator<(const PointDistance& p) const {
      return o<p.o;
    }
  };
}

void ParseTree::eval_k_nearest(atomparser_node *node, int num, int *flgs) {
  atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
  const Timestep *ts = selframe(ctxt->atom_sel_mol, ctxt->which_frame);
  if (!ts) {
    msgErr << "No timestep available for 'nearest' search!" << sendmsg;
    return;
  }
  const int N = node->ival;
  int i;

  /* evaluate subselection */
  std::vector<int> others(num);
  for (i=0; i<num; ++i) 
    others[i] = 1;

  if (eval(node->left, num, &others[0])) {
    msgErr << "eval of a 'within' returned data when it shouldn't have." << sendmsg;
    return;
  }

  /* make sure we have something in other */
  for (i=0; i<num; i++) {
    if (others[i]) 
      break;
  }
  if (i==num) {
    memset(flgs, 0, num*sizeof(*flgs));
    return;
  }

  std::vector<PointDistance> distances;
  int numdists=0;
  for (i=0; i<num; i++) {
    if (others[i] || !flgs[i]) 
      continue;
#if 1
    float d2=1e37f;
#else
    float d2=std::numeric_limits<float>::max();
#endif
    for (int j=0; j<num; j++) {
      if (!others[j]) 
        continue;

      float d2_j=distance2(ts->pos+3L*i, ts->pos+3L*j);
      if (d2_j<d2) 
        d2=d2_j;
    }

    distances.push_back(PointDistance(d2,i));
    numdists++;
  }

  std::sort(distances.begin(), distances.end());
  int n=N;

  // XXX avoid signed vs. unsigned comparisons
  // if (n>distances.size()) n=distances.size();
  if (n>numdists) 
    n=numdists;
  memset(flgs, 0, num*sizeof(*flgs));
  for (i=0; i<n; i++) {
    flgs[distances[i].i]=1;
  }
}


//
// Find ring structures
//
void ParseTree::find_rings(int num, int *flgs, int *others, 
                           int minringsize, int maxringsize) {
#ifdef VMDWITHCARBS
  int i;

  // XXX We're hijacking the ring list in BaseMolecule at present.
  //     It might be better to build our own independent one, but
  //     this way there's only one ring list in memory at a time.
  atomsel_ctxt *ctxt = (atomsel_ctxt *)context;
  ctxt->atom_sel_mol->find_small_rings_and_links(5, maxringsize);
  SmallRing *ring;
  memset(flgs, 0, num*sizeof(int));
  for (i=0; i < ctxt->atom_sel_mol->smallringList.num(); i++) {
    ring = ctxt->atom_sel_mol->smallringList[i];
    int N = ring->num();
    if (N >= minringsize && N <= maxringsize) {
      int j;
      for (j=0; j<N; j++) {
        int ind = (*ring)[j];
        flgs[ind] = others[ind];
      } 
    } 
  }
#else
  memset(flgs, 0, num*sizeof(int));
#endif 
}


void ParseTree::eval_maxringsize(atomparser_node *node, int num, int *flgs) {
  // find the atoms in the rest of the selection
  int *others = new int[num];
  int i;
  for (i=0; i<num; ++i) 
    others[i] = 1;

  if (eval(node->left, num, others)) {
    delete [] others;
    msgErr << "eval of a 'maxringsize' returned data when it shouldn't have." << sendmsg;
    return;
  }

  find_rings(num, flgs, others, 1, node->ival);

  delete [] others;
}


void ParseTree::eval_ringsize(atomparser_node *node, int num, int *flgs) {
  // find the atoms in the rest of the selection
  int *others = new int[num];
  int i;
  for (i=0; i<num; ++i) 
    others[i] = 1;

  if (eval(node->left, num, others)) {
    delete [] others;
    msgErr << "eval of a 'ringsize' returned data when it shouldn't have." << sendmsg;
    return;
  }

  find_rings(num, flgs, others, node->ival, node->ival);

  delete [] others;
}


// a node of the tree merges symbol_datas
// a leaf of the tree produces symbol_datas
symbol_data *ParseTree::eval(atomparser_node *node, int num, int *flgs) {
  int i;
  int *flg1, *flg2;
  symbol_data *tmp;
  switch(node->node_type) {
    case AND:
      eval(node->left, num, flgs);  // implicit 'and'
      eval(node->right, num, flgs);
      return NULL;

    case NOT:
      flg1 = new int[num];
      memcpy(flg1, flgs, num*sizeof(int));
      // this gives: A and B
      eval(node->left, num, flg1);
      // I want A and (not B)
      for (i=num-1; i>=0; i--) {
        if (flgs[i]) 
          flgs[i] = !flg1[i];
      }
      delete [] flg1;
      break;

    case OR:
      flg1 = new int[num];
      memcpy(flg1, flgs, num*sizeof(int));
      eval(node->left, num, flg1);
      flg2 = new int[num];
      memcpy(flg2, flgs, num*sizeof(int));
      eval(node->right, num, flg2);
      for (i=num-1; i>=0; i--) {
        flgs[i] = flgs[i] && (flg1[i] || flg2[i]);
      }
      delete [] flg1;
      delete [] flg2;
      break;

    case FLOATVAL:
      tmp = new symbol_data(SymbolTableElement::IS_FLOAT, 1);
      tmp->dval[0] = node->dval;
      return tmp;

    case INTVAL:
      tmp = new symbol_data(SymbolTableElement::IS_INT, 1);
      tmp->ival[0] = node->ival;
      return tmp;

    case STRWORD:
      tmp = new symbol_data(SymbolTableElement::IS_STRING, 1);
      tmp->sval[0] = (char *)(const char *)node->sele.s;
      return tmp;

    case KEY: 
      return eval_key(node, num, flgs);

    case STRFCTN: 
      eval_stringfctn(node, num, flgs); 
      break;

    case FUNC:
      {
      // The only functions in the SymbolTable class are C functions 
      // that take a double and return a double.  Hence we don't need
      // handle all 3x3=9 different cases.  
      symbol_data *inp = eval(node->left, num, flgs);
      inp->convert(SymbolTableElement::IS_FLOAT);

      // set up space for the return
      symbol_data *ret = new symbol_data(SymbolTableElement::IS_FLOAT, num);
      SymbolTableElement *elem = table->fctns.data(node->extra_type);

      // If inp came frame a node like INT or FLOAT, it will contain only
      // one value, but if it came from KEY, it will have a different value
      // for each atom.  Check for the relevant case.
      if (inp->num == num) {
        for (i=0; i<num; i++) 
          ret->dval[i] = elem->fctn(inp->dval[i]);
      } else {
        // assumes that functions return the same value on the same input
        // (i.e. this would not work for functions like rand()...)
        double d = elem->fctn(inp->dval[0]);
        for (i=0; i<num; i++) 
          ret->dval[i] = d;
      }
      delete inp;
      return ret;
      }

    case ADD:
    case SUB:
    case MULT:
    case MOD:
    case EXP:
    case DIV: 
      return eval_mathop(node, num, flgs);

    case UMINUS:
      tmp = eval(node->left, num, flgs);
      tmp->convert(SymbolTableElement::IS_FLOAT);
      for (i=0; i<tmp->num; i++) {
        tmp->dval[i] = -tmp->dval[i];
      }
      return tmp;

    case COMPARE:
      eval_compare(node, num, flgs);
      break;

    case WITHIN:  // this gets the coordinates from 'x', 'y', and 'z'
      eval_within(node, num, flgs);
      break;

    case EXWITHIN:  // this gets the coordinates from 'x', 'y', and 'z'
      eval_exwithin(node, num, flgs);
      break;

    case PBWITHIN:  // this gets the coordinates from 'x', 'y', and 'z'
      eval_pbwithin(node, num, flgs);
      break;

#if defined(NEAREST)
    case NEAREST:  // this gets the coordinates from 'x', 'y', and 'z'
      eval_k_nearest(node, num, flgs);
      break;
#endif

#if defined(WITHINBONDS)
    case WITHINBONDS:
      eval_within_bonds(node, num, flgs);
      break;
#endif

#if defined(MAXRINGSIZE)
    case MAXRINGSIZE:
      eval_maxringsize(node, num, flgs);
      break;
#endif

#if defined(RINGSIZE)
    case RINGSIZE:
      eval_ringsize(node, num, flgs);
      break;
#endif

    case SAME:
      eval_same(node, num, flgs);
      break;

    case SINGLE:
      eval_single(node, num, flgs);
      break;

    default: 
      msgWarn << "ParseTree::eval() unknown node type: " << node->node_type << sendmsg;
      break;
  }

  return NULL;
}


// detect recursive atom selection macros
void ParseTree::eval_find_recursion(atomparser_node *node, int *found,
                                    hash_t *hash) {
  // walk the parse tree just like in eval.  If any new node types are
  // created whose operands can contain singlewords, they must be included
  // here.
  switch (node->node_type) {
    case AND:
    case OR:
      eval_find_recursion(node->left, found, hash);
      eval_find_recursion(node->right, found, hash);
      // we don't need to check for COMPARE because singlewords cannot be
      // part of the operands.
      break;

    case NOT:
    case UMINUS:
    case WITHIN:
    case EXWITHIN:
    case SAME:
    case FUNC:
      eval_find_recursion(node->left, found, hash);
      break;

    case SINGLE:
      {
        const char *thisword = table->fctns.name(node->extra_type);
        const char *macro = table->get_custom_singleword(thisword);
        if (macro) {
          if (hash_insert(hash, thisword, 0) != HASH_FAIL) {
            *found = 1;
          } else {
            ParseTree *subtree = table->parse(macro);
            if (subtree != NULL) {
              eval_find_recursion(subtree->tree, found, hash);
              delete subtree;
            } else {
              /* XXX prevent things like this from causing a crash:
               *   atomselect macro A { segid A }
               *   atomselect macro AB { A }
               */
              msgErr << "ParseTree) internal processing error, NULL "
                     << "subtree value while checking recursion" << sendmsg;
            }
            hash_delete(hash, thisword);
          }
        }
      }
      break;
  }
}


// detect recursive atom selection macros 
int ParseTree::find_recursion(const char *head) {
  hash_t hash;
  hash_init(&hash, 10);
  hash_insert(&hash, head, 0);
  int found = 0;
  eval_find_recursion(tree, &found, &hash);
  hash_destroy(&hash);
  return found;
}


// this will set a list of flags, then call eval on that array
// it returns 0 if things went bad, 1 otherwise
// the array either set to 1 (if selected) or 0.
int ParseTree::evaluate(int num_atoms, int *flgs) {
  int num = num_atoms;
  if (!tree || num < 0 ) {  // yes, I allow 0 atoms
    return 0;
  }

  // initialize flags array to true, eval() results are AND'd/OR'd in
  for (int i=0; i<num; i++) {
    flgs[i] = 1;
  }

  // things should never return data so complain if that happens
  symbol_data *retdat = eval(tree, num, flgs);
  if (retdat) {
    msgErr << "Atom selection returned data when it shouldn't\n" << sendmsg;
    delete retdat;
  }

  return 1;
}


// delete and recreate the data space
void symbol_data::make_space(void) {
  free_space(); // delete any existing array first
  switch(type) {
    case SymbolTableElement::IS_FLOAT:
      dval = new double[num];
      break;

    case SymbolTableElement::IS_INT:
      ival = new int[num];
      break;

    case SymbolTableElement::IS_STRING:
      sval = new char *[num];
      memset(sval, 0, num*sizeof(char *)); // init pointers to NULL
      break;
  }
}


// just delete the space
void symbol_data::free_space(void) {
  switch (type) {
    case SymbolTableElement::IS_FLOAT:
      if (dval) 
        delete [] dval;
      dval = NULL;
      break;

    case SymbolTableElement::IS_INT: 
      if (ival) 
        delete [] ival;
      ival = NULL;
      break;

    case SymbolTableElement::IS_STRING:
      if (sval) {
        // free individual strings if necessary
        if (free_sval) 
          for (int i=0; i<num; i++) free(sval[i]);

        delete [] sval;
        sval = NULL;
      }
      free_sval = 0;
      break;

    default:
      msgErr << "Unknown data type " << (int)type
             << " in symbol_data::free_space" << sendmsg;
  }
}


// given the new type and the number of elements, create space
symbol_data::symbol_data(SymbolTableElement::symtype new_type, int new_num) {
  type = new_type;
  num = new_num;
  dval = NULL;
  ival = NULL;
  sval = NULL;
  free_sval = 0;
  make_space();
}


symbol_data::~symbol_data(void) {
  free_space();
}


void symbol_data::convert(SymbolTableElement::symtype totype) {
  // do nothing if types are the same
  if (totype == type) 
    return;

  // convert to floating point
  if (totype == SymbolTableElement::IS_FLOAT) {
    double *tmp = new double[num];
    if (type == SymbolTableElement::IS_INT) {
      for (int i=num-1; i>=0; i--) {
        tmp[i] = (double) ival[i];
      }
    } else { // SymbolTableElement::IS_STRING
      for (int i=num-1; i>=0; i--) {
        // XXX sval[i] should _never_ be NULL, but there's a bug somewhere
        // that allows a conversion from a residue name to a floating point
        // value, which occurs without setting the string value since it's
        // a built-in query rather than a user-provided string.  When this
        // (extremely rare) situation occurs, the code could crash here.
        // e.g.: mol selection {(not resname SOD) and (segname % 11 == 0)}
        // This test will prevent the crash, but does not solve the root of
        // the problem.
        if (sval[i] != NULL) {
          tmp[i] = atof(sval[i]);
        } else {
          for (int j=num-1; j>=0; j--) {
            tmp[i] = 0.0f; 
          } 
          msgErr << "ParseTree) internal processing error, NULL string value " 
                 << "while converting to floating point" << sendmsg;
          break;
        }
      }
    }
    free_space();
    type = totype;
    dval = tmp;
    return;
  }

  // convert to string
  if (totype == SymbolTableElement::IS_STRING) {
    char **tmp = new char*[num];
    memset(tmp, 0, num*sizeof(char *)); // init pointers to NULL
    char s[100];
    if (type == SymbolTableElement::IS_INT) {
      for (int i=num-1; i>=0; i--) {
        sprintf(s, "%ld", (long) ival[i]);
        tmp[i] = strdup(s);
      }
    } else { // SymbolTableElement::IS_FLOAT
      for (int i=num-1; i>=0; i--) {
        sprintf(s, "%f", (double) dval[i]);
        tmp[i] = strdup(s);
      }
    }
    free_space();
    type = totype;
    sval = tmp;
    free_sval = TRUE;
    return;
  }

  // convert to integer
  if (totype == SymbolTableElement::IS_INT) {
    int *tmp = new int[num];
    if (type == SymbolTableElement::IS_FLOAT) {
      for (int i=num-1; i>=0; i--) {
        tmp[i] = (int) dval[i];
      }
    } else { // SymbolTableElement::IS_STRING
      for (int i=num-1; i>=0; i--) {
        // XXX sval[i] should _never_ be NULL, but there's a bug somewhere
        // that allows a conversion from a residue name to a floating point
        // value, which occurs without setting the string value since it's
        // a built-in query rather than a user-provided string.  When this
        // (extremely rare) situation occurs, the code could crash here.
        // e.g.: mol selection {(not resname SOD) and (segname % 11 == 0)}
        // This test will prevent the crash, but does not solve the root of
        // the problem.
        if (sval[i] != NULL) {
          tmp[i] = atoi(sval[i]);
        } else {
          for (int j=num-1; j>=0; j--) {
            tmp[i] = 0; 
          } 
          msgErr << "ParseTree) internal processing error, NULL string value " 
                 << "while converting to integer" << sendmsg;
          break;
        }
      }
    }
    free_space();
    type = totype;
    ival = tmp;
    return;
  }
}

