
%{


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
 *	$RCSfile: AtomLexer.l,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.54 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * break atom selection information into its tokens
 *
 ***************************************************************************/


// Avoid compilation problems on Redhat Linux 7.x and Windows.
#define YY_ALWAYS_INTERACTIVE 1

#include <stdlib.h>

#if defined(ARCH_AIX4) || defined(ARCH_AIX3)
#include <strings.h>
#else
#include <string.h>
#endif

#include "AtomParser.h"
#include "y.tab.h"
#include "Inform.h"
#include "SymbolTable.h"

/* redefine the input to come from a string */
#undef input
#undef unput
#if 1
#define YY_DECL extern "C" int yylex YY_PROTO(( void ))

// this was compiled with flex, so I need a different input mechanism
#undef YY_INPUT
#define YY_INPUT(buf,result,max) { \
	int c = *atomparser_yystring; \
	result = (c == 0) ? YY_NULL : \
	(buf[0] = c, atomparser_yystring++, 1); \
}
#else
// XXX Alternate input/unput routines, no longer used.
#define input() (*atomparser_yystring++)
#define unput(c) (*--atomparser_yystring = c)
#endif

/* Native Solaris lex has its own way of getting input when */
/* using C++; lex_input() */
#if 0
//#if defined(ARCH_SOLARIS2) || defined(ARCH_SOLARIS2_64) || defined(ARCH_SOLARISX86)
#undef lex_input
#define lex_input() input()
#endif

/* Also, GCC on Solaris, and Sun C++ 5.x put "all" in read-only memory */
/* which causes unput to core dump without this special case.          */
/* This may also be a problem for other platforms, don't know yet      */
#if defined(ARCH_SOLARIS2) || defined(ARCH_SOLARIS2_64) || defined(ARCH_SOLARISX86)
#undef unput
#define unput lex_unput_only_if_needed

int lex_unput_only_if_needed(int c) {
  atomparser_yystring--;
  if (c != *atomparser_yystring) {
    *atomparser_yystring = c;
  }
  return c;
}
#endif

%}

float		([0-9]+\.|([0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?)
numeric		[0-9]
alpha		[a-zA-Z_]
alphanum	({alpha}|{numeric})

%{
 /* like_1A  catches raw words like 1A, 3', 5*A
		(start with a number and contain an alpha or ', ", or *)
    like_C5' catches ones like C5' O*, O5*
		(starts with an alpha and has a ', ", or *)
  */
 /*  Problem is that * is too easy to confuse, eg, "x* x"
   I won't worry about this until the next rewrite and only allow
   primes
wierd		[\'\"\*]
  */

%}

wierd		[\'\"]
like_1A		{numeric}+({alpha}|{wierd})+({alphanum}|{wierd})*
like_C5prime	{alpha}+{alphanum}*{wierd}+({alphanum}|{wierd})*
		
%%
{float}		{
		  yylval.dval = atof((char *) yytext);
		  return(FLOATVAL);
		}
(\(|\{)		{ return('('); /* allow '(' and '{' */ }
(\)|\})		{ return(')'); /* allow ')' and '}' */ }
(and|\&\&)	{ return(AND);}
(or|\|\|)	{ return(OR);}
not		{ return(NOT);}
within		{ return(WITHIN);}
exwithin	{ return(EXWITHIN);}
pbwithin	{ return(PBWITHIN);}
withinbonds     { return(WITHINBONDS);}
maxringsize	{ return(MAXRINGSIZE);}
ringsize	{ return(RINGSIZE);}
of              { return(OF);}
from            { return(FROM);}
same		{ return(SAME);}
as		{ return(AS);}
(to|\.\.\.)	{ return(THROUGH);}
where           { return(WHERE);}
nearest         { return(NEAREST);}

:		{ return(RANGE);}
\"([^\"]|\\\\|\\\")*\"	{ /* escapes \ and " in a "string"  */
		  yylval.node = new atomparser_node(STRWORD);
		  yylval.node->sele.s = ((char *) yytext)+1;
		  /* GNU needs a "chop" */
		  yylval.node->sele.s.chop(1);
		  /* replace \\ with \ */
		  yylval.node->sele.s.gsub("\\\"", "\"");
		  /* replace \" with " */
		  yylval.node->sele.s.gsub("\\\\", "\\");
		  yylval.node->sele.st = DQ_STRING;
		  return STRWORD;
		}
\'([^\']|\\\\|\\\')*\'	{ /* escapes \ and ' in a 'string', */
		  /* for instance, this lets you do 'C5\''  */
		  yylval.node = new atomparser_node(STRWORD);
		  yylval.node->sele.s = ((char *) yytext)+1;
		  /* GNU needs a "chop" */
		  yylval.node->sele.s.chop(1);
		  /* replace \' with ' */
		  yylval.node->sele.s.gsub("\\'", "'");
		  /* replace \\ with \ */
		  yylval.node->sele.s.gsub("\\\\", "\\");
		  yylval.node->sele.st = SQ_STRING;
		  return STRWORD;
		}
\"[^\"]*	{
		  msgErr << "Unterminated double quoted string: "
			   << (char *) yytext << sendmsg;
		  return PARSEERROR;
		}
\'[^\']*	{
		  msgErr << "Unterminated single quoted string: "
			   << (char *) yytext << sendmsg;
		  return PARSEERROR;
		}

"<"		{ return(NLT); /* these are for numeric comparisons */}
"<="		{ return(NLE);}
"=="		{ return(NEQ);}
"="		{ return(NEQ); /* many people use it */}
">="		{ return(NGE);}
">"		{ return(NGT);}
"!="		{ return(NNE);}

"lt"		{ return(SLT); /* these are for string comparisons */}
"le"		{ return(SLE);}
"eq"		{ return(SEQ);}
"ge"		{ return(SGE);}
"gt"		{ return(SGT);}
"ne"		{ return(SNE);}
"=~"		{ return(MATCH);}

"+"		{ return(ADD);}
"-"		{ return(SUB);}
"/"		{ return(DIV);}
"*"		{ return(MULT);}
"%"		{ return(MOD);}
"^"		{ return(EXP);}
"**"		{ return(EXP);}

[ \t\n]		;

{like_1A}|{like_C5prime} { /* catch raw names like: */
		  /* 5' 1A C4' 3'A           */
		  yylval.node = new atomparser_node(STRWORD);
		  yylval.node->sele.s = (char *) yytext;
		  yylval.node->sele.st = RAW_STRING;
		  return STRWORD;
		}
{numeric}+	{ yylval.ival = atoi((char *) yytext); 
		  return(INTVAL); 
		}
({alpha}{alphanum}*)|("$"{alphanum}*)|("@"{alphanum}*) {  
                       /* standard variable names */
		  yylval.node = new atomparser_node(STRWORD);
		  yylval.node->sele.s = (char *) yytext;
		  yylval.node->sele.st = RAW_STRING;
		  int len = strlen((char *) yytext);
		  int i;
		  if ((i= atomparser_yylookup((char *) yytext, len)) >= 0) {
		     yylval.node->extra_type = i;
		     if (atomparser_symbols->fctns.data(i)->is_a ==
			 SymbolTableElement::FUNCTION) {
			yylval.node->node_type = FUNC;
			return FUNC;
		     } else if (atomparser_symbols->fctns.data(i)->is_a ==
				SymbolTableElement::KEYWORD) {
			yylval.node->node_type = KEY;
			return KEY;
		     } else if (atomparser_symbols->fctns.data(i)->is_a ==
				SymbolTableElement::SINGLEWORD) {
			yylval.node->node_type = SINGLE;
			return SINGLE;
		     } else if (atomparser_symbols->fctns.data(i)->is_a ==
				SymbolTableElement::STRINGFCTN) {
			yylval.node->node_type = STRFCTN;
			return STRFCTN;
		     }
		  }
		  return STRWORD;
                }
.		{ msgErr << "Bad character:"
			 << int(*yytext) << ':'
			 << *yytext << sendmsg; 
		  return PARSEERROR; 
		}
%%

#include "SymbolTable.h"

int atomparser_yylookup(const char *s, int) 
{
  return atomparser_symbols->find_attribute(s);
}

// pointer to the input string
char *atomparser_yystring;

// pointer to the array of symbols
SymbolTable *atomparser_symbols;


