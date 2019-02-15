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
 *	$RCSfile: AtomParser.y,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.51 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  a parser for atom selections
 *
 ***************************************************************************/



#include <stdio.h>
#include <string.h>
#include "AtomParser.h"
#include "Inform.h"

#if !defined(_MSC_VER)
extern "C" int yyparse();
#endif
extern "C" void yyerror(const char *s);
extern "C" int yylex();

atomparser_node *atomparser_result;
%}

%union {
	int ival;
	double dval;
	atomparser_node *node;
}
	

%type <ival> compare

%token <node> KEY WITHIN EXWITHIN PBWITHIN WITHINBONDS MAXRINGSIZE RINGSIZE WHERE FUNC STRFCTN SAME NEAREST
%left <node> SINGLE
%nonassoc FROM OF AS
%nonassoc THROUGH
%token PARSEERROR RANGE
%token <dval> FLOATVAL
%token <ival> INTVAL 
%token <node> STRWORD
%token COMPARE

%left OR
%left AND
%token LT LE EQ GE GT NE 
%nonassoc <ival> NLT NLE NEQ NGE NGT NNE
%nonassoc <ival> SLT SLE SEQ SGE SGT SNE MATCH 

%left ADD SUB
%left MULT DIV MOD
%left EXP
%left nonassoc NOT
%nonassoc UMINUS

%type <node> selection  expression
%type <node> keyword_list string_list word string_list_ele strfctn_list
%type <dval> number


%%
selection_list:				{// printf("Blank line.\n");
					  atomparser_result =  NULL;
					}
	| selection			{ //printf("Parsed a line\n");
					  if (*atomparser_yystring != 0) {
    msgErr << "Selection terminated too early" << sendmsg;
    if ($1) delete $1;
    $1 = NULL;
					  }
					  atomparser_result = $1;
					}
	| error				{ //printf("Error occured\n");
					  atomparser_result = NULL;
//					  yyerrok;
					}
	;

selection: '(' selection ')'		{ // printf("Parens\n");
					  $$ = $2;
					}
	| NOT selection			{ $$ = new atomparser_node(NOT);
					  $$->left = $2;
					}
	| SINGLE			{ $$ = $1; }
	| SINGLE selection		{ $$ = new atomparser_node(AND);
					  $$->left = $1;
					  $$->right = $2;
					}
	| selection AND selection	{ //printf("AND\n");
					  $$ = new atomparser_node(AND);
					  $$->left = $1;
					  $$->right = $3;
					}
	| selection OR selection	{ //printf("OR\n");
					  $$ = new atomparser_node(OR);
					  $$->left = $1;
					  $$->right = $3;
					}
	| keyword_list			{ $$ = $1; }
	| strfctn_list			{ $$ = $1; }
	| WITHIN number OF selection	{ $$ = new atomparser_node(WITHIN);
					  $$->left = $4;
					  $$->dval = $2;
					}
	| EXWITHIN number OF selection	{ $$ = new atomparser_node(EXWITHIN);
					  $$->left = $4;
					  $$->dval = $2;
					}
	| PBWITHIN number OF selection	{ $$ = new atomparser_node(PBWITHIN);
					  $$->left = $4;
					  $$->dval = $2;
					}
	| WITHINBONDS INTVAL OF selection { $$ = new atomparser_node(WITHINBONDS);
					  $$->left = $4;
					  $$->ival = $2;
					}
	| NEAREST INTVAL THROUGH selection { $$ = new atomparser_node(NEAREST);
					  $$->left = $4;
					  $$->ival = $2;
					}
	| MAXRINGSIZE INTVAL FROM selection { $$ = new atomparser_node(MAXRINGSIZE);
					  $$->left = $4;
					  $$->ival = $2;
					}
	| RINGSIZE INTVAL FROM selection { $$ = new atomparser_node(RINGSIZE);
					  $$->left = $4;
					  $$->ival = $2;
					}
	| SAME KEY AS selection		{ $$ = $2;
					  $$->node_type = SAME;
					  $$->left = $4;
					}
	| expression compare expression {$$ = new atomparser_node(COMPARE);
					  $$ -> ival = $2;
					  $$ -> left = $1;
					  $$ -> right = $3;
					}
	;

keyword_list: KEY string_list		{
					  $1 -> left = $2;
					  $2 -> right = NULL;
					  $$ = $1;
					}
	;

strfctn_list: STRFCTN string_list	{  $1 -> left = $2;
					   $2 -> right = NULL;
					   $$ = $1;
					}
	;

 word:  STRWORD				{ 
					  $$ = $1;
					  //printf("Single\n");
					}
	| INTVAL		 	{ $$ = new atomparser_node(STRWORD);
					  //printf("converted int\n");
					  char s[100];
					  sprintf(s, "%ld", (long) $1);
					  $$ -> sele.s = s;
					  $$ -> sele.st = RAW_STRING;
					}
	| FLOATVAL			{ $$ = new atomparser_node(STRWORD);
					  char s[100];
					  sprintf(s, "%f", (double) $1);
					  $$ -> sele.s = s;
					  $$ -> sele.st = RAW_STRING;
					}
	;

string_list_ele: word			{ 
					  $1 -> right = $1;
   					  $$ = $1; 
					}
	| word THROUGH word		{ $1 -> right = $3;
					  $1 -> left = $3;
					  $1 -> extra_type = 1;
					  $$ = $1;
					  //printf("Using through\n");
					}
	;

string_list:
	string_list_ele			{ $$ = $1; }
        | string_list string_list_ele	{ /* copy the new word on the list */
               /* like a linked list, with head's right pointed to the end */
	       /* element and head's left pointed to the second element    */
				          $1 -> right -> left = $2;
					  $1 -> right = $2 -> right;
					  $2 -> right = NULL;
					  // printf("Returning\n");
					  $$ = $1;
				       }
	;

number: FLOATVAL			{ $$ = $1;// printf("## %lf\n", $$);
					}
	| INTVAL			{ $$ = (double) $1; 
					  // printf("# %lf\n", $$);
					}
	;

expression: FLOATVAL			{ $$ = new atomparser_node(FLOATVAL);
					  $$->dval = $1; 
					}
	|INTVAL				{ $$ = new atomparser_node(INTVAL);
					  $$->ival = $1; 
					}
	|STRWORD				{ $$ = $1; 
					}
	| '(' expression ')'		{ $$ = $2; }
	| expression ADD expression	{ $$ = new atomparser_node(ADD);
					  $$->left = $1;
					  $$->right = $3;
					}
	| expression SUB expression	{ $$ = new atomparser_node(SUB);
					  $$->left = $1;
					  $$->right = $3;
					}
	| SUB expression %prec UMINUS	{ $$ = new atomparser_node(UMINUS);
					  $$->left = $2;
					}
	| ADD expression %prec UMINUS	{ $$ = $2;
					}
	| expression MOD expression     { $$ = new atomparser_node(MOD);
					  $$->left = $1;
					  $$->right = $3;
					}
	| expression EXP expression	{ $$ = new atomparser_node(EXP);
					  $$->left = $1;
					  $$->right = $3;
					}
	| expression MULT expression	{ $$ = new atomparser_node(MULT);
					  $$->left = $1;
					  $$->right = $3;
					}
	| expression DIV expression	{ $$ = new atomparser_node(DIV);
					  $$->left = $1;
					  $$->right = $3;
					}
	| KEY				{ $$ = $1; }
	| FUNC '(' expression ')'	{ $1->left = $3;
					  $$ = $1;
					}
	;
					
compare:	  NLT  	{ $$ = NLT; }
		| NLE	{ $$ = NLE; }
		| NEQ	{ $$ = NEQ; }
		| NGE	{ $$ = NGE; }
		| NGT	{ $$ = NGT; }
		| NNE   { $$ = NNE; }
		| SLT  	{ $$ = SLT; }
		| SLE	{ $$ = SLE; }
		| SEQ	{ $$ = SEQ; }
		| SGE	{ $$ = SGE; }
		| SGT	{ $$ = SGT; }
		| SNE	{ $$ = SNE; }
		| MATCH { $$ = MATCH; }
		;


%%

extern "C" void yyerror(const char *s) {
  msgErr << s << sendmsg;
}

// everything comes from a string, so there is no way to
// reset "yyin" (or whatever) to the next input
extern "C" int yywrap(void) {
  return 1;
}

