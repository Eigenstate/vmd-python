/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current contact addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

   The WORK is Copyright (1997,1998,1999) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
/****************************************************************************
gjutil.c:  Various utility routines - error checking malloc and
free, string functions etc...

TERMS OF USE:

The computer software and associated documentation called STAMP hereinafter
referred to as the WORK is more particularly identified and described in 
the LICENSE.

The WORK was written and developed by: Geoffrey J. Barton

EMBL-European Bioinformatics Institute
Wellcome Genome Campus
Hinxton, Cambridge, CB10 1SD U.K.

The WORK is Copyright (1992) Geoffrey J. Barton.


CONDITIONS:

The WORK is made available for educational and non-commercial research 
purposes.

For commercial use, a commercial licence is required - contact the author
at the above address for details.

The WORK may be modified, however this text, all copyright notices and
the authors' address must be left unchanged on every copy, and all
changes must be documented after this notice.  A copy of the
modified WORK must be supplied to the author.

******************************************************************************/
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <gjutil.h>

/* define pointers for standard streams to allow redefinition to files */

FILE *std_err;
FILE *std_in;
FILE *std_out;

void *GJmalloc(size_t size)
/* malloc with simple error check */
/* G. J. Barton 1992 */
{
	void *ptr;
	ptr = (void *) malloc(size);
	if(ptr == NULL){
		GJerror("malloc error");
		exit(0);
	}
	return ptr;
}

void *GJrealloc(void *ptr,size_t size)
/* realloc with error check */
/* G. J. Barton 1992 */
{
	ptr = (void *) realloc(ptr,size);
	if(ptr == NULL){
		GJerror("realloc error");
		exit(0);
	}
	return ptr;
}
void *GJmallocNQ(size_t size)
/* as for GJmalloc, but returns NULL on error*/
/* G. J. Barton 1992 */
{
	void *ptr;
	ptr = (void *) malloc(size);
	if(ptr == NULL){
		GJerror("malloc error");
                return NULL;
	}
	return ptr;
}

void *GJreallocNQ(void *ptr,size_t size)
/* as for GJrealloc with error check but returns NULL on error*/
/* G. J. Barton 1992 */
{
	ptr = (void *) realloc(ptr,size);
	if(ptr == NULL){
		GJerror("realloc error");
		return NULL;
	}
	return ptr;
}
void GJfree(void *ptr)
/* free with error check */
/* G. J. Barton 1992 */
{
	if(ptr == NULL){
		GJerror("Attempt to free NULL pointer");
		exit(0);
	}
	free(ptr);
}

void GJerror(const char *prefix)
/* writes error message contained in prefix and contents of errno
   to std_err.
*/
/* G. J. Barton 1992 */
{
	if(prefix != NULL){
		if(*prefix != '\0'){
			fprintf(std_err,"%s: ",prefix);
		}
	}
	fprintf(std_err,"%s\n",strerror(errno));
}

/*
error:   calls GJerror 
*/
void error(const char *str,int flag)
{
    GJerror(str);
    if(flag)exit(0);
}


char *GJstoupper(const char *s)
/* return a copy of s in upper case */
/* G. J. Barton 1992 */
{
	char *temp;
	int i;
	temp = GJstrdup(s);
	i=0;
	while(temp[i] != '\0'){
		temp[i] = toupper(temp[i]);
		++i;
	}
	return temp;
}
char *GJstolower(const char *s)
/* return a copy of s in lower case */
/* G. J. Barton 1992 */
{
	char *temp;
	int i;
	temp = GJstrdup(s);
	i=0;
	while(temp[i] != '\0'){
		temp[i] = tolower(temp[i]);
		++i;
	}
	return temp;
}
char *GJstoup(char *s)
/* return s in upper case */
/* G. J. Barton 1992 */
{
	int i;
	i=0;
	while(s[i] != '\0'){
		s[i] = toupper(s[i]);
		++i;
	}
	return s;
}
char *GJstolo(char *s)
/* return s in lower case */
/* G. J. Barton 1992 */
{
	int i;
	i=0;
	while(s[i] != '\0'){
		s[i] = tolower(s[i]);
		++i;
	}
	return s;
}  

char *GJstrdup(const char *s)
/* returns a pointer to a copy of string s */
/* G. J. Barton 1992 */

{
	char *temp;
	temp = (char *) GJmalloc(sizeof(char) * (strlen(s)+1));
	temp = strcpy(temp,s);
	return temp;
}

FILE *GJfopen(const char *fname,const char *type,int action)
/* a file open function with error checking.  The third argument
is set to 0 if we want a failed open to return, or 1 if we
want a failed open to exit the program.
*/
/* G. J. Barton 1992 */
/* modified July 1995 - error message only printed if action is 1 */
{
	FILE *ret_val;
	ret_val = fopen(fname,type);
	if(ret_val == NULL){
	  /*	  GJerror(strcat("Cannot Open File: ",fname));*/
		if(action == 1){
		  GJerror("Cannot Open File");
		  exit(1);
		}
	}
	return ret_val;
}

int GJfclose(FILE *fp,int action)
/* a file close function with error checking.  The second argument
is set to 0 if we want a failed close to return, or 1 if we
want a failed close to exit the program.
*/
/* G. J. Barton 1992 */
{
	int ret_val;
	ret_val = fclose(fp);
	if(ret_val != 0){
		if(action == 1){
		        GJerror("Error closing File");
			exit(1);
		}
	}
	return ret_val;
}


struct file *GJfilemake(const char *name,const char *type,int action)
/* If action = 1 then 
Tries to open the file with the given name.  If successful returns 
a pointer to a struct file structure with the name and filehandle.  If
the open fails, or action= 0 then returns a struct file structure 
with the name and a NULL filehandle */
/* G. J. Barton 1995 */
{
	struct file *ret_val;
	ret_val = (struct file *) GJmalloc(sizeof(struct file));
	ret_val->name = GJstrdup(name);
	if(action == 1) {
	  ret_val->handle = GJfopen(ret_val->name,type,0);
	}
	return ret_val;
}

struct file *GJfilerename(struct file *ret_val, const char *name)
/* When passed the fval structure - renames the name part of the
file structure to name, if the handle is non null it tries to close 
the file, then sets the file handle to NULL. */
/* G. J. Barton 1995 */
{
	if(ret_val->name != NULL) {
	  GJfree(ret_val->name);
	  ret_val->name = GJstrdup(name);
	}
	if(ret_val->handle != NULL) {
	  GJfclose(ret_val->handle,0);
	  ret_val->handle = NULL;
	}
	return ret_val;
}

int GJfileclose(struct file *fval,int action)
/* Closes a file named in the  struct file structure.   */

/* G. J. Barton July 1995 */
{
	int ret_val;
	ret_val = fclose(fval->handle);
	if(ret_val != 0){
		if(action == 1){
		        GJerror("Error closing File");
			exit(1);
		}
	}
	return ret_val;  

}

int GJfileclean(struct file *fval,int action)
/*  Closes the file then sets the file pointer to NULL, then 
    frees the filename string */

/* G. J. Barton July 1995 */
{
	int ret_val;
	ret_val = GJfclose(fval->handle,0);
	if(ret_val != 0){
		if(action == 1){
		        GJerror("Error closing File");
			exit(1);
		}
	}
	fval->handle = NULL;
	GJfree(fval->name);
	return ret_val;  
}

void GJinitfile()
/* just set the standard streams */
{
	std_err = stderr;
	std_in = stdin;
	std_out = stdout;
}

char *GJfnonnull(char *string)
/* return pointer to first non null character in the string */
/* this could cause problems if the string is not null terminated */
{
	while(*string != '\0'){
		++string;
	}
	return ++string;
}

char *GJstrappend(char *string1, char *string2)
/* appends string2 to the end of string2.  Any newline characters are removed
from string1, then the first character of string2 overwrites the null at the
end of string1.
string1 and string2 must have been allocated with malloc.
*/
/* G. J. Barton July 1992 */
{
	char *ret_val;
	ret_val = GJremovechar(string1,'\n');
	ret_val = (char *) GJrealloc(ret_val,
			   sizeof(char) * (strlen(ret_val) + strlen(string2) + 1));
        ret_val = strcat(ret_val,string2);
        return ret_val;
}

char *GJremovechar(char *string,char c)
/* removes all instances of character c from string
   returns a pointer to the reduced, null terminated string
*/
/* G. J. Barton (July 1992) */
{
	char *temp;
	int j,i,nchar;
	nchar = 0;
	i=0;
	while(string[i] != '\0'){
		if(string[i] == c){
			++nchar;
		}
		++i;
	}
	if(nchar == 0){
		 return string;
	}else{
		temp = (char *) GJmalloc(sizeof(char) * (strlen(string)-nchar));
		j=0;
		i=0;
		while(string[i] != '\0'){
			if(string[i] != c){
				temp[j] = string[i];
				++j;
			}
			++i;
		}
		temp[++j] = '\0';
		GJfree(string);
		return temp;
	}
}

char *GJsubchar(char *string,char c2,char c1)
/* substitutes c1 for c2 in string
*/
/* G. J. Barton (July 1992) */
{
	int /*j,*/i,nchar;
	nchar = 0;
	i=0;
	while(string[i] != '\0'){
		if(string[i] == c1){
                    string[i] = c2;
		}
		++i;
	}
	return string;
}

/* create a string and if fchar != NULL fill with characters  */
/* always set the len-1 character to '\0' */

char *GJstrcreate(size_t len,char *fchar)
{
	char *ret_val;
	ret_val = (char *) GJmalloc(sizeof(char) * len);
	--len;
	ret_val[len] = '\0';
	if(fchar != NULL){
		while(len > -1){
			ret_val[len] = *fchar;
			--len;
		}
	}
	return ret_val;
}

/* searches for string s2 in string s1 and returns pointer to first instance
of s2 in s1 or NULL if no instance found.  s1 and s2 must be null terminated
*/		
char *GJstrlocate(char *s1, char *s2)
{
    int i=0;
    int j=0;
    int k;
    if(strlen(s1) == 0 || strlen(s2) == 0) return NULL;
    while(i<strlen(s1)){
        j=0;
        k=i;
        while(j<strlen(s2) && s1[k] == s2[j]){
                ++k;
                ++j;
        }
        if(j == strlen(s2)) return &s1[i];
        ++i;
    }
    return NULL;
}


/* GJstrtok()

This version of strtok places the work pointer at the location of the first 
character in the next token, rather than just after the last character of the 
current token.  This is useful for extracting quoted strings 
*/

char *GJstrtok(char *input_string,const char *token_list)
{
  static char *work;
  char *return_ptr;

  if(input_string != NULL){
    /* first call */
    work =  input_string;
  }

  /* search for next non-token character */
  while(strchr(token_list,*work)!=NULL){
    ++work;
  }

  if(*work == '\0'){
    /* if we've reached the end of string, then return NULL */
    return NULL;
  }else{
    return_ptr = (char *) work;
    while(strchr(token_list,*work) == NULL){
      if(*work == '\0'){
	/* end of the string */
	return return_ptr;
      }else{
	++work;
      }
    }
    *work = '\0';
    ++work;
    /* now increment work until we find the next non-delimiter character */
    while(strchr(token_list,*work) != NULL){
      if(*work == '\0'){
	break;
      }else{
	++work;
      }
    }
    return return_ptr;
  }
}
/**************************************************************************
return a pointer to space for a rectangular unsigned character array
Version 2.0  ANSI and uses GJmallocNQ
--------------------------------------------------------------------------*/

unsigned char **uchararr(int i,int j)
{
    unsigned char **temp;
    int k, rowsiz;

    temp = (unsigned char **) GJmallocNQ(sizeof(unsigned char *) * i);
    if(temp == NULL) return NULL;

    rowsiz = sizeof(unsigned char) * j;

    for (k = 0; k < i; ++k){
	temp[k] =  (unsigned char *) GJmallocNQ(rowsiz);
	if(temp[k] == NULL) return NULL;
    }
    return temp;
}

/**************************************************************************
return a pointer to space for a rectangular signed character array
Version 2.0  ANSI
--------------------------------------------------------------------------*/
signed char **chararr(int i,int j)

{
    signed char **temp;
    int k, rowsiz;

    temp = (signed char **) GJmallocNQ(sizeof(char *) * i);

    if(temp == NULL) return NULL;

    rowsiz = sizeof(char) * j;

    for (k = 0; k < i; ++k){
	temp[k] =  (signed char *) GJmallocNQ(rowsiz);
	if(temp[k] == NULL) return NULL;
    }
    return temp;
}


/* mcheck - check a call to malloc - if the call has failed, print the
error message and exit the program */
/* ANSI Version - also uses GJerror routine and ptr is declared void*/

void mcheck(void *ptr,char *msg)

{
    if(ptr == NULL){
        GJerror("malloc/realloc error");
	exit(0);
    }
}

/* set a string to blanks and add terminating nul */
char *GJstrblank(char *string,int len)

{
  --len;
  string[len] = '\0';
  --len;
  while(len > -1){
    string[len] = ' ';
    --len;
  }
  return string;
}

/* Initialise an unsigned char array */  
void GJUCinit(unsigned char **array,int i,int j,unsigned char val)
{
  int k,l;

  for(k=0;k<i;++k){
    for(l=0;l<j;++l){
      array[k][l] = val;
    }
  }
}
/*Initialise a signed char array */

void GJCinit(signed char **array,int i,int j,char val)

{
  int k,l;

  for(k=0;k<i;++k){
    for(l=0;l<j;++l){
      array[k][l] = val;
    }
  }
}

/* Initialise an integer vector  */  
void GJIinit(int *array,int i,int val)
{
  int k;

  for(k=0;k<i;++k){
      array[k] = val;
  }
}

/******************************************************************
GJcat:  concatenate N NULL terminated strings into a single string.
The source strings are not altered
Author: G. J. Barton (Feb 1993)
------------------------------------------------------------------*/
char *GJcat(int N,...)
{
	va_list parminfo;
	int i,j,k;
	char **values;	/*input strings */
	int *value_len; /*lengths of input strings */
	int ret_len;    /*length of returned string */
	char *ret_val;  /*returned string */

	ret_len = 0;
	values = (char **) GJmalloc(sizeof(char *) * N);
	value_len = (int *) GJmalloc(sizeof(int *) * N);

	va_start(parminfo,N);

	/* Get pointers and lengths for the N arguments */
	for(i=0;i<N;++i){
		values[i] = va_arg(parminfo,char *);
		value_len[i] = strlen(values[i]);
		ret_len += value_len[i];
	}
	
	ret_val = (char *) GJmalloc(sizeof(char) * (ret_len+1));

	/* Transfer input strings to output string */
	k=0;
	for(i=0;i<N;++i){
		for(j=0;j<value_len[i];++j){
			ret_val[k] = values[i][j];
			++k;
		}
	}
	ret_val[k] = '\0';
	GJfree(values);
	GJfree(value_len);

	va_end(parminfo);

	return ret_val;
}

/************************************************************************

GJGetToken:

The aim of this routine is to emulate the strtok() function, but reading
from a file.  The functionality may differ slightly...

Characters are read from the file until a character that is found in
delim is encountered.  When this occurs, token is returned.  If the
file consists entirely of delimiters, then token is freed
and NULL is returned.  Similarly, if end of file is encountered.

------------------------------------------------------------------------*/


char *GJGetToken(FILE *in, const char *delim)

{
	int i;
	int c;
	char *token;

	i=0;

	token = GJmalloc(sizeof(char));
	
	while((c=fgetc(in)) != EOF){
		if(strchr(delim,c) == NULL){
			/* not a delimiter */
			token[i++] = c;
			token = GJrealloc(token,sizeof(char) * (i+1));
		}else if(i>0){
		        token[i] = '\0';
			return token;
		}
	}
/*	GJerror("End of File Encountered");*/
	GJfree(token);
	return NULL;
}

struct tokens *GJgettokens(const char *delims, char *buff)
/* This splits a buffer into tokens at each position defined by delims.*/
/* The structure returned contains the number of tokens and the */
/* tokens themselves as a char ** array */
{
  char *token;
  struct tokens *tok;
  
  token = strtok(buff,delims);
  if(token == NULL) return NULL;

  tok = (struct tokens *) GJmalloc(sizeof(struct tokens));
  tok->ntok = 0;

  tok->tok = (char **) GJmalloc(sizeof(char *));
  tok->tok[0] = GJstrdup(token);
  ++tok->ntok;

  while((token = strtok(NULL,delims)) != NULL) {
      tok->tok = (char **) GJrealloc(tok->tok,sizeof(char *) * (tok->ntok+1));
      tok->tok[tok->ntok] = GJstrdup(token);
      ++tok->ntok;
  }
  
  return tok;
}

/* frees a tokens structure */
void *GJfreetokens(struct tokens *tok)
{
  int i;
  for(i=0;i<tok->ntok;++i) {
    GJfree(tok->tok[i]);
  }
  GJfree(tok->tok);
  GJfree(tok);
  return 0;
}
  













