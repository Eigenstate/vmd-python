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
 *      $RCSfile: JString.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.21 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A minimalistic string class we use instead of similar classes from the
 *   STL or the GNU libraries for better portability and greatly reduced
 *   code size.  (only implements the functionality we actually need, doesn't
 *   balloon the entire VMD binary as some past string class implementations
 *   did).  Implements regular expression matching methods used by VMD.
 ***************************************************************************/

#include "JString.h"
#include <ctype.h>
#include <stdio.h>

char *JString::defstr = (char *)"";

JString& JString::operator+=(const JString& s) {
  char *newrep = new char[length() + s.length() + 1];
  ::strcpy(newrep, rep);
  ::strcat(newrep, s.rep);
  if (do_free) delete [] rep;
  rep = newrep;
  do_free = 1;
  return *this;
}


JString& JString::operator=(const JString& s) {
  if (rep != s.rep) {
    if (do_free) delete [] rep;
    rep  = new char[s.length()+1];
    ::strcpy(rep, s.rep);
    do_free = 1;
  }
  return *this;
}

JString& JString::operator=(const char *s) {
  if (s == NULL) return *this;
  if (do_free) delete [] rep;
  rep = new char[strlen(s)+1];
  ::strcpy(rep, s);
  do_free = 1;
  return *this;
}
  
JString& JString::operator=(const char c) {
  if (do_free) delete [] rep;
  rep=new char[2];
  rep[0]=c;
  rep[1]='\0';
  do_free =1 ;
  return *this;
}

JString& JString::operator+=(const char *s) {
  if (s==NULL) return *this;
  char *newrep = new char[length() + strlen(s) + 1];
  ::strcpy(newrep, rep);
  ::strcat(newrep, s);
  if (do_free) delete [] rep;
  rep = newrep;
  do_free =1; 
  return *this;
}
 
JString& JString::operator+=(const char c) {
  char *newrep = new char[length() + 2];
  ::strcpy(newrep,rep);
  newrep[length()]=c;
  newrep[length()+1]='\0';
  if (do_free) delete [] rep;
  rep=newrep;
  do_free = 1;
  return *this;
} 

JString operator+(const char* s, const JString& S) {
  JString retval;
  retval.rep = new char[::strlen(s) + S.length() + 1];
  ::strcpy(retval.rep, s);
  ::strcat(retval.rep, S.rep);
  retval.do_free = 1;
  return retval;
}

JString JString::operator+(const JString& s) const {
  JString retval;
  retval.rep = new char[length() + s.length() + 1];
  ::strcpy(retval.rep, rep);
  ::strcat(retval.rep, s.rep);
  retval.do_free = 1;
  return retval;
}

void JString::upcase() {
  char *s = rep;
  while (*s != '\0') {
    *s=toupper(*s);
    s++;
  }
}

void JString::to_camel() {
  char *s = rep;
  int have_first = 0;
  while (*s) {
    if (have_first) {
      *s = tolower(*s);
    } else {
      if (isalpha(*s)) {
        *s = toupper(*s);
        have_first = 1;
      }
    }
    s++;
  }
}
 
int JString::gsub(const char *pat, const char *repl) {
  char *found;
  int patn = strlen(pat);
  int repln = strlen(repl);
  int ind = 0;
  int nreplace = 0;
  while ((found = strstr(rep+ind, pat)) != NULL) {
    int loc = found - rep;
    if (repln > patn) {
      char *tmp = new char[length() + repln + 1];
      strcpy(tmp, rep);
      if (do_free) delete [] rep;
      rep = tmp;
      found = rep+loc;
      do_free = 1;
    }
    memmove(found+repln, found+patn, strlen(found+patn)+1);
    memcpy(found, repl, repln);
    ind = loc + repln;
    nreplace++;
  }
  return nreplace;
}

void JString::chop(int n) {
  for (int i=length()-1; n > 0 && i >= 0; --n)
    rep[i] = '\0';
}

