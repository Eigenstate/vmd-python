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
 *      $RCSfile: JString.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $      $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A minimalistic string class we use instead of similar classes from the
 *   STL or the GNU libraries for better portability and greatly reduced
 *   code size.  (only implements the functionality we actually need, doesn't
 *   balloon the entire VMD binary as some past string class implementations
 *   did).  Implements regular expression matching methods used by VMD.
 ***************************************************************************/

#ifndef JSTRING_H__
#define JSTRING_H__

#include <string.h>

/// A minimalistic string class we use instead of similar classes from the
/// STL or the GNU libraries for better portability and greatly reduced
/// code size.  (only implements the functionality we actually need, doesn't
/// balloon the entire VMD binary as some past string class implementations
/// did).  Implements regular expression matching methods used by VMD.
class JString {
private:
  static char *defstr;
  char *rep;
  int do_free;

public:
  JString()
  : rep(defstr), do_free(0) {} 
    
  JString(const char *str)
  : rep(defstr), do_free(0) {
    if (str) {
      rep = new char[strlen(str)+1];
      strcpy(rep, str);
      do_free = 1;
    }
  }
  JString(const JString& s) {
    rep = new char[strlen(s.rep)+1];
    strcpy(rep, s.rep);
    do_free = 1;
  }
  ~JString() { if (do_free) delete [] rep; }

  int operator==(const char *s) {return !strcmp(s,rep);}
  int operator!=(const char *s) {return strcmp(s,rep);}
  int operator<(const char *s) {return (strcmp(s,rep)<0);}
  int operator>(const char *s) {return (strcmp(s,rep)>0);}
  int operator<=(const char *s) {return (strcmp(s,rep)<=0);}
  int operator>=(const char *s) {return (strcmp(s,rep)>=0);}

  JString& operator=(const char *);
  JString& operator=(const JString&);
  JString& operator=(const char);
  JString& operator+=(const char *);
  JString& operator+=(const JString&);
  JString& operator+=(const char);

  friend int compare(const JString& s1, const JString& s2) {
    return strcmp(s1.rep, s2.rep);
  }

  friend JString operator+(const char*, const JString&);
  JString operator+(const JString&) const;
 
  int length() const { return (int) strlen(rep); }

  operator const char *() const {return rep; }

  // convert to uppercase
  void upcase();

  // convert to camel case (only first letter capitalized)
  void to_camel();
 
  // Replace all instances of pat with repl
  int gsub(const char *pat, const char *repl);
  
  // remove the last n non-NULL characters from the end of the string
  void chop(int n);
};

#endif

