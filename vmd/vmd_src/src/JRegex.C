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
 *      $RCSfile: JRegex.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.14 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Interface for performing regular expression pattern matching, 
 *  encapsulating the PCRE regular expression package.
 ***************************************************************************/

#include "JRegex.h"
#include "Inform.h" 

JRegex::JRegex(const char *pattern, int) {
  if (pattern == NULL) {
    msgErr << "NULL pattern passed to JRegex!" << sendmsg;
  }
  else {
    const char *errptr;
    int erroffset;
    rpat = vmdpcre_compile(pattern, // the regex pattern string
                        0,       // options
                        &errptr, // points to error message, if any        
                        &erroffset, // offset into line where error was found 
                        NULL);      // Table pointer; NULL for use default
    if (rpat == NULL) {
      msgWarn << "JRegex: Error in pcre_compile, " << errptr << sendmsg;
      msgWarn << "Error in regex pattern begins with " << pattern+erroffset
              << sendmsg;
    }
  }
}

JRegex::~JRegex() {
  vmdpcre_free(rpat);
}

int JRegex::match(const char *str, int len) const {
  if (rpat==NULL) {
//  msgWarn << "JRegex::match: bad regex pattern, no match" << sendmsg;
    return -1;
  } 
  int retval;
  retval=vmdpcre_exec(rpat,   // my regex pattern
                  NULL,   // No extra study wisdom
                  str,    // subject of the search
                  len,    // strlen of str
                  0,      // offset at which to start finding substrings
                  0,      // options
                  NULL,   // return vector for location of substrings
                  0);     // size of return vector
  return retval;
}

int JRegex::search(const char *str, int len, int &length, int start) {
  if (rpat==NULL) {
//  msgWarn << "JRegex::search: bad regex pattern, no match" << sendmsg;
    return -1;
  } 
  int ovec[6], retval;
  retval=vmdpcre_exec(rpat,  // my regex pattern
                  NULL,   // No extra study wisdom
                  str,    // subject of the search
                  len,    // strlen of str
                  start,  // offset at which to start finding substrings
                  0,      // options
                  ovec,   // return vector for location of substrings
                  6);     // size of return vector
  if (retval < 0) return retval;
  length = ovec[1]-ovec[0]; 
  return ovec[0]; 
}

