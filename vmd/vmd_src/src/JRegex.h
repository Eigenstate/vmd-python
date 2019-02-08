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
 *      $RCSfile: JRegex.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.15 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Regular expression matching interface.
 ***************************************************************************/

#ifndef J_REGEX_H__
#define J_REGEX_H__
#include "pcre.h"

/// Regular expression matching interface
class JRegex {
public:
  /// constructor takes an optional second argument; set to 1 if you're going 
  /// to use this pattern several times and want to optimize it.
  JRegex(const char *pattern, int fast=0);
  ~JRegex();

  /// Check for a match in str.  Returns -1 for no match.
  int match(const char *str, int len) const;

  /// Search for the first match, starting at str+start.  Returns the offset
  /// into the string where the match begins.  The match has length length.  
  /// If no match was found, returns -1.
  int search(const char *str, int len, int &length, int start=0);  

private:
  JRegex(const JRegex&) {}
  pcre *rpat; 
};

#endif
