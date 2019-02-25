
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
 *	$RCSfile: Inform.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.43 $	$Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Inform - takes messages and displays them to the given ostream.
 *
 ***************************************************************************/

#include "Inform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "config.h"
#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif
#ifdef ANDROID
#include "androidvmdstart.h"
#endif

#if defined(VMDTKCON)
// XXX global instances of the Inform class
Inform msgInfo("Info) ",    VMDCON_INFO);
Inform msgWarn("Warning) ", VMDCON_WARN);
Inform msgErr("ERROR) ",    VMDCON_ERROR);
#else
// XXX global instances of the Inform class
Inform msgInfo("Info) ");
Inform msgWarn("Warning) ");
Inform msgErr("ERROR) ");
#endif

Inform& sendmsg(Inform& inform) { 
  Inform& rc = inform.send(); 

#if defined(VMDTKCON)
  vmdcon_purge();
#else
  fflush(stdout); // force standard output to be flushed here, otherwise output
                  // from Inform, stdio, Tcl, and Python can be weirdly 
                  // buffered, resulting in jumbled output from batch runs
#endif
  return rc;
}

Inform& ends(Inform& inform)    { return inform; }

#if defined(VMDTKCON)  
Inform::Inform(const char *myname, int lvl) {
  name = strdup(myname);
  loglvl=lvl;
  muted=0;
  reset();
}
#else
Inform::Inform(const char *myname) {
  name = strdup(myname);
  muted=0;
  reset();
}
#endif

Inform::~Inform() {
  free(name);
}

Inform& Inform::send() {
#if defined(VMD_SHARED) && !defined(PYDEBUG)
    // Don't emit output to stdout if running as shared library w/o debug
  return *this;
#endif
  char *nlptr, *bufptr;
 
  if (!muted) {
    bufptr = buf;
    if (!strchr(buf, '\n'))
      strcat(buf, "\n");

    while ((nlptr = strchr(bufptr, '\n'))) {
      *nlptr = '\0';
#if defined(VMDTKCON)
      vmdcon_append(loglvl, name, -1);
      vmdcon_append(loglvl, bufptr, -1);
      vmdcon_append(loglvl, "\n", 1);
#else
#ifdef ANDROID
      log_android(name, bufptr);
#else
      printf("%s%s\n", name, bufptr);
#endif
#endif
      bufptr = nlptr + 1; 
    }  
  }

  buf[0] = '\0';     
  return *this;
}

Inform& Inform::reset() {
  memset(buf, 0, sizeof(buf));
  memset(tmpbuf, 0, sizeof(tmpbuf));
  return *this;
}

Inform& Inform::operator<<(const char *s) {
  strncat(buf, s, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(char c) {
  tmpbuf[0] = c;
  tmpbuf[1] = '\0';
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(int i) {
  sprintf(tmpbuf, "%d", i);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(unsigned int i) {
  sprintf(tmpbuf, "%d", i);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(long i) {
  sprintf(tmpbuf, "%ld", i);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(unsigned long u) {
  sprintf(tmpbuf, "%ld", u);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(double d) {
  sprintf(tmpbuf, "%f", d);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(Inform& (*f)(Inform &)) {
  return f(*this);
}

#ifdef TEST_INFORM

int main() {
  msgInfo << "1\n";
  msgInfo << "12\n";
  msgInfo << "123\n";
  msgInfo << sendmsg;
  msgInfo << "6789";
  msgInfo << sendmsg;
  return 0;
}

#endif
 
