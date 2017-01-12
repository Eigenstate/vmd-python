
#if defined(NAMD_TCL) || ! defined(NAMD_VERSION)

#include <tcl.h>

extern int Psfgen_Init(Tcl_Interp *);

int main(int argc, char *argv[]) {
  Tcl_Main(argc, argv, Psfgen_Init);
  return 0;
}

#ifdef NAMD_VERSION
/* 
 * Provide user feedback and warnings beyond result values.
 * If we are running interactively, Tcl_Main will take care of echoing results
 * to the console.  If we run a script, we need to output the results
 * ourselves.
 */
void newhandle_msg(void *v, const char *msg) {
  Tcl_Interp *interp = (Tcl_Interp *)v;
  const char *words[3] = {"puts", "-nonewline", "psfgen) "};
  char *script = NULL;
  
  // prepend "psfgen) " to all output 
  script = Tcl_Merge(3, words);
  Tcl_Eval(interp,script); 
  Tcl_Free(script);

  // emit the output
  words[1] = msg;
  script = Tcl_Merge(2, words);
  Tcl_Eval(interp,script);
  Tcl_Free(script);
}

/*
 * Same as above but allow user control over prepending of "psfgen) "
 * and newlines.
 */
void newhandle_msg_ex(void *v, const char *msg, int prepend, int newline) {
  Tcl_Interp *interp = (Tcl_Interp *)v;
  const char *words[3] = {"puts", "-nonewline", "psfgen) "};
  char *script = NULL;
  
  if (prepend) { 
    // prepend "psfgen) " to all output
    script = Tcl_Merge(3, words);
    Tcl_Eval(interp,script);
    Tcl_Free(script);  
  } 
  
  // emit the output
  if (newline) {
    words[1] = msg;
    script = Tcl_Merge(2, words);
  } else {
    words[2] = msg;
    script = Tcl_Merge(3, words);
  }
  Tcl_Eval(interp,script);
  Tcl_Free(script);
} 
#endif

#else

#include <stdio.h>

int main(int argc, char **argv) {
  fprintf(stderr,"%s unavailable on this platform (no Tcl)\n",argv[0]);
  exit(-1);
}

#endif

