%module TCL_Libbiokit
%{
#include "ShortIntList.h"
#include "alphabet.h"
#include "alignedSequence.h"
#include "tcl_libbiokit.h"
%}

const char* seq(const char *arg1=NULL, const char *arg2=NULL, const char *arg3=NULL, const char *arg4=NULL, const char *arg5=NULL, const char *arg6=NULL);
