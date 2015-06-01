/*****************************************************************************
*
*            (C) Copyright 1995-2005 The Board of Trustees of the
*                        University of Illinois
*                         All Rights Reserved
*
******************************************************************************/


#ifndef STRUCTQR_H
#define STRUCTQR_H

#include "version.h"
//#include "structureAlignment.h"

class StructureAlignment;

int parseArgs(int argc, char** argv);
void printUsage(int argc, char** argv);
void printCopyright(int argc, char** argv);
StructureAlignment* readStructureAlignment(char* fastaFilename, char* inputDir);

#endif
