/*****************************************************************************
*
*            (C) Copyright 1995-2005 The Board of Trustees of the
*                        University of Illinois
*                         All Rights Reserved
*
******************************************************************************/

/*****************************************************************************
* RCS INFORMATION:
*
*       $RCSfile: profcombine.h,v $
*       $Author: kvandivo $        $Locker:  $             $State: Exp $
*       $Revision: 1.1 $           $Date: 2009/03/31 17:00:33 $
*
******************************************************************************/

// $Id: profcombine.h,v 1.1 2009/03/31 17:00:33 kvandivo Exp $

#ifndef PROFCOMBINE_H
#define PROFCOMBINE_H

#ifndef VERSION
#define VERSION     "1.0.7" //Only used for WIN32, other platforms use automake version.
#endif

int parseArgs(int argc, char** argv);
void printUsage(int argc, char** argv);
void printCopyright(int argc, char** argv);

#endif
