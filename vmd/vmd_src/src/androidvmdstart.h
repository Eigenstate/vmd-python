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
 *      $RCSfile: androidvmdstart.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $        $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Prototypes for Android startup code
 ***************************************************************************/

extern "C" {
  void log_android(const char *prompt, const char * msg);
}

