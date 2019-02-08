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
 *      $RCSfile: frame_selector.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Fltk dialogs for selecting/deleting ranges of frames
 ***************************************************************************/

#ifndef FRAME_SELECTOR_H
#define FRAME_SELECTOR_H

// Pop up a modal Fltk dialog box to select a range of frames.   Return 0
// if cancelled, or 1 on success.  On success, the parameters first, last,
// and stride will be filled in with the values the user selected.
// If max is unknown, use -1 to set no limit.
// Set first, last, and stride to the desired initial values.
extern int frame_selector(const char *wintitle, const char *moleculename, 
                          int maxframe, int *first, int *last, int *stride); 
                          
// A frame selector with a different appearance and text.
int frame_delete_selector (const char *moleculename, int maxframe, int *first, int *last, int *stride); 

#endif
