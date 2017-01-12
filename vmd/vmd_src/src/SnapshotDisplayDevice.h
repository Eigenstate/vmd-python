/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2016 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: SnapshotDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.25 $	$Date: 2016/11/28 03:05:04 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Dump the screen shot to a file by calling the DisplayDevice routine or
 *  machine specific external program with the right options. 
 *
 ***************************************************************************/


#ifndef SNAPSHOTDISPLAYDEVICE
#define SNAPSHOTDISPLAYDEVICE

#include "FileRenderer.h"

/// FileRenderer subclass to save VMD images in a supported image file format
class SnapshotDisplayDevice : public FileRenderer {
private:
  DisplayDevice *display;

public:
  /// set up the commands for grabbing images from the screen
  /// pass in display to grab image from
  SnapshotDisplayDevice(DisplayDevice *);
  virtual int open_file(const char *filename);   ///< open output
  virtual void render(const VMDDisplayList*) {}  ///< ignore renders
  virtual void close_file(void); ///< capture and save the image to a file
};  
#endif

