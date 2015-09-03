/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: SnapshotDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.44 $	$Date: 2011/03/02 18:30:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Dump the screen shot to a file by calling the machine specific external
 * program with the right options.  (Actually, someday the routines might
 * be put here as well; your preference.)
 *
 ***************************************************************************/

// This turned out to be mroe painless than I expected
// Basically, I already know the screen parameters since I'm a DisplayDevice,
// and I got the info from "display".  It is just a matter of constructing
// the correct string.

#include <stdio.h>
#include <stdlib.h>
#include "ImageIO.h"
#include "SnapshotDisplayDevice.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"

#if defined(_MSC_VER) || defined(WIN32)
#define DEF_SNAPSHOT_FILENAME "vmdscene.bmp"
#else
#define DEF_SNAPSHOT_FILENAME "vmdscene.tga"
#endif

SnapshotDisplayDevice::SnapshotDisplayDevice(DisplayDevice *d) 
: FileRenderer("snapshot", "Snapshot (VMD OpenGL window)", DEF_SNAPSHOT_FILENAME, DEF_VMDIMAGEVIEWER) {
  display = d; 
  // override default command line if environment variable set
  const char *envtxt = getenv("VMDIMAGEVIEWER");
  if (envtxt) {
    delete [] defaultCommandLine;
    defaultCommandLine = stringdup(envtxt);
    set_exec_string(envtxt); // change current exec command as well
  }
}

int SnapshotDisplayDevice::open_file(const char *filename) {
  if ((outfile = fopen(filename, "wb")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return FALSE;
  }
  my_filename = stringdup(filename);
  isOpened = TRUE;
  return TRUE;
}

static int checkfileextension(const char * s, const char * extension) {
  int sz, extsz; 
  sz = strlen(s);
  extsz = strlen(extension); 

  if (extsz > sz)
    return 0;

  if (!strupncmp(s + (sz - extsz), extension, extsz)) {
    return 1;
  }

  return 0;
}


// construct the exec string, then system() it
// pretty easy, eh?
void SnapshotDisplayDevice::close_file(void) {
  int xs=0, ys=0;
  unsigned char * img;

  img = display->readpixels(xs, ys);

  // write the image to a file on disk
  if (checkfileextension(my_filename, ".bmp")) {
    vmd_writebmp(outfile, img, xs, ys);
#if defined(VMDPNG)
  } else if (checkfileextension(my_filename, ".png")) {
    vmd_writepng(outfile, img, xs, ys);
#endif
  } else if (checkfileextension(my_filename, ".ppm")) {
    vmd_writeppm(outfile, img, xs, ys);
  } else if (checkfileextension(my_filename, ".rgb")) {
    vmd_writergb(outfile, img, xs, ys);
  } else if (checkfileextension(my_filename, ".tga")) {
    vmd_writetga(outfile, img, xs, ys);
  } else {
#if defined(_MSC_VER) || defined(WIN32)
    msgErr << "Unrecognized image file extension, writing Windows Bitmap file." 
           << sendmsg;
    vmd_writebmp(outfile, img, xs, ys);
#else
    msgErr << "Unrecognized image file extension, writing Targa file." 
           << sendmsg;
    vmd_writetga(outfile, img, xs, ys);
#endif
  }

  free(img); // free img memory block
  fclose(outfile);
  outfile = NULL;
  delete [] my_filename;
  my_filename = NULL;
  isOpened = FALSE;
}

