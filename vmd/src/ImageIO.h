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
 *      $RCSfile: ImageIO.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $       $Date: 2010/12/16 04:08:20 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Write an RGB image to a file.  Image routines donated by John Stone,
 *   derived from Tachyon source code.  For now these image file writing
 *   routines are statically linked into VMD and are not even in an extensible
 *   list structure.  Long-term the renderer interface should abstract from
 *   most of the details, and use a plugin interface for extensibility.
 *   For the short-term, this gets the job done.
 *
 ***************************************************************************/

#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <stdio.h>
#include <stdlib.h>

/// Write 24-bit uncompressed SGI RGB image file
void vmd_writergb(FILE *dfile, unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed Windows Bitmap file
void vmd_writebmp(FILE *dfile, unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed NetPBM Portable Pixmap file
void vmd_writeppm(FILE *dfile, unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed Truevision "Targa" file
void vmd_writetga(FILE *dfile, unsigned char * img, int xs, int ys);

#if defined(VMDPNG)
/// Write 24-bit uncompressed PNG file
void vmd_writepng(FILE *dfile, unsigned char * img, int xs, int ys);
#endif

#endif
