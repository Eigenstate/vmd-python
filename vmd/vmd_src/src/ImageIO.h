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
 *      $RCSfile: ImageIO.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $       $Date: 2019/01/17 21:20:59 $
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

/// copy/convert a 32-bit RGBA color buffer to a 24-bit RGB color buffer
unsigned char * cvt_rgb4u_rgb3u(const unsigned char * rgb4u, int xs, int ys);

/// copy/convert a 128-bit RGBA float color buffer to a 24-bit RGB color buffer
unsigned char * cvt_rgb4f_rgb3u(const float *rgb4f, int xs, int ys);

/// Write an unsigned RGB 24-bit color image, by filename extension
int write_image_file_rgb3u(const char *filename,
                           const unsigned char *rgb3u, int xs, int ys);

/// Write an unsigned RGBA 32-bit color image, by filename extension
int write_image_file_rgb4u(const char *filename,
                           const unsigned char *rgb4u, int xs, int ys);

/// Write an unsigned RGBA 32-bit color image, with alpha channel, by 
/// filename extension
int write_image_file_rgba4u(const char *filename,
                            const unsigned char *rgba4u, int xs, int ys);

/// Write an float RGBA 128-bit color image, by filename extension
int write_image_file_rgb4f(const char *filename,
                           const float *rgb4f, int xs, int ys);

/// Write an float RGBA 128-bit color image, with alpha channel,
/// by filename extension
int write_image_file_rgba4f(const char *filename,
                            const float *rgb4f, int xs, int ys);

/// Write 24-bit uncompressed SGI RGB image file
void vmd_writergb(FILE *dfile, const unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed Windows Bitmap file
void vmd_writebmp(FILE *dfile, const unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed NetPBM Portable Pixmap file
void vmd_writeppm(FILE *dfile, const unsigned char * img, int xs, int ys);

/// Write 24-bit uncompressed Truevision "Targa" file
void vmd_writetga(FILE *dfile, const unsigned char * img, int xs, int ys);

#if defined(VMDLIBPNG)
/// Write 24-bit RGB compressed PNG file
void vmd_writepng(FILE *dfile, const unsigned char * img, int xs, int ys);

/// Write 32-bit RGBA compressed PNG file
void vmd_writepng_alpha(FILE *dfile, const unsigned char * img, int xs, int ys);
#endif

#endif
