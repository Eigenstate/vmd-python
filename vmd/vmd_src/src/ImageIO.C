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
 *	$RCSfile: ImageIO.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.25 $	$Date: 2019/01/17 21:20:59 $
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ImageIO.h"
#include "Inform.h"
#include "utilities.h"

#if defined(VMDLIBPNG)
#include "png.h" // libpng header file
#endif

static void putbyte(FILE * outf, unsigned char val) {
  unsigned char buf[1];
  buf[0] = val;  
  fwrite(buf, 1, 1, outf);
}

static void putshort(FILE * outf, unsigned short val) {
  unsigned char buf[2];
  buf[0] = val >> 8;  
  buf[1] = val & 0xff;  
  fwrite(buf, 2, 1, outf);
}

static void putint(FILE * outf, unsigned int val) {
  unsigned char buf[4];
  buf[0] = (unsigned char) (val >> 24);  
  buf[1] = (unsigned char) (val >> 16);  
  buf[2] = (unsigned char) (val >>  8);  
  buf[3] = (unsigned char) (val & 0xff);  
  fwrite(buf, 4, 1, outf);
}


unsigned char * cvt_rgb4u_rgb3u(const unsigned char * rgb4u, int xs, int ys) {
  int rowlen3u = xs*3;
  int sz = xs * ys * 3;
  unsigned char * rgb3u = (unsigned char *) calloc(1, sz);

  int x3u, x4f, y;
  for (y=0; y<ys; y++) {
    int addr3u = y * xs * 3;
    int addr4f = y * xs * 4;
    for (x3u=0,x4f=0; x3u<rowlen3u; x3u+=3,x4f+=4) {
      rgb3u[addr3u + x3u    ] = rgb4u[addr4f + x4f    ];
      rgb3u[addr3u + x3u + 1] = rgb4u[addr4f + x4f + 1];
      rgb3u[addr3u + x3u + 2] = rgb4u[addr4f + x4f + 2];
    }
  }

  return rgb3u;
}


unsigned char * cvt_rgb4f_rgb3u(const float * rgb4f, int xs, int ys) {
  int rowlen3u = xs*3;
  int sz = xs * ys * 3;
  unsigned char * rgb3u = (unsigned char *) calloc(1, sz);

  int x3u, x4f, y;
  for (y=0; y<ys; y++) {
    int addr3u = y * xs * 3;
    int addr4f = y * xs * 4;
    for (x3u=0,x4f=0; x3u<rowlen3u; x3u+=3,x4f+=4) {
      int tmp;

      tmp = int(rgb4f[addr4f + x4f    ] * 255.0f);
      rgb3u[addr3u + x3u    ] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = int(rgb4f[addr4f + x4f + 1] * 255.0f);
      rgb3u[addr3u + x3u + 1] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = int(rgb4f[addr4f + x4f + 2] * 255.0f);
      rgb3u[addr3u + x3u + 2] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);
    }
  }

  return rgb3u;
}


unsigned char * cvt_rgba4f_rgba4u(const float * rgba4f, int xs, int ys) {
  int rowlen4u = xs*4;
  int sz = xs * ys * 4;
  unsigned char * rgba4u = (unsigned char *) calloc(1, sz);

  int x4u, x4f, y;
  for (y=0; y<ys; y++) {
    int addr4 = y * xs * 4;
    for (x4u=0,x4f=0; x4u<rowlen4u; x4u+=4,x4f+=4) {
      int tmp;

      tmp = int(rgba4f[addr4 + x4f    ] * 255.0f);
      rgba4u[addr4 + x4u    ] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = int(rgba4f[addr4 + x4f + 1] * 255.0f);
      rgba4u[addr4 + x4u + 1] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = int(rgba4f[addr4 + x4f + 2] * 255.0f);
      rgba4u[addr4 + x4u + 2] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);

      tmp = int(rgba4f[addr4 + x4f + 3] * 255.0f);
      rgba4u[addr4 + x4u + 3] = (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);
    }
  }

  return rgba4u;
}


static int checkfileextension(const char *s, const char *extension) {
  int sz, extsz;
  sz = strlen(s);
  extsz = strlen(extension);

  if (extsz > sz) return 0;

  if (!strupncmp(s + (sz - extsz), extension, extsz)) return 1;

  return 0;
}


int write_image_file_rgb3u(const char *filename,
                           const unsigned char *rgb3u, int xs, int ys) {
  FILE *outfile=NULL;
  if ((outfile = fopen(filename, "wb")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return -1;
  }

  // write the image to a file on disk
  if (checkfileextension(filename, ".bmp")) {
    vmd_writebmp(outfile, rgb3u, xs, ys);
#if defined(VMDLIBPNG)
  } else if (checkfileextension(filename, ".png")) {
    vmd_writepng(outfile, rgb3u, xs, ys);
#endif
  } else if (checkfileextension(filename, ".ppm")) {
    vmd_writeppm(outfile, rgb3u, xs, ys);
  } else if (checkfileextension(filename, ".rgb")) {
    vmd_writergb(outfile, rgb3u, xs, ys);
  } else if (checkfileextension(filename, ".tga")) {
    vmd_writetga(outfile, rgb3u, xs, ys);
  } else {
#if defined(_MSC_VER) || defined(WIN32)
    msgErr << "Unrecognized image file extension, writing Windows Bitmap file."
           << sendmsg;
    vmd_writebmp(outfile, rgb3u, xs, ys);
#else
    msgErr << "Unrecognized image file extension, writing Targa file."
           << sendmsg;
    vmd_writetga(outfile, rgb3u, xs, ys);
#endif
  }

  fclose(outfile);
  return 0;
}


int write_image_file_rgb4u(const char *filename,
                           const unsigned char *rgb4u, int xs, int ys) {
  unsigned char *rgb3u = cvt_rgb4u_rgb3u(rgb4u, xs, ys);
  if (rgb3u == NULL)
    return -1;

  if (write_image_file_rgb3u(filename, rgb3u, xs, ys)) {
    free(rgb3u);
    return -1;
  }

  free(rgb3u);
  return 0;
}


int write_image_file_rgb4f(const char *filename,
                           const float *rgb4f, int xs, int ys) {
  unsigned char *rgb3u = cvt_rgb4f_rgb3u(rgb4f, xs, ys);
  if (rgb3u == NULL)
    return -1;

  if (write_image_file_rgb3u(filename, rgb3u, xs, ys)) {
    free(rgb3u);
    return -1;
  }

  free(rgb3u);
  return 0;
}


int write_image_file_rgba4u(const char *filename,
                            const unsigned char *rgba4u, int xs, int ys) {
  FILE *outfile=NULL;
  if ((outfile = fopen(filename, "wb")) == NULL) {
    msgErr << "Could not open file " << filename
           << " in current directory for writing!" << sendmsg;
    return -1;
  }

  if (checkfileextension(filename, ".png")) {
#if defined(VMDLIBPNG)
    vmd_writepng_alpha(outfile, rgba4u, xs, ys);
#else
    msgErr << "Unrecognized or unsupported alpha-channel image format."
           << sendmsg;
    return -1;
#endif
  } else {
    msgErr << "Unrecognized or unsupported alpha-channel image format."
           << sendmsg;
    return -1;
  }

  fclose(outfile);
  return 0;
}


int write_image_file_rgba4f(const char *filename,
                            const float *rgba4f, int xs, int ys) {
  unsigned char *rgba4u = cvt_rgba4f_rgba4u(rgba4f, xs, ys);
  if (rgba4u == NULL)
    return -1;

  if (write_image_file_rgba4u(filename, rgba4u, xs, ys)) {
    free(rgba4u);
    return -1;
  }

  free(rgba4u);
  return 0;
}


void vmd_writergb(FILE *dfile, const unsigned char * img, int xs, int ys) {
  char iname[80];               /* Image name */
  int x, y, i;

  if (img == NULL) 
    return;

  putshort(dfile, 474);         /* Magic                       */
  putbyte(dfile, 0);            /* STORAGE is VERBATIM         */
  putbyte(dfile, 1);            /* BPC is 1                    */
  putshort(dfile, 3);           /* DIMENSION is 3              */
  putshort(dfile, xs);          /* XSIZE                       */
  putshort(dfile, ys);          /* YSIZE                       */
  putshort(dfile, 3);           /* ZSIZE                       */
  putint(dfile, 0);             /* PIXMIN is 0                 */
  putint(dfile, 255);           /* PIXMAX is 255               */

  for(i=0; i<4; i++)            /* DUMMY 4 bytes               */
    putbyte(dfile, 0);

  strcpy(iname, "VMD Snapshot");
  fwrite(iname, 80, 1, dfile);  /* IMAGENAME                   */
  putint(dfile, 0);             /* COLORMAP is 0               */
  for(i=0; i<404; i++)          /* DUMMY 404 bytes             */
    putbyte(dfile,0);

  for(i=0; i<3; i++)
    for(y=0; y<ys; y++)
      for(x=0; x<xs; x++)
        fwrite(&img[(y*xs + x)*3 + i], 1, 1, dfile);
}

static void write_le_int32(FILE * dfile, int num) {
  fputc((num      ) & 0xFF, dfile);
  fputc((num >> 8 ) & 0xFF, dfile);
  fputc((num >> 16) & 0xFF, dfile);
  fputc((num >> 24) & 0xFF, dfile);
}

static void write_le_int16(FILE * dfile, int num) {
  fputc((num      ) & 0xFF, dfile);
  fputc((num >> 8 ) & 0xFF, dfile);
}


void vmd_writebmp(FILE *dfile, const unsigned char * img, int xs, int ys) {
  if (img != NULL) {
      int imgdataoffset = 14 + 40;     // file header size + bitmap header size
      int rowlen = xs * 3;             // non-padded length of row of pixels
      int rowsz = ((rowlen) + 3) & -4; // size of one padded row of pixels
      int imgdatasize = rowsz * ys;    // size of image data
      int filesize = imgdataoffset + imgdatasize;

      // write out bitmap file header (14 bytes)
      fputc('B', dfile); 
      fputc('M', dfile);
      write_le_int32(dfile, filesize);
      write_le_int16(dfile, 0);
      write_le_int16(dfile, 0);
      write_le_int32(dfile, imgdataoffset);

      // write out bitmap header (40 bytes)
      write_le_int32(dfile, 40); // size of bitmap header structure
      write_le_int32(dfile, xs); // size of image in x
      write_le_int32(dfile, ys); // size of image in y
      write_le_int16(dfile, 1);  // number of color planes (only "1" is legal)
      write_le_int16(dfile, 24); // bits per pixel

      // fields added in Win 3.x
      write_le_int32(dfile, 0);           // compression used (0 == none)
      write_le_int32(dfile, imgdatasize); // size of bitmap in bytes 

      // imported improvements from the Tachyon BMP writer to address 
      // the behavior of BMP files loaded for display on Android devices
      write_le_int32(dfile, 11811);       // X pixels per meter (300dpi)
      write_le_int32(dfile, 11811);       // Y pixels per meter (300dpi)
      write_le_int32(dfile, 0);           // color count (0 for RGB)
      write_le_int32(dfile, 0);           // important colors (0 for RGB)
       
      // write out actual image data
      int i, y;
      unsigned char * rowbuf = (unsigned char *) malloc(rowsz);
      if (rowbuf != NULL) { 
        memset(rowbuf, 0, rowsz); // clear the buffer (and padding) to black.

        for (y=0; y<ys; y++) {
          int addr = xs * 3 * y;

          // write one row of the image, in reversed RGB -> BGR pixel order
          // padding bytes should remain 0's, shouldn't have to re-clear them.
          for (i=0; i<rowlen; i+=3) {
            rowbuf[i    ] = img[addr + i + 2]; // blue
            rowbuf[i + 1] = img[addr + i + 1]; // green
            rowbuf[i + 2] = img[addr + i    ]; // red 
          }

          fwrite(rowbuf, rowsz, 1, dfile); // write the whole row of pixels 
        }
        free(rowbuf); 
      } else {
        msgErr << "Failed to save snapshot image!" << sendmsg;
      }

  }  // img != NULL
}


void vmd_writeppm(FILE *dfile, const unsigned char * img, int xs, int ys) {
  if (img != NULL) {
    int y;

    fprintf(dfile,"%s\n","P6");
    fprintf(dfile,"%d\n", xs);
    fprintf(dfile,"%d\n", ys);
    fprintf(dfile,"%d\n",255); /* maxval */
  
    for (y=(ys - 1); y>=0; y--) {
      fwrite(&img[xs * 3 * y], 1, (xs * 3), dfile);
    }
  }
}


void vmd_writetga(FILE *dfile, const unsigned char * img, int xs, int ys) {
  int x, y;

  const unsigned char * bufpos;
  int filepos, numbytes;
  unsigned char * fixbuf;

  fputc(0, dfile); /* IdLength      */
  fputc(0, dfile); /* ColorMapType  */
  fputc(2, dfile); /* ImageTypeCode */
  fputc(0, dfile); /* ColorMapOrigin, low byte */
  fputc(0, dfile); /* ColorMapOrigin, high byte */
  fputc(0, dfile); /* ColorMapLength, low byte */
  fputc(0, dfile); /* ColorMapLength, high byte */
  fputc(0, dfile); /* ColorMapEntrySize */
  fputc(0, dfile); /* XOrigin, low byte */
  fputc(0, dfile); /* XOrigin, high byte */
  fputc(0, dfile); /* YOrigin, low byte */
  fputc(0, dfile); /* YOrigin, high byte */
  fputc((xs & 0xff),         dfile); /* Width, low byte */
  fputc(((xs >> 8) & 0xff),  dfile); /* Width, high byte */
  fputc((ys & 0xff),         dfile); /* Height, low byte */
  fputc(((ys >> 8) & 0xff),  dfile); /* Height, high byte */
  fputc(24, dfile);   /* ImagePixelSize */
  fputc(0x20, dfile); /* ImageDescriptorByte 0x20 == flip vertically */

  fixbuf = (unsigned char *) malloc(xs * 3);
  if (fixbuf == NULL) {
    msgErr << "vmd_writetga: failed memory allocation!" << sendmsg;
    return;
  }

  for (y=0; y<ys; y++) {
    bufpos=img + (xs*3)*(ys-y-1);
    filepos=18 + xs*3*y;

    if (filepos >= 18) {
      fseek(dfile, filepos, 0);

      for (x=0; x<(3*xs); x+=3) {
        fixbuf[x    ] = bufpos[x + 2];
        fixbuf[x + 1] = bufpos[x + 1];
        fixbuf[x + 2] = bufpos[x    ];
      }

      numbytes = fwrite(fixbuf, 3, xs, dfile);

      if (numbytes != xs) {
        msgErr << "vmd_writetga: file write problem, " 
               << numbytes << " bytes written." << sendmsg;
      }
    }
    else {
      msgErr << "vmd_writetga: file ptr out of range!!!" << sendmsg;
      return;  /* don't try to continue */
    }
  }

  free(fixbuf);
}

#if defined(VMDLIBPNG)
void vmd_writepng(FILE *dfile, const unsigned char * img, int xs, int ys) {
  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep *row_pointers;
  png_textp text_ptr;
  int y;

  /* Create and initialize the png_struct with the default error handlers */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not initialize PNG library, return error */
  }

  /* Allocate/initialize the memory for image information.  REQUIRED. */
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not initialize PNG library, return error */
  }

  /* Set error handling for setjmp/longjmp method of libpng error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    /* Free all of the memory associated with the png_ptr and info_ptr */
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    /* If we get here, we had a problem writing the file */
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not open image, return error */
  }

  /* Set up the input control if you are using standard C streams */
  png_init_io(png_ptr, dfile);

  png_set_IHDR(png_ptr, info_ptr, xs, ys,
               8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_set_gAMA(png_ptr, info_ptr, 1.0);

  text_ptr = (png_textp) png_malloc(png_ptr, (png_uint_32)sizeof(png_text) * 2);

  text_ptr[0].key = (char *) "Description";
  text_ptr[0].text = (char *) "A molecular scene rendered by VMD";
  text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE;
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[0].lang = NULL;
#endif

  text_ptr[1].key = (char *) "Software";
  text_ptr[1].text = (char *) "VMD -- Visual Molecular Dynamics";
  text_ptr[1].compression = PNG_TEXT_COMPRESSION_NONE;
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[1].lang = NULL;
#endif
  png_set_text(png_ptr, info_ptr, text_ptr, 1);

  row_pointers = (png_bytep *) png_malloc(png_ptr, ys*sizeof(png_bytep));
  for (y=0; y<ys; y++) {
    row_pointers[ys - y - 1] = (png_bytep) &img[y * xs * 3];
  }

  png_set_rows(png_ptr, info_ptr, row_pointers);

  /* one-shot call to write the whole PNG file into memory */
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  png_free(png_ptr, row_pointers);
  png_free(png_ptr, text_ptr);

  /* clean up after the write and free any memory allocated - REQUIRED */
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

  return; /* No fatal errors */
}


void vmd_writepng_alpha(FILE *dfile, const unsigned char * img, int xs, int ys) {
  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep *row_pointers;
  png_textp text_ptr;
  int y;

  /* Create and initialize the png_struct with the default error handlers */
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not initialize PNG library, return error */
  }

  /* Allocate/initialize the memory for image information.  REQUIRED. */
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not initialize PNG library, return error */
  }

  /* Set error handling for setjmp/longjmp method of libpng error handling */
  if (setjmp(png_jmpbuf(png_ptr))) {
    /* Free all of the memory associated with the png_ptr and info_ptr */
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    /* If we get here, we had a problem writing the file */
    msgErr << "Failed to write PNG file" << sendmsg;
    return; /* Could not open image, return error */
  }

  /* Set up the input control if you are using standard C streams */
  png_init_io(png_ptr, dfile);

  png_set_IHDR(png_ptr, info_ptr, xs, ys,
               8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

//  png_set_alpha_mode(png_ptr, PNG_ALPHA_STANDARD, PNG_GAMMA_LINEAR);

#if 0
  /* optional significant bit chunk */
  png_color_8 sig_bit;
  memset(&sig_bit, 0, sizeof(sig_bit));

  sig_bit.red = 8;
  sig_bit.green = 8;
  sig_bit.blue = 8;
  sig_bit.alpha = 8;

  png_set_sBIT(png_ptr, info_ptr, &sig_bit);
#endif

  png_set_gAMA(png_ptr, info_ptr, 1.0);

  text_ptr = (png_textp) png_malloc(png_ptr, (png_uint_32)sizeof(png_text) * 2);

  text_ptr[0].key = (char *) "Description";
  text_ptr[0].text = (char *) "A molecular scene rendered by VMD";
  text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE;
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[0].lang = NULL;
#endif

  text_ptr[1].key = (char *) "Software";
  text_ptr[1].text = (char *) "VMD -- Visual Molecular Dynamics";
  text_ptr[1].compression = PNG_TEXT_COMPRESSION_NONE;
#ifdef PNG_iTXt_SUPPORTED
  text_ptr[1].lang = NULL;
#endif
  png_set_text(png_ptr, info_ptr, text_ptr, 1);

//  png_write_info(png_ptr, info_ptr);

  row_pointers = (png_bytep *) png_malloc(png_ptr, ys*sizeof(png_bytep));
  for (y=0; y<ys; y++) {
    row_pointers[ys - y - 1] = (png_bytep) &img[y * xs * 4];
  }

  png_set_rows(png_ptr, info_ptr, row_pointers);

  /* one-shot call to write the whole PNG file into memory */
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  png_free(png_ptr, row_pointers);
  png_free(png_ptr, text_ptr);

  /* clean up after the write and free any memory allocated - REQUIRED */
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

  return; /* No fatal errors */
}

#endif
