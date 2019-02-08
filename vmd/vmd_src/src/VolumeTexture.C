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
 *      $RCSfile: VolumeTexture.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.23 $      $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class for managing volumetric texture maps for use by the various
 *   DrawMolItem representation methods.
 ***************************************************************************/

#include "VolumeTexture.h"
#include "VolumetricData.h"
#include "AtomColor.h"
#include "Scene.h"
#include "VMDApp.h"
#include "Inform.h"

#include <math.h>
#include <stdio.h>

VolumeTexture::VolumeTexture()
: v(NULL), texmap(NULL), texid(0) {

  size[0] = size[1] = size[2] = 0;
}

VolumeTexture::~VolumeTexture() {
  if (texmap) vmd_dealloc(texmap);
}

void VolumeTexture::setGridData(VolumetricData *voldata) {
  if (v == voldata) return;
  v = voldata;
  if (texmap) {
    vmd_dealloc(texmap);
    texmap = NULL;
  }
  size[0] = size[1] = size[2] = 0;
  texid = 0;
}

int VolumeTexture::allocateTextureMap(long ntexels) {
  texid = 0;
  if (texmap) {
    vmd_dealloc(texmap);
    texmap = NULL;
  }
  long sz = ntexels*3L*sizeof(unsigned char);
  texmap = (unsigned char *) vmd_alloc(sz);
  if (texmap == NULL) {
    msgErr << "Texture map allocation failed, out of memory?" << sendmsg;
    msgErr << "Texture map texel count: " << ntexels << sendmsg;
    msgErr << "Failed allocation of size: " << sz / (1024L*1024L) 
           << "MB" <<sendmsg;
    return FALSE;
  }
  memset(texmap, 0, ntexels*3L*sizeof(unsigned char));
  texid = VMDApp::get_texserialnum();
  return TRUE;
}

void VolumeTexture::generatePosTexture() {
  // nice small texture that will work everywhere
  size[0] = size[1] = size[2] = 32; 
  int x, y, z;
  long addr, addr2;
  long num = num_texels();

  if (!allocateTextureMap(num)) return;

  for (z=0; z<size[2]; z++) {
    for (y=0; y<size[1]; y++) {
      addr = z * size[0] * size[1] + y * size[0];
      for (x=0; x<size[0]; x++) {
        addr2 = (addr + x) * 3L;
        texmap[addr2    ] = (unsigned char) (((float) x / (float) size[0]) * 255.0f);
        texmap[addr2 + 1] = (unsigned char) (((float) y / (float) size[1]) * 255.0f);
        texmap[addr2 + 2] = (unsigned char) (((float) z / (float) size[2]) * 255.0f);
      }
    }
  }
}

// convert Hue/Saturation/Value to RGB colors
static void HSItoRGB(float h, float s, float i, 
                     unsigned char *r, unsigned char *g, unsigned char *b) {
  float rv, gv, bv, t;

  t=float(VMD_TWOPI)*h;
  rv=(float) (1 + s*sinf(t - float(VMD_TWOPI)/3.0f));
  gv=(float) (1 + s*sinf(t));
  bv=(float) (1 + s*sinf(t + float(VMD_TWOPI)/3.0f));

  t=(float) (254.9999 * i / 2);
  
  *r=(unsigned char)(rv*t);
  *g=(unsigned char)(gv*t);
  *b=(unsigned char)(bv*t);
}

// return the smallest power of two greater than size, up to 2^16.
static int nextpower2(int size) {
  int i;
  int power;

  if (size == 1)
    return 1;

  power=1;
  for (i=0; i<16; i++) {
    power <<= 1;
    if (power >= size)
      return power;
  } 
 
  return power; 
}

void VolumeTexture::generateIndexTexture() {
  // nice small texture that will work everywhere
  size[0] = size[1] = size[2] = 32; 
  int x, y, z;
  long addr, addr2, addr3, index;
  long num = num_texels();
  unsigned char coltable[3 * 4096];

  if (!allocateTextureMap(num)) return;

  // build a fast color lookup table
  for (index=0; index<4096; index++) {
    addr = index * 3;
    HSItoRGB(8.0f * index / 4096.0f, 0.75, 1.0,
             coltable+addr, coltable+addr+1, coltable+addr+2);
  }

  for (z=0; z<size[2]; z++) {
    for (y=0; y<size[1]; y++) {
      addr = z * size[0] * size[1] + y * size[0];
      for (x=0; x<size[0]; x++) {
        index = addr + x;
        addr2 = index * 3;
        addr3 = ((int) ((index / (float) num) * 4095)) * 3;
        texmap[addr2    ] = coltable[addr3    ];
        texmap[addr2 + 1] = coltable[addr3 + 1];
        texmap[addr2 + 2] = coltable[addr3 + 2];
      }
    }
  }
}

void VolumeTexture::generateChargeTexture(float vmin, float vmax) {
  // need a volumetric dataset for this
  if (!v) return;

  int x, y, z;
  long addr, addr2;
  long daddr;
  float vscale, vrange;

  size[0] = v->xsize;
  size[1] = v->ysize;
  size[2] = v->zsize;
  for (int i=0; i<3; i++) {
    size[i] = nextpower2(size[i]);
  }
  long num = num_texels();
  if (!allocateTextureMap(num)) return;

  vrange = vmax - vmin;
  if (fabs(vrange) < 0.00001)
    vscale = 0.0f;
  else
    vscale = 1.00001f / vrange;

  // map volume data scalars to colors
  for (z=0; z<v->zsize; z++) {
    for (y=0; y<v->ysize; y++) {
       addr = z * size[0] * size[1] + y * size[0];
      daddr = z * v->xsize * v->ysize + y * v->xsize;
      for (x=0; x<v->xsize; x++) {
        addr2 = (addr + x) * 3;
        float level, r, g, b;

        // map data to range 0->1        
        level = (v->data[daddr + x] - vmin) * vscale; 
        level = level < 0 ? 0 :
                level > 1 ? 1 : level;

        // low values are mapped to red, high to blue
        r = (1.0f - level) * 255.0f;
        b = level * 255.0f;
        if (level < 0.5f) {
          g = level * 2.0f * 128.0f;
        } else {
          g = (0.5f - (level - 0.5f)) * 2.0f * 128.0f;
        }

        texmap[addr2    ] = (unsigned char) r;
        texmap[addr2 + 1] = (unsigned char) g;
        texmap[addr2 + 2] = (unsigned char) b;
      }
    }
  }
}

void VolumeTexture::generateHSVTexture(float vmin, float vmax) {
  int x, y, z;
  long index, addr, addr2, addr3;
  long daddr;
  float vscale, vrange;
  unsigned char coltable[3 * 4096];

  size[0] = v->xsize;
  size[1] = v->ysize;
  size[2] = v->zsize;
  for (int i=0; i<3; i++) {
    size[i] = nextpower2(size[i]);
  }
  long num = num_texels();
  if (!allocateTextureMap(num)) return;

  // build a fast color lookup table
  for (index=0; index<4096; index++) {
    addr = index * 3;
    HSItoRGB(4.0f * index / 4096.0f, 0.75, 1.0,
             coltable+addr, coltable+addr+1, coltable+addr+2);
  }

  // calculate scaling factors
  vrange = vmax - vmin;
  if (fabs(vrange) < 0.00001)
    vscale = 0.0f;
  else
    vscale = 1.00001f / vrange;

  // map volume data scalars to colors
  for (z=0; z<v->zsize; z++) {
    for (y=0; y<v->ysize; y++) {
       addr = z * size[0] * size[1] + y * size[0];
      daddr = z * v->xsize * v->ysize + y * v->xsize;
      for (x=0; x<v->xsize; x++) {
        addr2 = (addr + x) * 3;
        float level;

        // map data to range 0->1        
        level = (v->data[daddr + x] - vmin) * vscale; 

        // Conditional range test written in terms of inclusion
        // within the 0:1 range so that cases that encounter a level
        // value of NaN will be clamped to 0 rather than remaining NaN
        // and subsequently calculating an out-of-bounds color map address.
        // Writing the range test this way works because IEEE FP is defined
        // such that all comparisons vs. NaN return false.
        level = (level >= 0) ? ((level <= 1) ? level : 1) : 0;

        // map values to an HSV color map
        addr3 = ((int) (level * 4095)) * 3;
        texmap[addr2    ] = coltable[addr3    ];
        texmap[addr2 + 1] = coltable[addr3 + 1];
        texmap[addr2 + 2] = coltable[addr3 + 2];
      }
    }
  }
}

void VolumeTexture::generateColorScaleTexture(float vmin, float vmax, const Scene *scene) {

  int x, y, z;
  long addr, addr2;
  long daddr;
  float vscale, vrange;

  size[0] = v->xsize;
  size[1] = v->ysize;
  size[2] = v->zsize;
  for (int i=0; i<3; i++) {
    size[i] = nextpower2(size[i]);
  }
  long num = num_texels();
  if (!allocateTextureMap(num)) return;

  vrange = vmax - vmin;
  if (fabs(vrange) < 0.00001)
    vscale = 0.0f;
  else
    vscale = 1.00001f / vrange;

  // map volume data scalars to colors
  for (z=0; z<v->zsize; z++) {
    for (y=0; y<v->ysize; y++) {
       addr = z * size[0] * size[1] + y * size[0];
      daddr = z * v->xsize * v->ysize + y * v->xsize;
      for (x=0; x<v->xsize; x++) {
        addr2 = (addr + x) * 3;
        float level;

        // map data min/max to range 0->1
        // values must be clamped before use, since user-specified
        // min/max can cause out-of-range color indices to be generated
        level = (v->data[daddr + x] - vmin) * vscale; 

        int colindex = (int)(level * MAPCLRS-1);

        // This code isn't vulnerable to the effects of NaN inputs 
        // because the value clamping logic is performed in the integer
        // domain, so regardless what comes out of the colindex calculation,
        // it will be clamped to the legal color index range.
        if (colindex < 0) 
          colindex = 0;
        else if (colindex >= MAPCLRS) 
          colindex = MAPCLRS-1;

        const float *rgb = scene->color_value(MAPCOLOR(colindex));
        texmap[addr2    ] = (unsigned char)(rgb[0]*255.0f);
        texmap[addr2 + 1] = (unsigned char)(rgb[1]*255.0f);
        texmap[addr2 + 2] = (unsigned char)(rgb[2]*255.0f);
      }
    }
  }
}

void VolumeTexture::generateContourLineTexture(float densityperline, float linewidth) {
  int x, y, z;
  long addr, addr2;
  float xp, yp, zp;

  float datamin, datamax;
  v->datarange(datamin, datamax);
printf("Contour lines...\n");
printf("range / densityperline: %f\n", log(datamax - datamin) / densityperline);

  size[0] = nextpower2(v->xsize*2);
  size[1] = nextpower2(v->ysize*2);
  size[2] = nextpower2(v->zsize*2);
  long num = num_texels();
  if (!allocateTextureMap(num)) return;

  // map volume data scalars to contour line colors
  for (z=0; z<size[2]; z++) {
    zp = ((float) z / size[2]) * v->zsize;
    for (y=0; y<size[1]; y++) {
      addr = z * size[0] * size[1] + y * size[0];
      yp = ((float) y / size[1]) * v->ysize;
      for (x=0; x<size[0]; x++) {
        addr2 = (addr + x) * 3;
        xp = ((float) x / size[0]) * v->xsize;
        float level;

        level = float(fmod(log(v->voxel_value_interpolate(xp,yp,zp)), densityperline) / densityperline);

        if (level < linewidth) {
          texmap[addr2    ] = 0;
          texmap[addr2 + 1] = 0;
          texmap[addr2 + 2] = 0;
        } else {        
          texmap[addr2    ] = 255;
          texmap[addr2 + 1] = 255;
          texmap[addr2 + 2] = 255;
        }
      }
    }
  }
}


void VolumeTexture::calculateTexgenPlanes(float v0[3], float v1[3], float v2[3], float v3[3]) const {

  int i;
  if (!texmap || !v) {
    // do something sensible
    vec_zero(v0);
    vec_zero(v1);
    vec_zero(v2);
    vec_zero(v3);
    v1[0] = v2[1] = v3[2] = 1;
    return;
  }

  // rescale texture coordinates by the portion of the 
  // entire texture volume they reference
  // XXX added an additional scale factor to keep "nearest" texture modes 
  //     rounding into the populated area rather than catching black
  //     texels in the empty part of the texture volume
  float tscale[3];
  tscale[0] = (v->xsize / (float)size[0]) * 0.99999f;
  tscale[1] = (v->ysize / (float)size[1]) * 0.99999f;
  tscale[2] = (v->zsize / (float)size[2]) * 0.99999f;

  // calculate length squared of volume axes
  float lensq[3];
  vec_zero(lensq);
  for (i=0; i<3; i++) {
    lensq[0] += float(v->xaxis[i] * v->xaxis[i]);
    lensq[1] += float(v->yaxis[i] * v->yaxis[i]);
    lensq[2] += float(v->zaxis[i] * v->zaxis[i]);
  }

  // Calculate reciprocal space lattice vectors, which are used
  // in the OpenGL texgen eye space plane equations in order to transform
  // incoming world coordinates to the correct texture coordinates.
  // This code should work for both orthogonal and non-orthogonal volumes.
  // The last step adds in the NPOT texture scaling where necessary.
  // Reference: Introductory Solid State Physics, H.P.Myers, page 43
  float xaxdir[3], yaxdir[3], zaxdir[3];
  float nxaxdir[3], nyaxdir[3], nzaxdir[3];
  float bxc[3], cxa[3], axb[3];
  float tmp;

  // copy axis direction vectors
  for (i=0; i<3; i++) {
    xaxdir[i] = float(v->xaxis[i]);
    yaxdir[i] = float(v->yaxis[i]);
    zaxdir[i] = float(v->zaxis[i]);
  }

  // calculate reciprocal lattice vector for X texture coordiante
  cross_prod(bxc, yaxdir, zaxdir);
  tmp = dot_prod(xaxdir, bxc); 
  for (i=0; i<3; i++) {
    nxaxdir[i] = bxc[i] / tmp;
  }

  // calculate reciprocal lattice vector for Y texture coordiante
  cross_prod(cxa, zaxdir, xaxdir);
  tmp = dot_prod(yaxdir, cxa); 
  for (i=0; i<3; i++) {
    nyaxdir[i] = cxa[i] / tmp;
  }

  // calculate reciprocal lattice vector for Z texture coordiante
  cross_prod(axb, xaxdir, yaxdir);
  tmp = dot_prod(zaxdir, axb); 
  for (i=0; i<3; i++) {
    nzaxdir[i] = axb[i] / tmp;
  }

  // negate and transform the volume origin to reciprocal space
  // for use in the OpenGL texgen plane equation
  float norigin[3];
  for (i=0; i<3; i++)
    norigin[i] = float(v->origin[i]);

  v0[0] = -dot_prod(norigin, nxaxdir) * tscale[0];
  v0[1] = -dot_prod(norigin, nyaxdir) * tscale[1];
  v0[2] = -dot_prod(norigin, nzaxdir) * tscale[2];

  // scale the volume axes for the OpenGL texgen plane equation
  for (i=0; i<3; i++) {
    v1[i] = nxaxdir[i] * tscale[0];
    v2[i] = nyaxdir[i] * tscale[1];
    v3[i] = nzaxdir[i] * tscale[2];
  }
}

