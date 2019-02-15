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
 *	$RCSfile: QuickSurf.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.122 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Fast gaussian surface representation
 ***************************************************************************/

// pgcc 2016 has troubles with hand-vectorized x86 intrinsics presently
#if !defined(__PGIC__)
#if defined(VMDUSEAVX512) && defined(__AVX512F__) && defined(__AVX512ER__)
// AVX512F + AVX512ER for Xeon Phi
#define VMDQSURFUSEAVX512 1
#else
// fall-back to SSE or AVX
#define VMDQSURFUSESSE 1

// The x86 AVX code path requires FMA and AVX2 integer instructions
// in order to achieve performance that actually beats SSE2.
// #define VMDQSURFUSEAVX2 1
#endif
#endif

// The OpenPOWER VSX code path runs on POWER8 and later hardware, but is
// untested on older platforms that support VSX instructions.
// XXX GCC 4.8.5 breaks with conflicts between vec_xxx() routines
//     defined in utilities.h vs. VSX intrinsics in altivec.h and similar.
//     For now, we disable VSX for GCC for this source file.
#if !defined(__GNUC__) && defined(__VEC__)
#define VMDQSURFUSEVSX 1
#endif

#include <stdio.h>
#include <stdlib.h>
#if VMDQSURFUSESSE && defined(__SSE2__) 
#include <emmintrin.h>
#endif
#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif
#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
#include <immintrin.h>
#endif
#if (defined(VMDQSURFUSEVSX) && defined(__VSX__))
#if defined(__GNUC__) && defined(__VEC__)
#include <altivec.h>
#endif
#endif

#include <string.h>
#include <math.h>
#include "QuickSurf.h"
#if defined(VMDCUDA)
#include "CUDAQuickSurf.h"
#endif
#include "Measure.h"
#include "Inform.h"
#include "utilities.h"
#include "WKFUtils.h"
#include "VolumetricData.h"

#include "VMDDisplayList.h"
#include "Displayable.h"
#include "DispCmds.h"
#include "ProfileHooks.h"


#define MIN(X,Y) (((X)<(Y))? (X) : (Y))
#define MAX(X,Y) (((X)>(Y))? (X) : (Y))


/*
 * David J. Hardy
 * 12 Dec 2008
 *
 * aexpfnx() - Approximate expf() for negative x.
 *
 * Assumes that x <= 0.
 *
 * Assumes IEEE format for single precision float, specifically:
 * 1 sign bit, 8 exponent bits biased by 127, and 23 mantissa bits.
 *
 * Interpolates exp() on interval (-1/log2(e), 0], then shifts it by
 * multiplication of a fast calculation for 2^(-N).  The interpolation
 * uses a linear blending of 3rd degree Taylor polynomials at the end
 * points, so the approximation is once differentiable.
 *
 * The error is small (max relative error per interval is calculated
 * to be 0.131%, with a max absolute error of -0.000716).
 *
 * The cutoff is chosen so as to speed up the computation by early
 * exit from function, with the value chosen to give less than the
 * the max absolute error.  Use of a cutoff is unnecessary, except
 * for needing to shift smallest floating point numbers to zero,
 * i.e. you could remove cutoff and replace by:
 *
 * #define MINXNZ  -88.0296919311130  // -127 * log(2)
 *
 *   if (x < MINXNZ) return 0.f;
 *
 * Use of a cutoff causes a discontinuity which can be eliminated
 * through the use of a switching function.
 *
 * We can obtain arbitrarily smooth approximation by taking k+1 nodes on
 * the interval and weighting their respective Taylor polynomials by the
 * kth order Lagrange interpolant through those nodes.  The wiggle in the
 * polynomial interpolation due to equidistant nodes (Runge's phenomenon)
 * can be reduced by using Chebyshev nodes.
 */

#if defined(__GNUC__) && ! defined(__INTEL_COMPILER)
#define __align(X)  __attribute__((aligned(X) ))
#if (__GNUC__ < 4)
#define MISSING_mm_cvtsd_f64
#endif
#else
#define __align(X) __declspec(align(X) )
#endif

#define MLOG2EF    -1.44269504088896f

/*
 * Interpolating coefficients for linear blending of the
 * 3rd degree Taylor expansion of 2^x about 0 and -1.
 */
#define SCEXP0     1.0000000000000000f
#define SCEXP1     0.6987082824680118f
#define SCEXP2     0.2633174272827404f
#define SCEXP3     0.0923611991471395f
#define SCEXP4     0.0277520543324108f

/* for single precision float */
#define EXPOBIAS   127
#define EXPOSHIFT   23

/* cutoff is optional, but can help avoid unnecessary work */
#define ACUTOFF    -10

typedef union flint_t {
  float f;
  int n;
} flint;

#if VMDQSURFUSESSE && defined(__SSE2__)
// SSE variant of the 'flint' union above
typedef union SSEreg_t {
  __m128  f;  // 4x float (SSE)
  __m128i i;  // 4x 32-bit int (SSE2)
} SSEreg;
#endif
#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
// AVX variant of the 'flint' union above
typedef union AVXreg_t {
  __m256  f;  // 8x float (AVX)
  __m256i i;  // 8x 32-bit int (AVX)
  struct {
    float r0, r1, r2, r3, r4, r5, r6, r7;  // get the individual registers
  } floatreg;
} AVXreg;
#endif


#if 0
static float aexpfnx(float x) {
  /* assume x <= 0 */
  float mb;
  int mbflr;
  float d;
  float sy;
  flint scalfac;

  if (x < ACUTOFF) return 0.f;

  mb = x * MLOG2EF;    /* change base to 2, mb >= 0 */
  mbflr = (int) mb;    /* get int part, floor() */
  d = mbflr - mb;      /* remaining exponent, -1 < d <= 0 */
  sy = SCEXP0 + d*(SCEXP1 + d*(SCEXP2 + d*(SCEXP3 + d*SCEXP4)));
                       /* approx with linear blend of Taylor polys */
  scalfac.n = (EXPOBIAS - mbflr) << EXPOSHIFT;  /* 2^(-mbflr) */
  return (sy * scalfac.f);  /* scaled approx */
}


static void vmd_gaussdensity(int verbose, 
                             int natoms, const float *xyzr,
                             const float *atomicnum,
                             const float *colors,
                             float *densitymap, float *voltexmap, 
                             const int *numvoxels, 
                             float radscale, float gridspacing, 
                             float isovalue, float gausslim) {
  int i, x, y, z;
  int maxvoxel[3];
  maxvoxel[0] = numvoxels[0]-1; 
  maxvoxel[1] = numvoxels[1]-1; 
  maxvoxel[2] = numvoxels[2]-1; 
  const float invgridspacing = 1.0f / gridspacing;

  // compute colors only if necessary, since they are costly
  if (voltexmap != NULL) {
    float invisovalue = 1.0f / isovalue;
    // compute both density map and floating point color texture map
    for (i=0; i<natoms; i++) {
      if (verbose && ((i & 0x3fff) == 0)) {
        printf("."); 
        fflush(stdout);
      }

      long ind = i*4L;
      float scaledrad = xyzr[ind + 3L] * radscale;

      // MDFF atomic number weighted density factor
      float atomicnumfactor = 1.0f;
      if (atomicnum != NULL) {
        atomicnumfactor = atomicnum[i];
      }

      float arinv = 1.0f/(2.0f*scaledrad*scaledrad);
      float radlim = gausslim * scaledrad;
      float radlim2 = radlim * radlim; // cutoff test done in cartesian coords
      radlim *= invgridspacing;

      float tmp;
      tmp = xyzr[ind  ] * invgridspacing;
      int xmin = MAX((int) (tmp - radlim), 0);
      int xmax = MIN((int) (tmp + radlim), maxvoxel[0]);
      tmp = xyzr[ind+1] * invgridspacing;
      int ymin = MAX((int) (tmp - radlim), 0);
      int ymax = MIN((int) (tmp + radlim), maxvoxel[1]);
      tmp = xyzr[ind+2] * invgridspacing;
      int zmin = MAX((int) (tmp - radlim), 0);
      int zmax = MIN((int) (tmp + radlim), maxvoxel[2]);

      float dz = zmin*gridspacing - xyzr[ind+2];
      for (z=zmin; z<=zmax; z++,dz+=gridspacing) {
        float dy = ymin*gridspacing - xyzr[ind+1];
        for (y=ymin; y<=ymax; y++,dy+=gridspacing) {
          float dy2dz2 = dy*dy + dz*dz;

          // early-exit when outside the cutoff radius in the Y-Z plane
          if (dy2dz2 >= radlim2) 
            continue;

          int addr = z * numvoxels[0] * numvoxels[1] + y * numvoxels[0];
          float dx = xmin*gridspacing - xyzr[ind];
          for (x=xmin; x<=xmax; x++,dx+=gridspacing) {
            float r2 = dx*dx + dy2dz2;
            float expval = -r2 * arinv;
#if VMDUSEFULLEXP
            // use the math library exponential routine
            float density = exp(expval);
#else
            // use our (much faster) fast exponential approximation
            float density = aexpfnx(expval);
#endif

            density *= atomicnumfactor; // MDFF Cryo-EM atomic number density

            // accumulate density value to density map
            densitymap[addr + x] += density;

            // Accumulate density-weighted color to texture map.
            // Pre-multiply colors by the inverse isovalue we will extract   
            // the surface on, to cause the final color to be normalized.
            density *= invisovalue;
            long caddr = (addr + x) * 3L;

            // color by atom colors
            voltexmap[caddr    ] += density * colors[ind    ];
            voltexmap[caddr + 1] += density * colors[ind + 1];
            voltexmap[caddr + 2] += density * colors[ind + 2];
          }
        }
      }
    }
  } else {
    // compute density map only
    for (i=0; i<natoms; i++) {
      if (verbose && ((i & 0x3fff) == 0)) {
        printf("."); 
        fflush(stdout);
      }

      long ind = i*4L;
      float scaledrad = xyzr[ind + 3] * radscale;

      // MDFF atomic number weighted density factor
      float atomicnumfactor = 1.0f;
      if (atomicnum != NULL) {
        atomicnumfactor = atomicnum[i];
      }

      float arinv = 1.0f/(2.0f*scaledrad*scaledrad);
      float radlim = gausslim * scaledrad;
      float radlim2 = radlim * radlim; // cutoff test done in cartesian coords
      radlim *= invgridspacing;

      float tmp;
      tmp = xyzr[ind  ] * invgridspacing;
      int xmin = MAX((int) (tmp - radlim), 0);
      int xmax = MIN((int) (tmp + radlim), maxvoxel[0]);
      tmp = xyzr[ind+1] * invgridspacing;
      int ymin = MAX((int) (tmp - radlim), 0);
      int ymax = MIN((int) (tmp + radlim), maxvoxel[1]);
      tmp = xyzr[ind+2] * invgridspacing;
      int zmin = MAX((int) (tmp - radlim), 0);
      int zmax = MIN((int) (tmp + radlim), maxvoxel[2]);

      float dz = zmin*gridspacing - xyzr[ind+2];
      for (z=zmin; z<=zmax; z++,dz+=gridspacing) {
        float dy = ymin*gridspacing - xyzr[ind+1];
        for (y=ymin; y<=ymax; y++,dy+=gridspacing) {
          float dy2dz2 = dy*dy + dz*dz;

          // early-exit when outside the cutoff radius in the Y-Z plane
          if (dy2dz2 >= radlim2) 
            continue;

          int addr = z * numvoxels[0] * numvoxels[1] + y * numvoxels[0];
          float dx = xmin*gridspacing - xyzr[ind];
          for (x=xmin; x<=xmax; x++,dx+=gridspacing) {
            float r2 = dx*dx + dy2dz2;
            float expval = -r2 * arinv;
#if VMDUSEFULLEXP
            // use the math library exponential routine
            float density = exp(expval);
#else
            // use our (much faster) fast exponential approximation
            float density = aexpfnx(expval);
#endif

            density *= atomicnumfactor; // MDFF Cryo-EM atomic number density

            // accumulate density value to density map
            densitymap[addr + x] += density;
          }
        }
      }
    }
  }
}
#endif



static void vmd_gaussdensity_opt(int verbose,
                                 int natoms, const float *xyzr,
                                 const float *atomicnum,
                                 const float *colors,
                                 float *densitymap, float *voltexmap, 
                                 const int *numvoxels, 
                                 float radscale, float gridspacing, 
                                 float isovalue, float gausslim) {
  int i, x, y, z;
  int maxvoxel[3];
  maxvoxel[0] = numvoxels[0]-1; 
  maxvoxel[1] = numvoxels[1]-1; 
  maxvoxel[2] = numvoxels[2]-1; 
  const float invgridspacing = 1.0f / gridspacing;

#if (VMDQSURFUSESSE && defined(__SSE2__)) || (VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)) || (VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)) 
  // XXX replace with calls to centralized control system
  int usesse=1;
  if (getenv("VMDNOSSE")) {
    usesse=0;
  }
#endif

#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
  int useavx2=1;
  if (getenv("VMDNOAVX2")) {
    useavx2=0;
  }
#endif

#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
  int useavx512=1;
  if (getenv("VMDNOAVX512")) {
    useavx512=0;
  }
#endif

#if VMDQSURFUSEVSX && defined(__VEC__)
  int usevsx=1;
  if (getenv("VMDNOVSX")) {
    usevsx=0;
  }
#endif

#if VMDQSURFUSESSE && defined(__SSE2__)
  // Variables for SSE optimized inner loop
  __m128 gridspacing4_4;
  __attribute__((aligned(16))) float sxdelta4[4]; // 16-byte aligned for SSE

  if (usesse) {
    gridspacing4_4 = _mm_set1_ps(gridspacing * 4.0f);
    for (x=0; x<4; x++)
      sxdelta4[x] = ((float) x) * gridspacing;
  }
#endif

#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
  // Variables for AVX2 optimized inner loop
  __m256 gridspacing8_8;
  __attribute__((aligned(16))) float sxdelta8[8]; // 16-byte aligned for AVX2

  if (useavx2) {
    gridspacing8_8 = _mm256_set1_ps(gridspacing * 8.0f);
    for (x=0; x<8; x++)
      sxdelta8[x] = ((float) x) * gridspacing;
  }
#endif

#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
  // Variables for AVX512 optimized inner loop
  __m512 gridspacing16_16;
  __attribute__((aligned(64))) float sxdelta16[16]; // 16-byte aligned for AVX2

  if (useavx512) {
    gridspacing16_16 = _mm512_set1_ps(gridspacing * 16.0f);
    for (x=0; x<16; x++)
      sxdelta16[x] = ((float) x) * gridspacing;
  }
#endif

#if VMDQSURFUSEVSX && defined(__VEC__)
  // Variables for VSX optimized inner loop
  vector float gridspacing4_4;
  __attribute__((aligned(16))) float sxdelta4[4]; // 16-byte aligned for VSX

  if (usevsx) {
    gridspacing4_4 = vec_splats(gridspacing * 4.0f);
    for (x=0; x<4; x++)
      sxdelta4[x] = ((float) x) * gridspacing;
  }
#endif

  // compute colors only if necessary, since they are costly
  if (voltexmap != NULL) {
    float invisovalue = 1.0f / isovalue;
    // compute both density map and floating point color texture map
    for (i=0; i<natoms; i++) {
      if (verbose && ((i & 0x3fff) == 0)) {
        printf("."); 
        fflush(stdout);
      }

      long ind = i*4L;
      float scaledrad = xyzr[ind + 3] * radscale;

      // MDFF atomic number weighted density factor
      float atomicnumfactor = 1.0f;
      if (atomicnum != NULL) {
        atomicnumfactor = atomicnum[i];
      }

      // negate, precompute reciprocal, and change to base 2 from the outset
      float arinv = -(1.0f/(2.0f*scaledrad*scaledrad)) * MLOG2EF;
      float radlim = gausslim * scaledrad;
      float radlim2 = radlim * radlim; // cutoff test done in cartesian coords
      radlim *= invgridspacing;

#if VMDQSURFUSESSE && defined(__SSE2__)
      __m128 atomicnumfactor_4;
      __m128 arinv_4;
      if (usesse) {
        atomicnumfactor_4 = _mm_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_4 = _mm_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use our fully inlined exp approximation
        arinv_4 = _mm_set1_ps(arinv);
#endif
      }
#endif

#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
      __m256 atomicnumfactor_8;
      __m256 arinv_8;
      if (useavx2) {
        atomicnumfactor_8 = _mm256_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_8 = _mm256_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use our fully inlined exp approximation
        arinv_8 = _mm256_set1_ps(arinv);
#endif
      }
#endif

#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
      __m512 atomicnumfactor_16;
      __m512 arinv_16;
      if (useavx512) {
        atomicnumfactor_16 = _mm512_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_16 = _mm512_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use AVX512-based exp2() approximation
        arinv_16 = _mm512_set1_ps(arinv);
#endif
      }
#endif

      float tmp;
      tmp = xyzr[ind  ] * invgridspacing;
      int xmin = MAX((int) (tmp - radlim), 0);
      int xmax = MIN((int) (tmp + radlim), maxvoxel[0]);
      tmp = xyzr[ind+1] * invgridspacing;
      int ymin = MAX((int) (tmp - radlim), 0);
      int ymax = MIN((int) (tmp + radlim), maxvoxel[1]);
      tmp = xyzr[ind+2] * invgridspacing;
      int zmin = MAX((int) (tmp - radlim), 0);
      int zmax = MIN((int) (tmp + radlim), maxvoxel[2]);

      float dz = zmin*gridspacing - xyzr[ind+2];
      for (z=zmin; z<=zmax; z++,dz+=gridspacing) {
        float dy = ymin*gridspacing - xyzr[ind+1];
        for (y=ymin; y<=ymax; y++,dy+=gridspacing) {
          float dy2dz2 = dy*dy + dz*dz;

          // early-exit when outside the cutoff radius in the Y-Z plane
          if (dy2dz2 >= radlim2) 
            continue;

          int addr = z * numvoxels[0] * numvoxels[1] + y * numvoxels[0];
          float dx = xmin*gridspacing - xyzr[ind];
          x=xmin;

#if 0 && VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
          // Use AVX512 when we have a multiple-of-16 to compute
          // finish all remaining density map points with 
          // AVX2, SSE, or regular non-SSE loop
          if (useavx512) {
            __align(64) __m512 y;
            __m512 dy2dz2_16 = _mm512_set1_ps(dy2dz2);
            __m512 dx_16 = _mm512_add_ps(_mm512_set1_ps(dx), _mm512_load_ps(&sxdelta16[0]));

            for (; (x+15)<=xmax; x+=16,dx_16=_mm512_add_ps(dx_16, gridspacing16_16)) {
              __m512 r2 = _mm512_fmadd_ps(dx_16, dx_16, dy2dz2_16);
              __m512 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y = _mm512_exp2_ps(_mm512_mul_ps(r2, arinv_16));
#else
              // use (much faster) exp2() approximation instruction
              // inputs already negated and in base 2 
              y = _mm512_exp2a23_ps(_mm512_mul_ps(r2, arinv_16));
#endif

              // At present, we do unaligned loads/stores since we can't 
              // guarantee that the X-dimension is always a multiple of 16.
              float *ufptr = &densitymap[addr + x];
              d = _mm512_loadu_ps(ufptr); 
              _mm512_storeu_ps(ufptr, _mm512_add_ps(d, y)); 

              // Accumulate density-weighted color to texture map.
              // Pre-multiply colors by the inverse isovalue we will extract
              // the surface on, to cause the final color to be normalized.
              d = _mm512_mul_ps(y, _mm512_set1_ps(invisovalue));
              long caddr = (addr + x) * 3L;
#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
              // convert rgb3f AOS format to 8-element SOA vectors using shuffle instructions
              float *txptr = &voltexmap[caddr];
              __m128 *m = (__m128 *) txptr;
              // unaligned load of 8 consecutive rgb3f texture map texels
              __m256 m03 = _mm256_castps128_ps256(m[0]); // load lower halves
              __m256 m14 = _mm256_castps128_ps256(m[1]);
              __m256 m25 = _mm256_castps128_ps256(m[2]);
              m03  = _mm256_insertf128_ps(m03, m[3], 1); // load upper halves
              m14  = _mm256_insertf128_ps(m14, m[4], 1);
              m25  = _mm256_insertf128_ps(m25, m[5], 1);
 
              // upper Rs and Gs 
              __m256 rg = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2,1,3,2));
              // lower Gs and Bs
              __m256 gb = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1,0,2,1));
              __m256 r  = _mm256_shuffle_ps(m03, rg , _MM_SHUFFLE(2,0,3,0)); 
              __m256 g  = _mm256_shuffle_ps(gb , rg , _MM_SHUFFLE(3,1,2,0)); 
              __m256 b  = _mm256_shuffle_ps(gb , m25, _MM_SHUFFLE(3,0,3,1)); 

              // accumulate density-scaled colors into texels
#if 1
              r = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind    ]), r);
              g = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind + 1]), g);
              b = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind + 2]), b);
#else
              r = _mm256_add_ps(r, _mm256_mul_ps(d, _mm256_set1_ps(colors[ind    ])));
              g = _mm256_add_ps(g, _mm256_mul_ps(d, _mm256_set1_ps(colors[ind + 1])));
              b = _mm256_add_ps(b, _mm256_mul_ps(d, _mm256_set1_ps(colors[ind + 2])));
#endif

              // convert 8-element SOA vectors to rgb3f AOS format using shuffle instructions
              __m256 rrg = _mm256_shuffle_ps(r, g, _MM_SHUFFLE(2,0,2,0)); 
              __m256 rgb = _mm256_shuffle_ps(g, b, _MM_SHUFFLE(3,1,3,1)); 
              __m256 rbr = _mm256_shuffle_ps(b, r, _MM_SHUFFLE(3,1,2,0)); 
              __m256 r03 = _mm256_shuffle_ps(rrg, rbr, _MM_SHUFFLE(2,0,2,0));  
              __m256 r14 = _mm256_shuffle_ps(rgb, rrg, _MM_SHUFFLE(3,1,2,0)); 
              __m256 r25 = _mm256_shuffle_ps(rbr, rgb, _MM_SHUFFLE(3,1,3,1));  

              // unaligned store of consecutive rgb3f texture map texels
              m[0] = _mm256_castps256_ps128( r03 );
              m[1] = _mm256_castps256_ps128( r14 );
              m[2] = _mm256_castps256_ps128( r25 );
              m[3] = _mm256_extractf128_ps( r03 ,1);
              m[4] = _mm256_extractf128_ps( r14 ,1);
              m[5] = _mm256_extractf128_ps( r25 ,1);
#else
              // color by atom colors
              float r, g, b;
              r = colors[ind    ];
              g = colors[ind + 1];
              b = colors[ind + 2];

              AVXreg tmp;
              tmp.f = d;
              float density;
              density = tmp.floatreg.r0;
              voltexmap[caddr     ] += density * r;
              voltexmap[caddr +  1] += density * g;
              voltexmap[caddr +  2] += density * b;

              density = tmp.floatreg.r1;
              voltexmap[caddr +  3] += density * r;
              voltexmap[caddr +  4] += density * g;
              voltexmap[caddr +  5] += density * b;

              density = tmp.floatreg.r2;
              voltexmap[caddr +  6] += density * r;
              voltexmap[caddr +  7] += density * g;
              voltexmap[caddr +  8] += density * b;

              density = tmp.floatreg.r3;
              voltexmap[caddr +  9] += density * r;
              voltexmap[caddr + 10] += density * g;
              voltexmap[caddr + 11] += density * b;

              density = tmp.floatreg.r4;
              voltexmap[caddr + 12] += density * r;
              voltexmap[caddr + 13] += density * g;
              voltexmap[caddr + 14] += density * b;

              density = tmp.floatreg.r5;
              voltexmap[caddr + 15] += density * r;
              voltexmap[caddr + 16] += density * g;
              voltexmap[caddr + 17] += density * b;

              density = tmp.floatreg.r6;
              voltexmap[caddr + 18] += density * r;
              voltexmap[caddr + 19] += density * g;
              voltexmap[caddr + 20] += density * b;

              density = tmp.floatreg.r7;
              voltexmap[caddr + 21] += density * r;
              voltexmap[caddr + 22] += density * g;
              voltexmap[caddr + 23] += density * b;
#endif
            }
          }
#endif


#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
          // Use AVX when we have a multiple-of-8 to compute
          // finish all remaining density map points with SSE or regular non-SSE loop
          if (useavx2) {
            __align(16) AVXreg n;
            __align(16) AVXreg y;
            __m256 dy2dz2_8 = _mm256_set1_ps(dy2dz2);
            __m256 dx_8 = _mm256_add_ps(_mm256_set1_ps(dx), _mm256_load_ps(&sxdelta8[0]));

            for (; (x+7)<=xmax; x+=8,dx_8=_mm256_add_ps(dx_8, gridspacing8_8)) {
              __m256 r2 = _mm256_fmadd_ps(dx_8, dx_8, dy2dz2_8);
              __m256 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y.f = _mm256_exp2_ps(_mm256_mul_ps(r2, arinv_8));
#else
              // use our (much faster) fully inlined exponential approximation
              y.f = _mm256_mul_ps(r2, arinv_8);         /* already negated and in base 2 */
              n.i = _mm256_cvttps_epi32(y.f);
              d = _mm256_cvtepi32_ps(n.i);
              d = _mm256_sub_ps(d, y.f);

              // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
              // Perform Horner's method to evaluate interpolating polynomial.
              y.f = _mm256_fmadd_ps(d, _mm256_set1_ps(SCEXP4), _mm256_set1_ps(SCEXP3)); 
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP2));
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP1));
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP0));

              // Calculate 2^N exactly by directly manipulating floating point exponent,
              // then use it to scale y for the final result.
              // We need AVX2 instructions to be able to operate on 
              // 8-wide integer types efficiently.
              n.i = _mm256_sub_epi32(_mm256_set1_epi32(EXPOBIAS), n.i);
              n.i = _mm256_slli_epi32(n.i, EXPOSHIFT);
              y.f = _mm256_mul_ps(y.f, n.f);
              y.f = _mm256_mul_ps(y.f, atomicnumfactor_8); // MDFF density maps
#endif

              // At present, we do unaligned loads/stores since we can't guarantee
              // that the X-dimension is always a multiple of 8.
              float *ufptr = &densitymap[addr + x];
              d = _mm256_loadu_ps(ufptr); 
              _mm256_storeu_ps(ufptr, _mm256_add_ps(d, y.f)); 

              // Accumulate density-weighted color to texture map.
              // Pre-multiply colors by the inverse isovalue we will extract
              // the surface on, to cause the final color to be normalized.
              d = _mm256_mul_ps(y.f, _mm256_set1_ps(invisovalue));
              long caddr = (addr + x) * 3L;

#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
              // convert rgb3f AOS format to 8-element SOA vectors using shuffle instructions
              float *txptr = &voltexmap[caddr];
              __m128 *m = (__m128 *) txptr;
              // unaligned load of 8 consecutive rgb3f texture map texels
              __m256 m03 = _mm256_castps128_ps256(m[0]); // load lower halves
              __m256 m14 = _mm256_castps128_ps256(m[1]);
              __m256 m25 = _mm256_castps128_ps256(m[2]);
              m03  = _mm256_insertf128_ps(m03, m[3], 1); // load upper halves
              m14  = _mm256_insertf128_ps(m14, m[4], 1);
              m25  = _mm256_insertf128_ps(m25, m[5], 1);
 
              // upper Rs and Gs 
              __m256 rg = _mm256_shuffle_ps(m14, m25, _MM_SHUFFLE(2,1,3,2));
              // lower Gs and Bs
              __m256 gb = _mm256_shuffle_ps(m03, m14, _MM_SHUFFLE(1,0,2,1));
              __m256 r  = _mm256_shuffle_ps(m03, rg , _MM_SHUFFLE(2,0,3,0)); 
              __m256 g  = _mm256_shuffle_ps(gb , rg , _MM_SHUFFLE(3,1,2,0)); 
              __m256 b  = _mm256_shuffle_ps(gb , m25, _MM_SHUFFLE(3,0,3,1)); 

              // accumulate density-scaled colors into texels
              r = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind    ]), r);
              g = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind + 1]), g);
              b = _mm256_fmadd_ps(d, _mm256_set1_ps(colors[ind + 2]), b);

              // convert 8-element SOA vectors to rgb3f AOS format using shuffle instructions
              __m256 rrg = _mm256_shuffle_ps(r, g, _MM_SHUFFLE(2,0,2,0)); 
              __m256 rgb = _mm256_shuffle_ps(g, b, _MM_SHUFFLE(3,1,3,1)); 
              __m256 rbr = _mm256_shuffle_ps(b, r, _MM_SHUFFLE(3,1,2,0)); 
              __m256 r03 = _mm256_shuffle_ps(rrg, rbr, _MM_SHUFFLE(2,0,2,0));  
              __m256 r14 = _mm256_shuffle_ps(rgb, rrg, _MM_SHUFFLE(3,1,2,0)); 
              __m256 r25 = _mm256_shuffle_ps(rbr, rgb, _MM_SHUFFLE(3,1,3,1));  

              // unaligned store of consecutive rgb3f texture map texels
              m[0] = _mm256_castps256_ps128( r03 );
              m[1] = _mm256_castps256_ps128( r14 );
              m[2] = _mm256_castps256_ps128( r25 );
              m[3] = _mm256_extractf128_ps( r03 ,1);
              m[4] = _mm256_extractf128_ps( r14 ,1);
              m[5] = _mm256_extractf128_ps( r25 ,1);
#else
              // color by atom colors
              float r, g, b;
              r = colors[ind    ];
              g = colors[ind + 1];
              b = colors[ind + 2];

              AVXreg tmp;
              tmp.f = d;
              float density;
              density = tmp.floatreg.r0;
              voltexmap[caddr     ] += density * r;
              voltexmap[caddr +  1] += density * g;
              voltexmap[caddr +  2] += density * b;

              density = tmp.floatreg.r1;
              voltexmap[caddr +  3] += density * r;
              voltexmap[caddr +  4] += density * g;
              voltexmap[caddr +  5] += density * b;

              density = tmp.floatreg.r2;
              voltexmap[caddr +  6] += density * r;
              voltexmap[caddr +  7] += density * g;
              voltexmap[caddr +  8] += density * b;

              density = tmp.floatreg.r3;
              voltexmap[caddr +  9] += density * r;
              voltexmap[caddr + 10] += density * g;
              voltexmap[caddr + 11] += density * b;

              density = tmp.floatreg.r4;
              voltexmap[caddr + 12] += density * r;
              voltexmap[caddr + 13] += density * g;
              voltexmap[caddr + 14] += density * b;

              density = tmp.floatreg.r5;
              voltexmap[caddr + 15] += density * r;
              voltexmap[caddr + 16] += density * g;
              voltexmap[caddr + 17] += density * b;

              density = tmp.floatreg.r6;
              voltexmap[caddr + 18] += density * r;
              voltexmap[caddr + 19] += density * g;
              voltexmap[caddr + 20] += density * b;

              density = tmp.floatreg.r7;
              voltexmap[caddr + 21] += density * r;
              voltexmap[caddr + 22] += density * g;
              voltexmap[caddr + 23] += density * b;
#endif
            }
          }
#endif





#if VMDQSURFUSESSE && defined(__SSE2__)
          // Use SSE when we have a multiple-of-4 to compute
          // finish all remaining density map points with regular non-SSE loop
          if (usesse) {
            __align(16) SSEreg n;
            __align(16) SSEreg y;
            __m128 dy2dz2_4 = _mm_set1_ps(dy2dz2);
            __m128 dx_4 = _mm_add_ps(_mm_set1_ps(dx), _mm_load_ps(&sxdelta4[0]));

            for (; (x+3)<=xmax; x+=4,dx_4=_mm_add_ps(dx_4, gridspacing4_4)) {
              __m128 r2 = _mm_add_ps(_mm_mul_ps(dx_4, dx_4), dy2dz2_4);
              __m128 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y.f = _mm_exp2_ps(_mm_mul_ps(r2, arinv_4));
#else
              // use our (much faster) fully inlined exponential approximation
              y.f = _mm_mul_ps(r2, arinv_4);         /* already negated and in base 2 */
              n.i = _mm_cvttps_epi32(y.f);
              d = _mm_cvtepi32_ps(n.i);
              d = _mm_sub_ps(d, y.f);

              // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
              // Perform Horner's method to evaluate interpolating polynomial.
#if 0
              // SSE 4.x FMADD instructions are not universally available
              y.f = _mm_fmadd_ps(d, _mm_set1_ps(SCEXP4), _mm_set1_ps(SCEXP3)); 
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP2));
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP1));
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP0));
#else
              y.f = _mm_mul_ps(d, _mm_set_ps1(SCEXP4));      /* for x^4 term */
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP3));    /* for x^3 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP2));    /* for x^2 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP1));    /* for x^1 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP0));    /* for x^0 term */
#endif

              // Calculate 2^N exactly by directly manipulating floating point exponent,
              // then use it to scale y for the final result.
              n.i = _mm_sub_epi32(_mm_set1_epi32(EXPOBIAS), n.i);
              n.i = _mm_slli_epi32(n.i, EXPOSHIFT);
              y.f = _mm_mul_ps(y.f, n.f);
#endif

              // At present, we do unaligned loads/stores since we can't guarantee
              // that the X-dimension is always a multiple of 4.
              float *ufptr = &densitymap[addr + x];
              d = _mm_loadu_ps(ufptr);
              y.f = _mm_mul_ps(y.f, atomicnumfactor_4); // MDFF density maps
              _mm_storeu_ps(ufptr, _mm_add_ps(d, y.f)); 

              // Accumulate density-weighted color to texture map.
              // Pre-multiply colors by the inverse isovalue we will extract   
              // the surface on, to cause the final color to be normalized.
              d = _mm_mul_ps(y.f, _mm_set_ps1(invisovalue));
              long caddr = (addr + x) * 3L;

#if 1
              float *txptr = &voltexmap[caddr];
              // unaligned load of 4 consecutive rgb3f texture map texels
              __m128 r0g0b0r1 = _mm_loadu_ps(txptr+0);
              __m128 g1b1r2g2 = _mm_loadu_ps(txptr+4); 
              __m128 b2r3g3b3 = _mm_loadu_ps(txptr+8);

              // convert rgb3f AOS format to 4-element SOA vectors using shuffle instructions
              __m128 r2g2r3g3 = _mm_shuffle_ps(g1b1r2g2, b2r3g3b3, _MM_SHUFFLE(2, 1, 3, 2)); 
              __m128 g0b0g1b1 = _mm_shuffle_ps(r0g0b0r1, g1b1r2g2, _MM_SHUFFLE(1, 0, 2, 1));
              __m128 r        = _mm_shuffle_ps(r0g0b0r1, r2g2r3g3, _MM_SHUFFLE(2, 0, 3, 0)); // r0r1r2r3
              __m128 g        = _mm_shuffle_ps(g0b0g1b1, r2g2r3g3, _MM_SHUFFLE(3, 1, 2, 0)); // g0g1g2g3
              __m128 b        = _mm_shuffle_ps(g0b0g1b1, b2r3g3b3, _MM_SHUFFLE(3, 0, 3, 1)); // b0g1b2b3

              // accumulate density-scaled colors into texels
              r = _mm_add_ps(r, _mm_mul_ps(d, _mm_set_ps1(colors[ind    ])));
              g = _mm_add_ps(g, _mm_mul_ps(d, _mm_set_ps1(colors[ind + 1])));
              b = _mm_add_ps(b, _mm_mul_ps(d, _mm_set_ps1(colors[ind + 2])));

              // convert 4-element SOA vectors to rgb3f AOS format using shuffle instructions
              __m128 r0r2g0g2  = _mm_shuffle_ps(r, g, _MM_SHUFFLE(2, 0, 2, 0));
              __m128 g1g3b1b3  = _mm_shuffle_ps(g, b, _MM_SHUFFLE(3, 1, 3, 1));
              __m128 b0b2r1r3  = _mm_shuffle_ps(b, r, _MM_SHUFFLE(3, 1, 2, 0));
 
              __m128 rr0g0b0r1 = _mm_shuffle_ps(r0r2g0g2, b0b2r1r3, _MM_SHUFFLE(2, 0, 2, 0)); 
              __m128 rg1b1r2g2 = _mm_shuffle_ps(g1g3b1b3, r0r2g0g2, _MM_SHUFFLE(3, 1, 2, 0)); 
              __m128 rb2r3g3b3 = _mm_shuffle_ps(b0b2r1r3, g1g3b1b3, _MM_SHUFFLE(3, 1, 3, 1)); 
 
              // unaligned store of 4 consecutive rgb3f texture map texels
              _mm_storeu_ps(txptr+0, rr0g0b0r1);
              _mm_storeu_ps(txptr+4, rg1b1r2g2);
              _mm_storeu_ps(txptr+8, rb2r3g3b3);

#else

              // color by atom colors
              float r, g, b;
              r = colors[ind    ];
              g = colors[ind + 1];
              b = colors[ind + 2];

              SSEreg tmp; 
              tmp.f = d;
              float density;
              density = tmp.floatreg.r0; 
              voltexmap[caddr     ] += density * r;
              voltexmap[caddr +  1] += density * g;
              voltexmap[caddr +  2] += density * b;

              density = tmp.floatreg.r1; 
              voltexmap[caddr +  3] += density * r;
              voltexmap[caddr +  4] += density * g;
              voltexmap[caddr +  5] += density * b;

              density = tmp.floatreg.r2; 
              voltexmap[caddr +  6] += density * r;
              voltexmap[caddr +  7] += density * g;
              voltexmap[caddr +  8] += density * b;

              density = tmp.floatreg.r3; 
              voltexmap[caddr +  9] += density * r;
              voltexmap[caddr + 10] += density * g;
              voltexmap[caddr + 11] += density * b;
#endif
            }
          }
#endif

          // finish all remaining density map points with regular non-SSE loop
          for (; x<=xmax; x++,dx+=gridspacing) {
            float r2 = dx*dx + dy2dz2;

            // use our (much faster) fully inlined exponential approximation
            float mb = r2 * arinv;         /* already negated and in base 2 */
            int mbflr = (int) mb;          /* get int part, floor() */
            float d = mbflr - mb;          /* remaining exponent, -1 < d <= 0 */

            /* approx with linear blend of Taylor polys */
            float sy = SCEXP0 + d*(SCEXP1 + d*(SCEXP2 + d*(SCEXP3 + d*SCEXP4)));

            /* 2^(-mbflr) */
            flint scalfac;
            scalfac.n = (EXPOBIAS - mbflr) << EXPOSHIFT;  

            // XXX assume we are never beyond the cutoff value in this loop
            float density = (sy * scalfac.f);

            density *= atomicnumfactor; // MDFF Cryo-EM atomic number density

            // accumulate density value to density map
            densitymap[addr + x] += density;

            // Accumulate density-weighted color to texture map.
            // Pre-multiply colors by the inverse isovalue we will extract   
            // the surface on, to cause the final color to be normalized.
            density *= invisovalue;
            long caddr = (addr + x) * 3L;

            // color by atom colors
            voltexmap[caddr    ] += density * colors[ind    ];
            voltexmap[caddr + 1] += density * colors[ind + 1];
            voltexmap[caddr + 2] += density * colors[ind + 2];
          }
        }
      }
    }
  } else {
    // compute density map only
    for (i=0; i<natoms; i++) {
      if (verbose && ((i & 0x3fff) == 0)) {
        printf("."); 
        fflush(stdout);
      }

      long ind = i*4L;
      float scaledrad = xyzr[ind+3] * radscale;

      // MDFF atomic number weighted density factor
      float atomicnumfactor = 1.0f;
      if (atomicnum != NULL) {
        atomicnumfactor = atomicnum[i];
      }

      // negate, precompute reciprocal, and change to base 2 from the outset
      float arinv = -(1.0f/(2.0f*scaledrad*scaledrad)) * MLOG2EF;
      float radlim = gausslim * scaledrad;
      float radlim2 = radlim * radlim; // cutoff test done in cartesian coords
      radlim *= invgridspacing;

#if VMDQSURFUSESSE && defined(__SSE2__)
      __m128 atomicnumfactor_4;
      __m128 arinv_4;
      if (usesse) {
        atomicnumfactor_4 = _mm_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_4 = _mm_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use our fully inlined exp approximation
        arinv_4 = _mm_set1_ps(arinv);
#endif
      }
#endif

#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
      __m256 atomicnumfactor_8;
      __m256 arinv_8;
      if (useavx2) {
        atomicnumfactor_8 = _mm256_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_8 = _mm256_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use our fully inlined exp approximation
        arinv_8 = _mm256_set1_ps(arinv);
#endif
      }
#endif

#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
      __m512 atomicnumfactor_16;
      __m512 arinv_16;
      if (useavx512) {
        atomicnumfactor_16 = _mm512_set1_ps(atomicnumfactor);
#if VMDUSESVMLEXP
        // Use of Intel's SVML requires changing the pre-scaling factor
        arinv_16 = _mm512_set1_ps(arinv * (2.718281828f/2.0f) / MLOG2EF); 
#else
        // Use our fully inlined exp approximation
        arinv_16 = _mm512_set1_ps(arinv);
#endif
      }
#endif

#if VMDQSURFUSEVSX && defined(__VEC__)
      vector float atomicnumfactor_4;
      vector float arinv_4;
      if (usevsx) {
        atomicnumfactor_4 = vec_splats(atomicnumfactor);

        // Use our fully inlined exp approximation
        arinv_4 = vec_splats(arinv);
      }
#endif

      float tmp;
      tmp = xyzr[ind  ] * invgridspacing;
      int xmin = MAX((int) (tmp - radlim), 0);
      int xmax = MIN((int) (tmp + radlim), maxvoxel[0]);
      tmp = xyzr[ind+1] * invgridspacing;
      int ymin = MAX((int) (tmp - radlim), 0);
      int ymax = MIN((int) (tmp + radlim), maxvoxel[1]);
      tmp = xyzr[ind+2] * invgridspacing;
      int zmin = MAX((int) (tmp - radlim), 0);
      int zmax = MIN((int) (tmp + radlim), maxvoxel[2]);

      float dz = zmin*gridspacing - xyzr[ind+2];
      for (z=zmin; z<=zmax; z++,dz+=gridspacing) {
        float dy = ymin*gridspacing - xyzr[ind+1];
        for (y=ymin; y<=ymax; y++,dy+=gridspacing) {
          float dy2dz2 = dy*dy + dz*dz;

          // early-exit when outside the cutoff radius in the Y-Z plane
          if (dy2dz2 >= radlim2) 
            continue;

          int addr = z * numvoxels[0] * numvoxels[1] + y * numvoxels[0];
          float dx = xmin*gridspacing - xyzr[ind];
          x=xmin;

#if VMDQSURFUSEAVX512 && defined(__AVX512F__) && defined(__AVX512ER__)
          // Use AVX512 when we have a multiple-of-16 to compute
          // finish all remaining density map points with 
          // AVX2, SSE, or regular non-SSE loop
          if (useavx512) {
            __align(64) __m512 y;
            __m512 dy2dz2_16 = _mm512_set1_ps(dy2dz2);
            __m512 dx_16 = _mm512_add_ps(_mm512_set1_ps(dx), _mm512_load_ps(&sxdelta16[0]));

            for (; (x+15)<=xmax; x+=16,dx_16=_mm512_add_ps(dx_16, gridspacing16_16)) {
              __m512 r2 = _mm512_fmadd_ps(dx_16, dx_16, dy2dz2_16);
              __m512 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y = _mm512_exp2_ps(_mm512_mul_ps(r2, arinv_16));
#else
              // use (much faster) exp2() approximation instruction
              // inputs already negated and in base 2 
              y = _mm512_exp2a23_ps(_mm512_mul_ps(r2, arinv_16));
#endif

              // At present, we do unaligned loads/stores since we can't 
              // guarantee that the X-dimension is always a multiple of 16.
              float *ufptr = &densitymap[addr + x];
              d = _mm512_loadu_ps(ufptr); 
              _mm512_storeu_ps(ufptr, _mm512_add_ps(d, y)); 
            }
          }
#endif


#if VMDQSURFUSEAVX2 && defined(__AVX__) && defined(__AVX2__)
          // Use AVX when we have a multiple-of-8 to compute
          // finish all remaining density map points with SSE or regular non-SSE loop
          if (useavx2) {
            __align(16) AVXreg scal;
            __align(16) AVXreg n;
            __align(16) AVXreg y;
            __m256 dy2dz2_8 = _mm256_set1_ps(dy2dz2);
            __m256 dx_8 = _mm256_add_ps(_mm256_set1_ps(dx), _mm256_load_ps(&sxdelta8[0]));

            for (; (x+7)<=xmax; x+=8,dx_8=_mm256_add_ps(dx_8, gridspacing8_8)) {
              __m256 r2 = _mm256_fmadd_ps(dx_8, dx_8, dy2dz2_8);
              __m256 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y.f = _mm256_exp2_ps(_mm256_mul_ps(r2, arinv_8));
#else
              // use our (much faster) fully inlined exponential approximation
              y.f = _mm256_mul_ps(r2, arinv_8);         /* already negated and in base 2 */
              n.i = _mm256_cvttps_epi32(y.f);
              d = _mm256_cvtepi32_ps(n.i);
              d = _mm256_sub_ps(d, y.f);

              // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
              // Perform Horner's method to evaluate interpolating polynomial.
#if 1
              y.f = _mm256_fmadd_ps(d, _mm256_set1_ps(SCEXP4), _mm256_set1_ps(SCEXP3)); 
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP2));
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP1));
              y.f = _mm256_fmadd_ps(y.f, d, _mm256_set1_ps(SCEXP0));
#else
              y.f = _mm256_mul_ps(d, _mm256_set1_ps(SCEXP4));      /* for x^4 term */
              y.f = _mm256_add_ps(y.f, _mm256_set1_ps(SCEXP3));    /* for x^3 term */
              y.f = _mm256_mul_ps(y.f, d);
              y.f = _mm256_add_ps(y.f, _mm256_set1_ps(SCEXP2));    /* for x^2 term */
              y.f = _mm256_mul_ps(y.f, d);
              y.f = _mm256_add_ps(y.f, _mm256_set1_ps(SCEXP1));    /* for x^1 term */
              y.f = _mm256_mul_ps(y.f, d);
              y.f = _mm256_add_ps(y.f, _mm256_set1_ps(SCEXP0));    /* for x^0 term */
#endif

              // Calculate 2^N exactly by directly manipulating floating point exponent,
              // then use it to scale y for the final result.
              // We need AVX2 instructions to be able to operate on 
              // 8-wide integer types efficiently.
              n.i = _mm256_sub_epi32(_mm256_set1_epi32(EXPOBIAS), n.i);
              n.i = _mm256_slli_epi32(n.i, EXPOSHIFT);
              y.f = _mm256_mul_ps(y.f, n.f);
              y.f = _mm256_mul_ps(y.f, atomicnumfactor_8); // MDFF density maps
#endif

              // At present, we do unaligned loads/stores since we can't guarantee
              // that the X-dimension is always a multiple of 8.
              float *ufptr = &densitymap[addr + x];
              d = _mm256_loadu_ps(ufptr); 
              _mm256_storeu_ps(ufptr, _mm256_add_ps(d, y.f)); 
            }
          }
#endif


#if VMDQSURFUSESSE && defined(__SSE2__)
          // Use SSE when we have a multiple-of-4 to compute
          // finish all remaining density map points with regular non-SSE loop
          if (usesse) {
            __align(16) SSEreg n;
            __align(16) SSEreg y;
            __m128 dy2dz2_4 = _mm_set1_ps(dy2dz2);
            __m128 dx_4 = _mm_add_ps(_mm_set1_ps(dx), _mm_load_ps(&sxdelta4[0]));

            for (; (x+3)<=xmax; x+=4,dx_4=_mm_add_ps(dx_4, gridspacing4_4)) {
              __m128 r2 = _mm_add_ps(_mm_mul_ps(dx_4, dx_4), dy2dz2_4);
              __m128 d;
#if VMDUSESVMLEXP
              // use Intel's SVML exp2() routine
              y.f = _mm_exp2_ps(_mm_mul_ps(r2, arinv_4));
#else
              // use our (much faster) fully inlined exponential approximation
              y.f = _mm_mul_ps(r2, arinv_4);         /* already negated and in base 2 */
              n.i = _mm_cvttps_epi32(y.f);
              d = _mm_cvtepi32_ps(n.i);
              d = _mm_sub_ps(d, y.f);

              // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
              // Perform Horner's method to evaluate interpolating polynomial.
#if 0
              // SSE 4.x FMADD instructions are not universally available
              y.f = _mm_fmadd_ps(d, _mm_set1_ps(SCEXP4), _mm_set1_ps(SCEXP3)); 
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP2));
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP1));
              y.f = _mm_fmadd_ps(y.f, d, _mm_set1_ps(SCEXP0));
#else
              y.f = _mm_mul_ps(d, _mm_set_ps1(SCEXP4));      /* for x^4 term */
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP3));    /* for x^3 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP2));    /* for x^2 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP1));    /* for x^1 term */
              y.f = _mm_mul_ps(y.f, d);
              y.f = _mm_add_ps(y.f, _mm_set_ps1(SCEXP0));    /* for x^0 term */
#endif

              // Calculate 2^N exactly by directly manipulating floating point exponent,
              // then use it to scale y for the final result.
              n.i = _mm_sub_epi32(_mm_set1_epi32(EXPOBIAS), n.i);
              n.i = _mm_slli_epi32(n.i, EXPOSHIFT);
              y.f = _mm_mul_ps(y.f, n.f);
              y.f = _mm_mul_ps(y.f, atomicnumfactor_4); // MDFF density maps
#endif

              // At present, we do unaligned loads/stores since we can't guarantee
              // that the X-dimension is always a multiple of 4.
              float *ufptr = &densitymap[addr + x];
              d = _mm_loadu_ps(ufptr); 
              _mm_storeu_ps(ufptr, _mm_add_ps(d, y.f)); 
            }
          }
#endif


#if VMDQSURFUSEVSX && defined(__VEC__)
          // Use VSX when we have a multiple-of-4 to compute
          // finish all remaining density map points with regular non-VSX loop
          //
          // XXX it may be useful to compare the speed/accuracy of the
          // polynomial approximation vs. the hardware-provided 
          // exp2f() approximation: vec_expte()
          //
          if (usevsx) {
            vector float dy2dz2_4 = vec_splats(dy2dz2);
            vector float tmpvsxdelta4 = *((__vector float *) &sxdelta4[0]);
            vector float dx_4 = vec_add(vec_splats(dx), tmpvsxdelta4);

            for (; (x+3)<=xmax; x+=4,dx_4=vec_add(dx_4, gridspacing4_4)) {
              vector float r2 = vec_add(vec_mul(dx_4, dx_4), dy2dz2_4);

              // use our (much faster) fully inlined exponential approximation
              vector float mb = vec_mul(r2, arinv_4);   /* already negated and in base 2 */
              vector float mbflr = vec_floor(mb);
              vector float d = vec_sub(mbflr, mb);
              vector float y;

              // Approximate 2^{-d}, 0 <= d < 1, by interpolation.
              // Perform Horner's method to evaluate interpolating polynomial.
              y = vec_madd(d, vec_splats(SCEXP4), vec_splats(SCEXP3)); // x^4
              y = vec_madd(y, d, vec_splats(SCEXP2)); // x^2 
              y = vec_madd(y, d, vec_splats(SCEXP1)); // x^1 
              y = vec_madd(y, d, vec_splats(SCEXP0)); // x^0 

              // Calculate 2^N exactly via vec_expte()
              // then use it to scale y for the final result.
              y = vec_mul(y, vec_expte(-mbflr));
              y = vec_mul(y, atomicnumfactor_4); // MDFF density maps

              // At present, we do unaligned loads/stores since we can't 
              // guarantee that the X-dimension is always a multiple of 4.
              float *ufptr = &densitymap[addr + x];
              d = *((__vector float *) &ufptr[0]);
              // XXX there must be a cleaner way to implement this
              // d = _mm_loadu_ps(ufptr); 
              // _mm_storeu_ps(ufptr, _mm_add_ps(d, y.f)); 
              d = vec_add(d, y);

              ufptr[0] = d[0];
              ufptr[1] = d[1];
              ufptr[2] = d[2];
              ufptr[3] = d[3];
            }
          }
#endif

          // finish all remaining density map points with regular non-SSE loop
          for (; x<=xmax; x++,dx+=gridspacing) {
            float r2 = dx*dx + dy2dz2;

            // use our (much faster) fully inlined exponential approximation
            float mb = r2 * arinv;         /* already negated and in base 2 */
            int mbflr = (int) mb;          /* get int part, floor() */
            float d = mbflr - mb;          /* remaining exponent, -1 < d <= 0 */

            /* approx with linear blend of Taylor polys */
            float sy = SCEXP0 + d*(SCEXP1 + d*(SCEXP2 + d*(SCEXP3 + d*SCEXP4)));

            /* 2^(-mbflr) */
            flint scalfac;
            scalfac.n = (EXPOBIAS - mbflr) << EXPOSHIFT;  

            // XXX assume we are never beyond the cutoff value in this loop
            float density = (sy * scalfac.f);

            density *= atomicnumfactor; // MDFF Cryo-EM atomic number density

            densitymap[addr + x] += density;
          }
        }
      }
    }
  }
}


typedef struct {
  int verbose;
  int natoms;
  float radscale;
  float gridspacing;
  float isovalue;
  float gausslim;
  const int *numvoxels;
  const float *xyzr; 
  const float *atomicnum;
  const float *colors;
  float **thrdensitymaps;
  float **thrvoltexmaps;
} densitythrparms;


static void * densitythread(void *voidparms) {
  wkf_tasktile_t tile;
  densitythrparms *parms = NULL;
  int threadid;

  wkf_threadlaunch_getid(voidparms, &threadid, NULL);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

  while (wkf_threadlaunch_next_tile(voidparms, 16384, &tile) != WKF_SCHED_DONE) {
    int natoms = tile.end-tile.start;
    const float *atomicnum = (parms->atomicnum == NULL) ? NULL : &parms->atomicnum[tile.start]; 
    vmd_gaussdensity_opt(parms->verbose, natoms, 
                         &parms->xyzr[4L*tile.start],
                         atomicnum,
                         (parms->thrvoltexmaps[0]!=NULL) ? &parms->colors[4L*tile.start] : NULL,
                         parms->thrdensitymaps[threadid], 
                         parms->thrvoltexmaps[threadid], 
                         parms->numvoxels, 
                         parms->radscale, 
                         parms->gridspacing, 
                         parms->isovalue, 
                         parms->gausslim);
  }

  return NULL;
}


static void * reductionthread(void *voidparms) {
  wkf_tasktile_t tile;
  densitythrparms *parms = NULL;
  int threadid, numthreads;

  wkf_threadlaunch_getid(voidparms, &threadid, &numthreads);
  wkf_threadlaunch_getdata(voidparms, (void **) &parms);

  while (wkf_threadlaunch_next_tile(voidparms, 16384, &tile) != WKF_SCHED_DONE) {
    // do a reduction over each of the individual density grids
    int i, x;
    for (x=tile.start; x<tile.end; x++) {
      float tmp = 0.0f;
      for (i=1; i<numthreads; i++) {
        tmp += parms->thrdensitymaps[i][x];
      }
      parms->thrdensitymaps[0][x] += tmp;
    }

    // do a reduction over each of the individual texture grids
    if (parms->thrvoltexmaps[0] != NULL) {
      for (x=tile.start*3L; x<tile.end*3L; x++) {
        float tmp = 0.0f;
        for (i=1; i<numthreads; i++) {
          tmp += parms->thrvoltexmaps[i][x];
        }
        parms->thrvoltexmaps[0][x] += tmp;
      }
    }
  }

  return NULL;
}


static int vmd_gaussdensity_threaded(int verbose, 
                                     int natoms, const float *xyzr,
                                     const float *atomicnum,
                                     const float *colors,
                                     float *densitymap, float *voltexmap, 
                                     const int *numvoxels, 
                                     float radscale, float gridspacing, 
                                     float isovalue, float gausslim) {
  densitythrparms parms;
  memset(&parms, 0, sizeof(parms));

  parms.verbose = verbose;
  parms.natoms = natoms;
  parms.radscale = radscale;
  parms.gridspacing = gridspacing;
  parms.isovalue = isovalue;
  parms.gausslim = gausslim;
  parms.numvoxels = numvoxels;
  parms.xyzr = xyzr;
  parms.atomicnum = atomicnum;
  parms.colors = colors;
 
  int physprocs = wkf_thread_numprocessors();
  int maxprocs = physprocs;

  // We can productively use only a few cores per socket due to the
  // limited memory bandwidth per socket. Also, hyperthreading
  // actually hurts performance.  These two considerations combined
  // with the linear increase in memory use prevent us from using large
  // numbers of cores with this simple approach, so if we've got more 
  // than 8 CPU cores, we'll iteratively cutting the core count in 
  // half until we're under 8 cores.
  while (maxprocs > 8) 
    maxprocs /= 2;

  // Limit the number of CPU cores used so we don't run the 
  // machine out of memory during surface computation.
  // Use either a dynamic or hard-coded heuristic to limit the
  // number of CPU threads we will spawn so that we don't run
  // the machine out of memory.  
  long volsz = numvoxels[0] * numvoxels[1] * numvoxels[2];
  long volmemsz = sizeof(float) * volsz;
  long volmemszkb = volmemsz / 1024;
  long volmemtexszkb = volmemszkb + ((voltexmap != NULL) ? 3L*volmemszkb : 0);

  // Platforms that don't have a means of determining available
  // physical memory will return -1, in which case we fall back to the
  // simple hard-coded 2GB-max-per-core heuristic.
  long vmdcorefree = -1;

#if defined(ARCH_BLUEWATERS) || defined(ARCH_CRAY_XC) || defined(ARCH_CRAY_XK) || defined(ARCH_LINUXAMD64) || defined(ARCH_SOLARIS2_64) || defined(ARCH_SOLARISX86_64) || defined(ARCH_AIX6_64) || defined(ARCH_MACOSXX86_64) 
  // XXX The core-free query scheme has one weakness in that we might have a 
  // 32-bit version of VMD running on a 64-bit machine, where the available
  // physical memory may be much larger than is possible for a 
  // 32-bit VMD process to address.  To do this properly we must therefore
  // use conditional compilation safety checks here until we  have a better
  // way of determining this with a standardized helper routine.
  vmdcorefree = vmd_get_avail_physmem_mb();
#endif

  if (vmdcorefree >= 0) {
    // Make sure QuickSurf uses no more than a fraction of the free memory
    // as an upper bound alternative to the hard-coded heuristic.
    // This should be highly preferable to the fixed-size heuristic
    // we had used in all cases previously.
    while ((volmemtexszkb * maxprocs) > (1024L*vmdcorefree/4)) {
      maxprocs /= 2;
    }
  } else {
    // Set a practical per-core maximum memory use limit to 2GB, for all cores
    while ((volmemtexszkb * maxprocs) > (2L * 1024L * 1024L))
      maxprocs /= 2;
  }

  if (maxprocs < 1) 
    maxprocs = 1;

  // Loop over number of physical processors and try to create 
  // per-thread volumetric maps for each of them.
  parms.thrdensitymaps = (float **) calloc(1,maxprocs * sizeof(float *));
  parms.thrvoltexmaps = (float **) calloc(1, maxprocs * sizeof(float *));

  // first thread is already ready to go
  parms.thrdensitymaps[0] = densitymap;
  parms.thrvoltexmaps[0] = voltexmap;

  int i;
  int numprocs = maxprocs; // ever the optimist
  for (i=1; i<maxprocs; i++) {
    parms.thrdensitymaps[i] = (float *) calloc(1, volmemsz);
    if (parms.thrdensitymaps[i] == NULL) {
      numprocs = i;
      break;
    }
    if (voltexmap != NULL) {
      parms.thrvoltexmaps[i] = (float *) calloc(1, 3L * volmemsz);
      if (parms.thrvoltexmaps[i] == NULL) {
        free(parms.thrdensitymaps[i]);
        parms.thrdensitymaps[i] = NULL;
        numprocs = i;
        break;
      }
    }
  }

  // launch independent thread calculations
  wkf_tasktile_t tile;
  tile.start = 0;
  tile.end = natoms;
  wkf_threadlaunch(numprocs, &parms, densitythread, &tile);

  // do a parallel reduction of the resulting density maps
  tile.start = 0;
  tile.end = volsz;
  wkf_threadlaunch(numprocs, &parms, reductionthread, &tile);

  // free work area
  for (i=1; i<maxprocs; i++) {
    if (parms.thrdensitymaps[i] != NULL)
      free(parms.thrdensitymaps[i]);

    if (parms.thrvoltexmaps[i] != NULL)
      free(parms.thrvoltexmaps[i]);
  }
  free(parms.thrdensitymaps);
  free(parms.thrvoltexmaps);

  return 0;
}

QuickSurf::QuickSurf(int forcecpuonly) {
  volmap = NULL;
  voltexmap = NULL;
  s.clear();
  isovalue = 0.5f;

  numvoxels[0] = 128;
  numvoxels[1] = 128;
  numvoxels[2] = 128;

  origin[0] = 0.0f;
  origin[1] = 0.0f;
  origin[2] = 0.0f;

  xaxis[0] = 1.0f;
  xaxis[1] = 0.0f;
  xaxis[2] = 0.0f;

  yaxis[0] = 0.0f;
  yaxis[1] = 1.0f;
  yaxis[2] = 0.0f;

  zaxis[0] = 0.0f;
  zaxis[1] = 0.0f;
  zaxis[2] = 1.0f;
   
  cudaqs = NULL;
  force_cpuonly = forcecpuonly;
#if defined(VMDCUDA)
  if (!force_cpuonly && !getenv("VMDNOCUDA")) {
    cudaqs = new CUDAQuickSurf();
  }
#endif

  timer = wkf_timer_create();
}


void QuickSurf::free_gpu_memory(void) {
  if (cudaqs) {
#if defined(VMDCUDA)
    delete cudaqs; ///< Free any GPU memory that's in use, despite perf hit...
#endif
    cudaqs = NULL; ///< mark CUDA object NULL for next calculation...
  }
}


int QuickSurf::calc_surf(AtomSel *atomSel, DrawMolecule *mol,
                         const float *atompos, const float *atomradii,
                         int quality, float radscale, float gridspacing,
                         float isoval, const int *colidx, const float *cmap,
                         VMDDisplayList *cmdList) {
  PROFILE_PUSH_RANGE("QuickSurf", 3);

  wkf_timer_start(timer);
  int colorperatom = (colidx != NULL && cmap != NULL);
  int usebeads=0;

  int verbose = (getenv("VMDQUICKSURFVERBOSE") != NULL);

  // Disable MDFF atomic number weighted densities until we implement
  // GUI controls for this if it turns out to be useful for more than
  // than just analytical usage.
  const float *atomicnum = NULL;

  // clean up any existing CPU arrays before going any further...
  if (voltexmap != NULL)
    free(voltexmap);
  voltexmap = NULL;

  ResizeArray<float> beadpos(64 + (3L * atomSel->selected) / 20);
  ResizeArray<float> beadradii(64 + (3L * atomSel->selected) / 20);
  ResizeArray<float> beadcolors(64 + (3L * atomSel->selected) / 20);

  if (getenv("VMDQUICKSURFBEADS")) {
    usebeads=1;
    if (verbose)
      printf("QuickSurf using residue beads representation...\n");
  }

  int numbeads = 0;
  if (usebeads) {
    int i, resid, numres;

    // draw a bead for each residue
    numres = mol->residueList.num();
    for (resid=0; resid<numres; resid++) {
      float com[3] = {0.0, 0.0, 0.0};
      const ResizeArray<int> &atoms = mol->residueList[resid]->atoms;
      int numatoms = atoms.num();
      int oncount = 0;
   
      // find COM for residue
      for (i=0; i<numatoms; i++) {
        int idx = atoms[i];
        if (atomSel->on[idx]) {
          oncount++;
          vec_add(com, com, atompos + 3L*idx);
        }
      }

      if (oncount < 1)
        continue; // exit if there weren't any atoms

      vec_scale(com, 1.0f / (float) oncount, com);

      // find radius of bounding sphere and save last atom index for color
      int atomcolorindex=0; // initialize, to please compilers
      float boundradsq = 0.0f;
      for (i=0; i<numatoms; i++) {
        int idx = atoms[i];
        if (atomSel->on[idx]) {
          float tmpdist[3];
          atomcolorindex = idx;
          vec_sub(tmpdist, com, atompos + 3L*idx);
          float distsq = dot_prod(tmpdist, tmpdist);
          if (distsq > boundradsq) {
            boundradsq = distsq;
          }
        }
      }
      beadpos.append3(&com[0]);
      beadradii.append(sqrtf(boundradsq) + 1.0f);

      if (colorperatom) {
        const float *cp = &cmap[colidx[atomcolorindex] * 3L];
        beadcolors.append3(&cp[0]);
      }

      // XXX still need to add pick points...
    }

    numbeads = beadpos.num() / 3;
  }

  // initialize class variables
  isovalue=isoval;

  // If no volumetric texture will be computed we will use the cmap
  // parameter to pass in the solid color to be applied to all vertices
  // Since QS can now also be called by MDFF, we have to check whether
  // display related parms are set or not before using them.
  if (cmap != NULL)
    vec_copy(solidcolor, cmap);

  // compute min/max atom radius, build list of selected atom radii,
  // and compute bounding box for the selected atoms
  float minx, miny, minz, maxx, maxy, maxz;
  float minrad, maxrad;
  int i;
  if (usebeads) {
    minx = maxx = beadpos[0];
    miny = maxy = beadpos[1];
    minz = maxz = beadpos[2];
    minrad = maxrad = beadradii[0];
    for (i=0; i<numbeads; i++) {
      long ind = i * 3L;
      float tmpx = beadpos[ind  ];
      float tmpy = beadpos[ind+1];
      float tmpz = beadpos[ind+2];

      minx = (tmpx < minx) ? tmpx : minx;
      maxx = (tmpx > maxx) ? tmpx : maxx;

      miny = (tmpy < miny) ? tmpy : miny;
      maxy = (tmpy > maxy) ? tmpy : maxy;

      minz = (tmpz < minz) ? tmpz : minz;
      maxz = (tmpz > maxz) ? tmpz : maxz;
 
      // we always have to compute the rmin/rmax for beads
      // since these radii are defined on-the-fly
      float r = beadradii[i];
      minrad = (r < minrad) ? r : minrad;
      maxrad = (r > maxrad) ? r : maxrad;
    }
  } else {
    minx = maxx = atompos[atomSel->firstsel*3L  ];
    miny = maxy = atompos[atomSel->firstsel*3L+1];
    minz = maxz = atompos[atomSel->firstsel*3L+2];

    // Query min/max atom radii for the entire molecule
    mol->get_radii_minmax(minrad, maxrad);

    // We only compute rmin/rmax for the actual group of selected atoms if 
    // (rmax/rmin > 2.5) for the whole molecule, otherwise it's a small 
    // enough range that we don't care since it won't hurt our performance. 
    if (minrad <= 0.001 || maxrad/minrad > 2.5) {
      minrad = maxrad = atomradii[atomSel->firstsel];
      for (i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
        if (atomSel->on[i]) {
          long ind = i * 3L;
          float tmpx = atompos[ind  ];
          float tmpy = atompos[ind+1];
          float tmpz = atompos[ind+2];

          minx = (tmpx < minx) ? tmpx : minx;
          maxx = (tmpx > maxx) ? tmpx : maxx;

          miny = (tmpy < miny) ? tmpy : miny;
          maxy = (tmpy > maxy) ? tmpy : maxy;

          minz = (tmpz < minz) ? tmpz : minz;
          maxz = (tmpz > maxz) ? tmpz : maxz;
  
          float r = atomradii[i];
          minrad = (r < minrad) ? r : minrad;
          maxrad = (r > maxrad) ? r : maxrad;
        }
      }
    } else {
#if 1
      float fmin[3], fmax[3];
      minmax_selected_3fv_aligned(atompos, atomSel->on, atomSel->num_atoms,
                                  atomSel->firstsel, atomSel->lastsel,
                                  fmin, fmax);
      minx = fmin[0];
      miny = fmin[1];
      minz = fmin[2];

      maxx = fmax[0]; 
      maxy = fmax[1]; 
      maxz = fmax[2]; 
#else
      for (i=atomSel->firstsel; i<=atomSel->lastsel; i++) {
        if (atomSel->on[i]) {
          long ind = i * 3L;
          float tmpx = atompos[ind  ];
          float tmpy = atompos[ind+1];
          float tmpz = atompos[ind+2];

          minx = (tmpx < minx) ? tmpx : minx;
          maxx = (tmpx > maxx) ? tmpx : maxx;

          miny = (tmpy < miny) ? tmpy : miny;
          maxy = (tmpy > maxy) ? tmpy : maxy;

          minz = (tmpz < minz) ? tmpz : minz;
          maxz = (tmpz > maxz) ? tmpz : maxz;
        }
      }
#endif
    }
  }

  float mincoord[3], maxcoord[3];
  mincoord[0] = minx;
  mincoord[1] = miny;
  mincoord[2] = minz;
  maxcoord[0] = maxx;
  maxcoord[1] = maxy;
  maxcoord[2] = maxz;

  // crude estimate of the grid padding we require to prevent the
  // resulting isosurface from being clipped
  float gridpadding = radscale * maxrad * 1.70f;
  float padrad = gridpadding;
  padrad = 0.65f * sqrtf(4.0f/3.0f*((float) VMD_PI)*padrad*padrad*padrad);
  gridpadding = MAX(gridpadding, padrad);

  // Handle coarse-grained structures and whole-cell models
  // XXX The switch at 4.0A from an assumed all-atom scale structure to 
  //     CG or cell models is a simple heuristic at a somewhat arbitrary 
  //     threshold value.  
  //     For all-atom models the units shown in the GUI are in Angstroms
  //     and are absolute, but for CG or cell models the units in the GUI 
  //     are relative to the atom with the minimum radius.
  //     This code doesn't do anything to handle structures with a minrad 
  //     of zero, where perhaps only one particle has an unset radius.
  if (minrad > 4.0f) {
    gridspacing *= minrad;
  }

  if (verbose) {
    printf("QuickSurf: R*%.1f, I=%.1f, H=%.1f Pad: %.1f minR: %.1f maxR: %.1f)\n",
           radscale, isovalue, gridspacing, gridpadding, minrad, maxrad);
  }

  mincoord[0] -= gridpadding;
  mincoord[1] -= gridpadding;
  mincoord[2] -= gridpadding;
  maxcoord[0] += gridpadding;
  maxcoord[1] += gridpadding;
  maxcoord[2] += gridpadding;

  // compute the real grid dimensions from the selected atoms
  numvoxels[0] = (int) ceil((maxcoord[0]-mincoord[0]) / gridspacing);
  numvoxels[1] = (int) ceil((maxcoord[1]-mincoord[1]) / gridspacing);
  numvoxels[2] = (int) ceil((maxcoord[2]-mincoord[2]) / gridspacing);

  // recalc the grid dimensions from rounded/padded voxel counts
  xaxis[0] = (numvoxels[0]-1) * gridspacing;
  yaxis[1] = (numvoxels[1]-1) * gridspacing;
  zaxis[2] = (numvoxels[2]-1) * gridspacing;
  maxcoord[0] = mincoord[0] + xaxis[0];
  maxcoord[1] = mincoord[1] + yaxis[1];
  maxcoord[2] = mincoord[2] + zaxis[2];

  if (verbose) {
    printf("  GridSZ: (%4d %4d %4d)  BBox: (%.1f %.1f %.1f)->(%.1f %.1f %.1f)\n",
           numvoxels[0], numvoxels[1], numvoxels[2],
           mincoord[0], mincoord[1], mincoord[2],
           maxcoord[0], maxcoord[1], maxcoord[2]);
  }

  vec_copy(origin, mincoord);

  // build compacted lists of bead coordinates, radii, and colors
  float *xyzr = NULL;
  float *colors = NULL;
  if (usebeads) { 
    int ind =0;
    int ind4=0; 
    xyzr = (float *) malloc(numbeads * sizeof(float) * 4L);
    if (colorperatom) {
      colors = (float *) malloc(numbeads * sizeof(float) * 4L);

      // build compacted lists of bead coordinates, radii, and colors
      for (i=0; i<numbeads; i++) {
        const float *fp = &beadpos[0] + ind;
        xyzr[ind4    ] = fp[0]-origin[0];
        xyzr[ind4 + 1] = fp[1]-origin[1];
        xyzr[ind4 + 2] = fp[2]-origin[2];
        xyzr[ind4 + 3] = beadradii[i];
 
        const float *cp = &beadcolors[0] + ind;
        colors[ind4    ] = cp[0];
        colors[ind4 + 1] = cp[1];
        colors[ind4 + 2] = cp[2];
        colors[ind4 + 3] = 1.0f;
        ind4 += 4;
        ind += 3;
      }
    } else {
      // build compacted lists of bead coordinates and radii only
      for (i=0; i<numbeads; i++) {
        const float *fp = &beadpos[0] + ind;
        xyzr[ind4    ] = fp[0]-origin[0];
        xyzr[ind4 + 1] = fp[1]-origin[1];
        xyzr[ind4 + 2] = fp[2]-origin[2];
        xyzr[ind4 + 3] = beadradii[i];
        ind4 += 4;
        ind += 3;
      }
    }
  } else {
    long ind = atomSel->firstsel * 3L;
    long ind4=0; 
    xyzr = (float *) malloc(atomSel->selected * sizeof(float) * 4L);
    if (colorperatom) {
      colors = (float *) malloc(atomSel->selected * sizeof(float) * 4L);

      // build compacted lists of atom coordinates, radii, and colors
      for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
        if (atomSel->on[i]) {
          const float *fp = atompos + ind;
          xyzr[ind4    ] = fp[0]-origin[0];
          xyzr[ind4 + 1] = fp[1]-origin[1];
          xyzr[ind4 + 2] = fp[2]-origin[2];
          xyzr[ind4 + 3] = atomradii[i];
 
          const float *cp = &cmap[colidx[i] * 3L];
          colors[ind4    ] = cp[0];
          colors[ind4 + 1] = cp[1];
          colors[ind4 + 2] = cp[2];
          colors[ind4 + 3] = 1.0f;
          ind4 += 4;
        }
        ind += 3;
      }
    } else {
      // build compacted lists of atom coordinates and radii only
      for (i=atomSel->firstsel; i <= atomSel->lastsel; i++) {
        if (atomSel->on[i]) {
          const float *fp = atompos + ind;
          xyzr[ind4    ] = fp[0]-origin[0];
          xyzr[ind4 + 1] = fp[1]-origin[1];
          xyzr[ind4 + 2] = fp[2]-origin[2];
          xyzr[ind4 + 3] = atomradii[i];
          ind4 += 4;
        }
        ind += 3;
      }
    }
  }

  // set gaussian window size based on user-specified quality parameter
  float gausslim = 2.0f;
  switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality

    case 2: gausslim = 3.0f; break; // high quality

    case 1: gausslim = 2.5f; break; // medium quality

    case 0: 
    default: gausslim = 2.0f; // low quality
      break;
  }

  pretime = wkf_timer_timenow(timer);

#if defined(VMDCUDA)
  if (!force_cpuonly && !getenv("VMDNOCUDA")) {
    // allocate a new CUDAQuickSurf object if we destroyed the old one...
    if (cudaqs == NULL)
      cudaqs = new CUDAQuickSurf();

    // compute both density map and floating point color texture map
    int pcount = (usebeads) ? numbeads : atomSel->selected; 
    int rc = cudaqs->calc_surf(pcount, &xyzr[0],
                               (colorperatom) ? &colors[0] : &cmap[0],
                               colorperatom, origin, numvoxels, maxrad,
                               radscale, gridspacing, isovalue, gausslim,
                               cmdList);

    // If we're running in a memory-limited scenario, we can force
    // VMD to dump the QuickSurf GPU data to prevent out-of-memory
    // problems later on, either during other surface calcs or when 
    // using the GPU for things like OptiX ray tracing
    if (getenv("VMDQUICKSURFMINMEM")) {
      free_gpu_memory();
    }

    if (rc == 0) {
      free(xyzr);
      if (colors)
        free(colors);
  
      voltime = wkf_timer_timenow(timer);

      PROFILE_POP_RANGE(); // first return point

      return 0;
    }
  }
#endif

  if (verbose) {
    printf("  Computing density map grid on CPUs ");
  }

  long volsz = numvoxels[0] * numvoxels[1] * numvoxels[2];
  volmap = new float[volsz];
  if (colidx != NULL && cmap != NULL) {
    voltexmap = (float*) calloc(1, 3L * sizeof(float) * numvoxels[0] * numvoxels[1] * numvoxels[2]);
  }

  fflush(stdout);
  memset(volmap, 0, sizeof(float) * volsz);
  if ((volsz * atomSel->selected) > 20000000) {
    vmd_gaussdensity_threaded(verbose, atomSel->selected, &xyzr[0], atomicnum,
                              (voltexmap!=NULL) ? &colors[0] : NULL,
                              volmap, voltexmap, numvoxels, radscale, 
                              gridspacing, isovalue, gausslim);
  } else {
    vmd_gaussdensity_opt(verbose, atomSel->selected, &xyzr[0], atomicnum,
                         (voltexmap!=NULL) ? &colors[0] : NULL,
                         volmap, voltexmap, 
                         numvoxels, radscale, gridspacing, isovalue, gausslim);
  }

  free(xyzr);
  if (colors)
    free(colors);

  voltime = wkf_timer_timenow(timer);

  // draw the surface if the caller provided the display list
  if (cmdList != NULL) {
    draw_trimesh(cmdList);
  }

  if (verbose) {
    printf(" Done.\n");
  }

  PROFILE_POP_RANGE(); // second return point

  return 0;
}


// compute synthetic density map, but nothing else
VolumetricData * QuickSurf::calc_density_map(AtomSel * atomSel, 
                                             DrawMolecule *mymol,  
                                             const float *atompos, 
                                             const float *atomradii,
                                             int quality, float radscale, 
                                             float gridspacing) {
  if (!calc_surf(atomSel, mymol, atompos, atomradii, 
               quality, radscale, gridspacing, 1.0f, NULL, NULL, NULL)) {
    VolumetricData *surfvol;
    surfvol = new VolumetricData("density map", origin, xaxis, yaxis, zaxis,
                                 numvoxels[0], numvoxels[1], numvoxels[2],
                                 volmap);
    return surfvol;
  }

  return NULL;
}


// Extract the isosurface from the QuickSurf density map
int QuickSurf::get_trimesh(int &numverts, 
                           float *&v3fv, float *&n3fv, float *&c3fv, 
                           int &numfacets, int *&fiv) {

  int verbose = (getenv("VMDQUICKSURFVERBOSE") != NULL);

  if (verbose)
    printf("Running marching cubes on CPU...\n");

  VolumetricData *surfvol; ///< Container used to generate isosurface on the CPU
  surfvol = new VolumetricData("molecular surface",
                               origin, xaxis, yaxis, zaxis,
                               numvoxels[0], numvoxels[1], numvoxels[2],
                               volmap);

  // XXX we should calculate the volume gradient only for those
  //     vertices we extract, since for this rep any changes to settings
  //     will require recomputation of the entire volume
  surfvol->compute_volume_gradient(); // calc gradients: smooth vertex normals
  gradtime = wkf_timer_timenow(timer);

  // trimesh polygonalized surface, max of 6 triangles per voxel
  const int stepsize = 1;
  s.clear();                              // initialize isosurface data
  s.compute(surfvol, isovalue, stepsize); // compute the isosurface

  mctime = wkf_timer_timenow(timer);

  s.vertexfusion(9, 9);                   // eliminate duplicated vertices
  s.normalize();                          // normalize interpolated gradient/surface normals

  if (s.numtriangles > 0) {
    if (voltexmap != NULL) {
      // assign per-vertex colors by a 3-D texture map
      s.set_color_voltex_rgb3fv(voltexmap);
    } else {
      // use a single color for the entire mesh
      s.set_color_rgb3fv(solidcolor);
    }
  }

  numverts = s.v.num() / 3;
  v3fv=&s.v[0];
  n3fv=&s.n[0];
  c3fv=&s.c[0];

  numfacets = s.numtriangles;
  fiv=&s.f[0];

  delete surfvol;

  mcverttime = wkf_timer_timenow(timer);
  reptime = mcverttime;

  if (verbose) {
    char strmsg[1024];
    sprintf(strmsg, "QuickSurf: %.3f [pre:%.3f vol:%.3f gr:%.3f mc:%.2f mcv:%.3f]",
            reptime, pretime, voltime-pretime, gradtime-voltime, 
            mctime-gradtime, mcverttime-mctime);

    msgInfo << strmsg << sendmsg;
  }
 
  return 0;
}


int QuickSurf::draw_trimesh(VMDDisplayList *cmdList) {
  DispCmdTriMesh cmdTriMesh;

  int numverts=0;
  float *v=NULL, *n=NULL, *c=NULL;
  int numfacets=0;
  int *f=NULL;

  get_trimesh(numverts, v, n, c, numfacets, f);

  // Create a triangle mesh
  if (numfacets > 0) {
    cmdTriMesh.putdata(v, n, c, numverts, f, numfacets, 0, cmdList);
  }
 
  return 0;
}


QuickSurf::~QuickSurf() {
#if defined(VMDCUDA)
  free_gpu_memory();
#endif

  if (voltexmap != NULL)
    free(voltexmap);
  voltexmap = NULL;

  wkf_timer_destroy(timer);
}


