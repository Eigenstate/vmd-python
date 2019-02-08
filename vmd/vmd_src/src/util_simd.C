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
 *	$RCSfile: util_simd.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.13 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Hand-coded SIMD loops using compiler provided intrinsics, or inline
 * assembly code to generate highly optimized machine code for time-critical
 * loops that crop up commonly used features of VMD. 
 *
 ***************************************************************************/

// pgcc has troubles with hand-vectorized x86 intrinsics
#if !defined(__PGIC__)
#define VMDUSESSE 1
// #define VMDUSEVSX 1
// #define VMDUSEAVX 1
#endif
// #define VMDUSENEON 1

#if defined(VMDUSESSE) && defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(VMDUSEAVX) && defined(__AVX__)
#include <immintrin.h>
#endif
#if defined(VMDUSENEON) && defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if (defined(VMDUSEVSX) && defined(__VSX__))
#if defined(__GNUC__) && defined(__VEC__)
#include <altivec.h>
#endif
#endif

// #include <string.h>
// #include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#endif // _MSC_VER


#if 0
//
// XXX array init/copy routines that avoid polluting cache, where possible
//
// Fast 16-byte-aligned integer assignment loop for use in the
// VMD color scale routines
void set_1fv_aligned(const int *iv, int n, const int val) {
  int i=0;

#if defined(VMDUSESSE) && defined(__SSE2__)
  __m128i = _mm_set_p
  // do groups of four elements
  for (; i<(n-3); i+=4) {
  }
#endif
}
#endif


#if defined(VMDUSESSE) || defined(VMDUSEAVX) || defined(VMDUSEVSX) || defined(VMDUSENEON)

//
// Helper routine for use when coping with unaligned
// buffers returned by malloc() on many GNU systems:
//   http://gcc.gnu.org/bugzilla/show_bug.cgi?id=24261
//   http://www.sourceware.org/bugzilla/show_bug.cgi?id=206
//
// XXX until all compilers support uintptr_t, we have to do 
//     dangerous and ugly things with pointer casting here...
//
#if 1
/* sizeof(unsigned long) == sizeof(void*) */
#define myintptrtype unsigned long
#elif 1
/* sizeof(size_t) == sizeof(void*) */
#define myintptrtype size_t
#else
/* C99 */
#define myintptrtype uintptr_t
#endif

#if 0
// arbitrary pointer alignment test
static int is_Nbyte_aligned(const void *ptr, int N) {
  return ((((myintptrtype) ptr) % N) == 0);
}
#endif

#if (defined(VMDUSESSE) && defined(__SSE2__)) || (defined(VMDUSEVSX) && defined(__VSX__)) || (defined(VMDUSEAVX) && defined(__AVX__))
// Aligment test routine for x86 16-byte SSE vector instructions
static int is_16byte_aligned(const void *ptr) {
  return (((myintptrtype) ptr) == (((myintptrtype) ptr) & (~0xf)));
}
#endif

#if defined(VMDUSEAVX)
// Aligment test routine for x86 32-byte AVX vector instructions
static int is_32byte_aligned(const void *ptr) {
  return (((myintptrtype) ptr) == (((myintptrtype) ptr) & (~0x1f)));
}
#endif

#if 0
// Aligment test routine for x86 LRB/MIC 64-byte vector instructions
static int is_64byte_aligned(const void *ptr) {
  return (((myintptrtype) ptr) == (((myintptrtype) ptr) & (~0x3f)));
}
#endif
#endif 


//
// Small inlinable SSE helper routines to make code easier to read
//
#if defined(VMDUSESSE) && defined(__SSE2__)

#if 0
static void print_m128i(__m128i mask4) {
  int * iv = (int *) &mask4;
  printf("vec: %08x %08x %08x %08x\n", iv[0], iv[1], iv[2], iv[3]);
}

static int hand_m128i(__m128i mask4) {
  __m128i tmp = mask4;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1));
  tmp = _mm_and_si128(mask4, tmp);
  mask4 = tmp;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2));
  tmp = _mm_and_si128(mask4, tmp);
  mask4 = tmp; // all 4 elements are now set to the reduced mask

  int mask = _mm_cvtsi128_si32(mask4); // return zeroth element
  return mask;
}
#endif


static int hor_m128i(__m128i mask4) {
#if 0
  int mask = _mm_movemask_epi8(_mm_cmpeq_epi32(mask4, _mm_set1_epi32(1)));
#else
  __m128i tmp = mask4;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1));
  tmp = _mm_or_si128(mask4, tmp);
  mask4 = tmp;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2));
  tmp = _mm_or_si128(mask4, tmp);
  mask4 = tmp; // all 4 elements are now set to the reduced mask

  int mask = _mm_cvtsi128_si32(mask4); // return zeroth element
#endif
  return mask;
}


static int hadd_m128i(__m128i sum4) {
  __m128i tmp = sum4;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(2, 3, 0, 1));
  tmp = _mm_add_epi32(sum4, tmp);
  sum4 = tmp;
  tmp = _mm_shuffle_epi32(tmp, _MM_SHUFFLE(1, 0, 3, 2));
  tmp = _mm_add_epi32(sum4, tmp);
  sum4 = tmp; // all 4 elements are now set to the sum

  int sum = _mm_cvtsi128_si32(sum4); // return zeroth element
  return sum;
}


#if 0
static __m128i _mm_sel_m128i(const __m128i &a, const __m128i &b, const __m128i &mask) {
  // (((b ^ a) & mask)^a)
  return _mm_xor_si128(a, _mm_and_si128(mask, _mm_xor_si128(b, a)));
}
#endif


static __m128 _mm_sel_ps(const __m128 &a, const __m128 &b, const __m128 &mask) {
  // (((b ^ a) & mask)^a)
  return _mm_xor_ps(a, _mm_and_ps(mask, _mm_xor_ps(b, a)));
}


// helper routine to perform a min among all 4 elements of an __m128
static float fmin_m128(__m128 min4) {
  __m128 tmp;
  tmp = min4;
  tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(2, 3, 0, 1));
  tmp = _mm_min_ps(min4, tmp);
  min4 = tmp;
  tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2));
  tmp = _mm_min_ps(min4, tmp);
  min4 = tmp; // all 4 elements are now set to the min

  float fmin;
  _mm_store_ss(&fmin, min4);
  return fmin;
}


// helper routine to perform a max among all 4 elements of an __m128
static float fmax_m128(__m128 max4) {
  __m128 tmp = max4;
  tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(2, 3, 0, 1));
  tmp = _mm_max_ps(max4, tmp);
  max4 = tmp;
  tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(1, 0, 3, 2));
  tmp = _mm_max_ps(max4, tmp);
  max4 = tmp; // all 4 elements are now set to the max

  float fmax;
  _mm_store_ss(&fmax, max4);
  return fmax;
}
#endif


//
// Small inlinable ARM Neon helper routines to make code easier to read
//
#if defined(VMDUSENEON) && defined(__ARM_NEON__)

// helper routine to perform a min among all 4 elements of an __m128
static float fmin_f32x4(float32x4_t min4) {
  float *f1 = (float *) &min4;
  float min1 = f1[0];
  if (f1[1] < min1) min1 = f1[1];
  if (f1[2] < min1) min1 = f1[2];
  if (f1[3] < min1) min1 = f1[3];
  return min1;
}

static float fmax_f32x4(float32x4_t max4) {
  float *f1 = (float *) &max4;
  float max1 = f1[0];
  if (f1[1] > max1) max1 = f1[1];
  if (f1[2] > max1) max1 = f1[2];
  if (f1[3] > max1) max1 = f1[3];
  return max1;
}

#endif


// Find the first selected atom
int find_first_selection_aligned(int n, const int *on, int *firstsel) {
  int i;
  *firstsel = 0;

  // find the first selected atom, if any
#if defined(VMDUSEAVX) && defined(__AVX__)
  // roll up to the first 32-byte-aligned array index
  for (i=0; ((i<n) && !is_32byte_aligned(&on[i])); i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }

  // AVX vectorized search loop
  for (; i<(n-7); i+=8) {
    // aligned load of 8 selection flags
    __m256i on8 = _mm256_load_si256((__m256i*) &on[i]);
    if (!_mm256_testz_si256(on8, on8))
      break; // found a block containing the first selected atom
  }

  for (; i<n; i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }
#elif defined(VMDUSESSE) && defined(__SSE2__)
  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&on[i])); i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }

  // SSE vectorized search loop
  for (; i<(n-3); i+=4) {
    // aligned load of 4 selection flags
    __m128i on4 = _mm_load_si128((__m128i*) &on[i]);
    if (hor_m128i(on4))
      break; // found a block containing the first selected atom
  }

  for (; i<n; i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }
#elif 0 && (defined(VMDUSEVSX) && defined(__VSX__))
  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&on[i])); i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }

  // VSX vectorized search loop
  for (; i<(n-3); i+=4) {
    // aligned load of 4 selection flags
    __vector signed int on4 = *((__vector signed int *) &on[i]);
    if (vec_extract(vec_max(on4, on4), 0))
      break; // found a block containing the first selected atom
  }

  for (; i<n; i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }
#else
  // plain C...
  for (i=0; i<n; i++) {
    if (on[i]) {
      *firstsel = i; // found first selected atom
      return 0;
    }
  }
#endif

  // no atoms were selected if we got here
  *firstsel = 0;
  return -1;
}


// Find the last selected atom
int find_last_selection_aligned(int n, const int *on, int *lastsel) {
  int i;
  *lastsel =  -1;

  // find the last selected atom, if any
#if defined(VMDUSEAVX) && defined(__AVX__)
  // AVX vectorized search loop
  // Roll down to next 32-byte boundary
  for (i=n-1; i>=0; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }

    // drop out of the alignment loop once we hit a 32-byte boundary
    if (is_32byte_aligned(&on[i]))
      break;
  }

  for (i-=8; i>=0; i-=8) {
    // aligned load of 8 selection flags
    __m256i on8 = _mm256_load_si256((__m256i*) &on[i]);
    if (!_mm256_testz_si256(on8, on8))
      break; // found a block containing the last selected atom
  }

  int last8=i;
  for (i=last8+7; i>=last8; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }
  }
#elif defined(VMDUSESSE) && defined(__SSE2__)
  // SSE vectorized search loop
  // Roll down to next 16-byte boundary
  for (i=n-1; i>=0; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }

    // drop out of the alignment loop once we hit a 16-byte boundary
    if (is_16byte_aligned(&on[i]))
      break;
  }

  for (i-=4; i>=0; i-=4) {
    // aligned load of 4 selection flags
    __m128i on4 = _mm_load_si128((__m128i*) &on[i]);
    if (hor_m128i(on4))
      break; // found a block containing the last selected atom
  }

  int last4=i;
  for (i=last4+3; i>=last4; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }
  }
#elif 0 && (defined(VMDUSEVSX) && defined(__VSX__))
  // VSX vectorized search loop
  // Roll down to next 16-byte boundary
  for (i=n-1; i>=0; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }

    // drop out of the alignment loop once we hit a 16-byte boundary
    if (is_16byte_aligned(&on[i]))
      break;
  }

  for (i-=4; i>=0; i-=4) {
    // aligned load of 4 selection flags
    __vector signed int on4 = *((__vector signed int *) &on[i]);
    if (vec_extract(vec_max(on4, on4), 0))
      break; // found a block containing the last selected atom
  }

  int last4=i;
  for (i=last4+3; i>=last4; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }
  }
#else
  // plain C...
  for (i=n-1; i>=0; i--) {
    if (on[i]) {
      *lastsel = i; // found last selected atom
      return 0;
    }
  }
#endif

  // no atoms were selected if we got here
  *lastsel = -1;
  return -1;
}


// Find the first selected atom, the last selected atom,
// and the total number of selected atoms.
int analyze_selection_aligned(int n, const int *on, 
                              int *firstsel, int *lastsel, int *selected) {
  int sel   = *selected =  0;
  int first = *firstsel = 0;   // if we early-exit, firstsel is 0 
  int last  = *lastsel  = -1;  // and lastsel is -1
  int i;

  // find the first selected atom, if any
  if (find_first_selection_aligned(n, on, &first)) {
    return -1; // indicate that no selection was found
  }

  // find the last selected atom, if any
  if (find_last_selection_aligned(n, on, &last)) {
    return -1; // indicate that no selection was found
  }

  // count the number of selected atoms (there are only 0s and 1s)
  // and determine the index of the last selected atom

  // XXX the Intel 12.x compiler is able to beat this code in some
  //     cases, but GCC 4.x cannot, so for Intel C/C++ we use the plain C 
  //     loop and let it autovectorize, but for GCC we do it by hand.
#if !defined(__INTEL_COMPILER) && defined(VMDUSESSE) && defined(__SSE2__)
  // SSE vectorized search loop
  // Roll up to next 16-byte boundary
  for (i=first; ((i<=last) && (!is_16byte_aligned(&on[i]))); i++) {
    sel += on[i];
  }

  // Process groups of 4 flags at a time
  for (; i<=(last-3); i+=4) {
    // aligned load of four selection flags
    __m128i on4 = _mm_load_si128((__m128i*) &on[i]);

    // count selected atoms
    sel += hadd_m128i(on4);
  }

  // check the very end of the array (non-divisible by four)
  for (; i<=last; i++) {
    sel += on[i];
  }
#elif 1 && (defined(VMDUSEVSX) && defined(__VSX__))
  // VSX vectorized search loop
  // Roll up to next 16-byte boundary
  for (i=first; ((i<=last) && (!is_16byte_aligned(&on[i]))); i++) {
    sel += on[i];
  }

  // Process groups of 4 flags at a time
  vector signed int cnt4 = vec_splat_s32(0);
  for (; i<=(last-3); i+=4) {
    // aligned load of four selection flags
    vector signed int on4 = *((__vector signed int *) &on[i]);

    // count selected atoms
    cnt4 = vec_add(cnt4, on4);
  }
  sel += vec_extract(cnt4, 0) + vec_extract(cnt4, 1) + 
         vec_extract(cnt4, 2) + vec_extract(cnt4, 3);

  // check the very end of the array (non-divisible by four)
  for (; i<=last; i++) {
    sel += on[i];
  }
#else
  // plain C...
  for (i=first; i<=last; i++) {
    sel += on[i];
  }
#endif

  *selected = sel; 
  *firstsel = first;
  *lastsel = last;

  return 0;
}


// Compute min/max/mean values for a 16-byte-aligned array of floats
void minmaxmean_1fv_aligned(const float *f, long n, 
                            float *fmin, float *fmax, float *fmean) {
  if (n < 1) {
    *fmin = 0.0f;
    *fmax = 0.0f;
    *fmean = 0.0f;
    return;
  }

#if defined(VMDUSEAVX) && defined(__AVX__)
  long i=0;
  float min1 = f[0];
  float max1 = f[0];
  double mean1 = f[0];
  
  // roll up to the first 32-byte-aligned array index
  for (i=0; ((i<n) && !is_32byte_aligned(&f[i])); i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
    mean1 += f[i];
  }

  // AVX vectorized min/max loop
  __m256 min8 = _mm256_set1_ps(min1);
  __m256 max8 = _mm256_set1_ps(max1);
  __m256d mean4d = _mm256_set1_pd(0.0);

  // do groups of 64 elements
  for (; i<(n-63); i+=64) {
    __m256 f8 = _mm256_load_ps(&f[i]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    __m256 mean8 = f8;
    f8 = _mm256_load_ps(&f[i+8]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);
    f8 = _mm256_load_ps(&f[i+16]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);
    f8 = _mm256_load_ps(&f[i+24]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);

    f8 = _mm256_load_ps(&f[i+32]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);
    f8 = _mm256_load_ps(&f[i+40]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);
    f8 = _mm256_load_ps(&f[i+48]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);
    f8 = _mm256_load_ps(&f[i+56]); // assume 32-byte aligned array!
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean8 = _mm256_add_ps(mean8, f8);

    // sum mean8 into double-precision accumulator mean4d,
    // converting from single to double-precision, and 
    // appropriately shuffling the high and low subwords
    // so all four elements are accumulated
    mean4d = _mm256_add_pd(mean4d, _mm256_cvtps_pd(_mm256_extractf128_ps(mean8, 0)));
    mean4d = _mm256_add_pd(mean4d, _mm256_cvtps_pd(_mm256_extractf128_ps(mean8, 1)));
  }

  // sum the mean4d against itself so all elements have the total,
  // write out the lower element, and sum it into mean1
  __m256d pairs4d = _mm256_hadd_pd(mean4d, mean4d);
  mean4d = _mm256_add_pd(pairs4d, 
                         _mm256_permute2f128_pd(pairs4d, pairs4d, 0x01));
#if defined(__AVX2__)
  mean1 +=  _mm256_cvtsd_f64(mean4d); 
#else
  double tmp;
  _mm_storel_pd(&tmp, _mm256_castpd256_pd128(mean4d));
  mean1 += tmp;
#endif

  // finish last elements off
  for (; i<n; i++) {
    __m256 f8 = _mm256_set1_ps(f[i]);
    min8 = _mm256_min_ps(min8, f8);
    max8 = _mm256_max_ps(max8, f8);
    mean1 += f[i];
  }

  // compute min/max among the final 4-element vectors by shuffling
  // and and reducing the elements within the vectors
  float t0, t1;
  t0 = fmin_m128(_mm256_extractf128_ps(min8, 0));
  t1 = fmin_m128(_mm256_extractf128_ps(min8, 1));
  *fmin = (t0 < t1) ? t0 : t1;

  t0 = fmax_m128(_mm256_extractf128_ps(max8, 0));
  t1 = fmax_m128(_mm256_extractf128_ps(max8, 1));
  *fmax = (t0 > t1) ? t0 : t1;
  *fmean = mean1 / n; 
#elif defined(VMDUSESSE) && defined(__SSE2__) && (defined(__GNUC__) || defined(__INTEL_COMPILER))
  // XXX clang is broken /wrt _mm_set_pd1()
  long i=0;
  float min1 = f[0];
  float max1 = f[0];
  double mean1 = f[0];
  
  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&f[i])); i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
    mean1 += f[i];
  }

  // SSE vectorized min/max loop
  __m128 min4 = _mm_set_ps1(min1);
  __m128 max4 = _mm_set_ps1(max1);
#if (defined(ARCH_MACOSXX86) || defined(ARCH_MACOSXX86_64))
  __m128d mean2d = _mm_cvtps_pd(_mm_set_ps1(0.0f)); // XXX MacOS X workaround
#else
  __m128d mean2d = _mm_set_pd1(0.0); // XXX clang misses this
#endif

  // do groups of 32 elements
  for (; i<(n-31); i+=32) {
    __m128 f4 = _mm_load_ps(&f[i]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    __m128 mean4 = f4;
    f4 = _mm_load_ps(&f[i+4]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);
    f4 = _mm_load_ps(&f[i+8]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);
    f4 = _mm_load_ps(&f[i+12]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);

    f4 = _mm_load_ps(&f[i+16]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);
    f4 = _mm_load_ps(&f[i+20]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);
    f4 = _mm_load_ps(&f[i+24]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);
    f4 = _mm_load_ps(&f[i+28]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean4 = _mm_add_ps(mean4, f4);

    // sum mean4 into double-precision accumulator mean2d,
    // converting from single to double-precision, and 
    // appropriately shuffling the high and low subwords
    // so all four elements are accumulated
    mean2d = _mm_add_pd(mean2d, _mm_cvtps_pd(mean4));
    mean2d = _mm_add_pd(mean2d, _mm_cvtps_pd(_mm_shuffle_ps(mean4, mean4, _MM_SHUFFLE(3, 2, 3, 2))));
  }

  // do groups of 4 elements
  for (; i<(n-3); i+=4) {
    __m128 f4 = _mm_load_ps(&f[i]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);

    // sum f4 into double-precision accumulator mean2d,
    // converting from single to double-precision, and 
    // appropriately shuffling the high and low subwords
    // so all four elements are accumulated
    mean2d = _mm_add_pd(mean2d, _mm_cvtps_pd(f4));
    mean2d = _mm_add_pd(mean2d, _mm_cvtps_pd(_mm_shuffle_ps(f4, f4, _MM_SHUFFLE(3, 2, 3, 2))));
  }

  // sum the mean2d against itself so both elements have the total,
  // write out the lower element, and sum it into mean1
  mean2d = _mm_add_pd(mean2d, _mm_shuffle_pd(mean2d, mean2d, 1));
  double tmp;
  _mm_storel_pd(&tmp, mean2d);
  mean1 += tmp;

  // finish last elements off
  for (; i<n; i++) {
    __m128 f4 = _mm_set_ps1(f[i]);
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    mean1 += f[i];
  }

  // compute min/max among the final 4-element vectors by shuffling
  // and and reducing the elements within the vectors
  *fmin = fmin_m128(min4);
  *fmax = fmax_m128(max4);
  *fmean = mean1 / n; 
#else
  // scalar min/max/mean loop
  float min1 = f[0];
  float max1 = f[0];
  double mean1 = f[0];
  for (long i=1; i<n; i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
    mean1 += f[i];
  }
  *fmin = min1;
  *fmax = max1;
  *fmean = mean1 / n; 
#endif
}


// Compute min/max values for a 16-byte-aligned array of floats
void minmax_1fv_aligned(const float *f, long n, float *fmin, float *fmax) {
  if (n < 1)
    return;

#if defined(VMDUSESSE) && defined(__SSE2__)
  long i=0;
  float min1 = f[0];
  float max1 = f[0];

  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&f[i])); i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
  }

  // SSE vectorized min/max loop
  __m128 min4 = _mm_set_ps1(min1);
  __m128 max4 = _mm_set_ps1(max1);

  // do groups of 32 elements
  for (; i<(n-31); i+=32) {
    __m128 f4 = _mm_load_ps(&f[i]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+4]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+8]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+12]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);

    f4 = _mm_load_ps(&f[i+16]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+20]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+24]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
    f4 = _mm_load_ps(&f[i+28]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
  }

  // do groups of 4 elements
  for (; i<(n-3); i+=4) {
    __m128 f4 = _mm_load_ps(&f[i]); // assume 16-byte aligned array!
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
  }

  // finish last elements off
  for (; i<n; i++) {
    __m128 f4 = _mm_set_ps1(f[i]);
    min4 = _mm_min_ps(min4, f4);
    max4 = _mm_max_ps(max4, f4);
  }

  // compute min/max among the final 4-element vectors by shuffling
  // and and reducing the elements within the vectors
  *fmin = fmin_m128(min4);
  *fmax = fmax_m128(max4);
#elif defined(VMDUSEVSX) && defined(__VSX__)
  long i=0;
  float min1 = f[0];
  float max1 = f[0];

  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&f[i])); i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
  }

  // VSX vectorized min/max loop
  vector float min4 = vec_splats(min1);
  vector float max4 = vec_splats(max1);

  // do groups of 4 elements
  for (; i<(n-3); i+=4) {
    vector float f4 = *((vector float *) &f[i]); // assume 16-byte aligned array!
    min4 = vec_min(min4, f4);
    max4 = vec_max(max4, f4);
  }

  // finish last elements off
  for (; i<n; i++) {
    vector float f4 = vec_splats(f[i]);
    min4 = vec_min(min4, f4);
    max4 = vec_max(max4, f4);
  }

  // compute min/max among the final 4-element vectors by shuffling
  // and and reducing the elements within the vectors
  min1 = min4[0];
  min1 = (min1 < min4[1]) ? min1 : min4[1];
  min1 = (min1 < min4[2]) ? min1 : min4[2];
  min1 = (min1 < min4[3]) ? min1 : min4[3];

  max1 = max4[0];
  max1 = (max1 < max4[1]) ? max1 : max4[1];
  max1 = (max1 < max4[2]) ? max1 : max4[2];
  max1 = (max1 < max4[3]) ? max1 : max4[3];

  *fmin = min1;
  *fmax = max1;
#elif defined(VMDUSENEON) && defined(__ARM_NEON__)
  long i=0;
  float min1 = f[0];
  float max1 = f[0];

  // roll up to the first 16-byte-aligned array index
  for (i=0; ((i<n) && !is_16byte_aligned(&f[i])); i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
  }

  // NEON vectorized min/max loop
  float32x4_t min4 = vdupq_n_f32(min1);
  float32x4_t max4 = vdupq_n_f32(max1);

  // do groups of 32 elements
  for (; i<(n-31); i+=32) {
    float32x4_t f4;
    f4 = vld1q_f32(&f[i   ]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+ 4]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+ 8]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+12]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);

    f4 = vld1q_f32(&f[i+16]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+20]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+24]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
    f4 = vld1q_f32(&f[i+28]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
  }

  // do groups of 4 elements
  for (; i<(n-3); i+=4) {
    float32x4_t f4 = vld1q_f32(&f[i]); // assume 16-byte aligned array!
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
  }

  // finish last elements off
  for (; i<n; i++) {
    float32x4_t f4 = vdupq_n_f32(f[i]);
    min4 = vminq_f32(min4, f4);
    max4 = vmaxq_f32(max4, f4);
  }

  // compute min/max among the final 4-element vectors by shuffling
  // and and reducing the elements within the vectors
  *fmin = fmin_f32x4(min4);
  *fmax = fmax_f32x4(max4);
#else
  // scalar min/max loop
  float min1 = f[0];
  float max1 = f[0];
  for (long i=1; i<n; i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
  }
  *fmin = min1;
  *fmax = max1;
#endif
}


// Compute min/max values for a 16-byte-aligned array of float3s
// input value n3 is the number of 3-element vectors to process
void minmax_3fv_aligned(const float *f, const long n3, float *fmin, float *fmax) {
  float minx, maxx, miny, maxy, minz, maxz;
  const long end = n3*3L;

  if (n3 < 1)
    return;

  long i=0;
  minx=maxx=f[i  ];
  miny=maxy=f[i+1];
  minz=maxz=f[i+2];

#if defined(VMDUSESSE) && defined(__SSE2__)
  // Since we may not be on a 16-byte boundary when we start, we roll 
  // through the first few items with plain C until we get to one.
  for (; i<end; i+=3L) {
    // exit if/when we reach a 16-byte boundary for both arrays
    if (is_16byte_aligned(&f[i])) {
      break;
    }

    float tmpx = f[i  ];
    if (tmpx < minx) minx = tmpx;
    if (tmpx > maxx) maxx = tmpx;

    float tmpy = f[i+1];
    if (tmpy < miny) miny = tmpy;
    if (tmpy > maxy) maxy = tmpy;

    float tmpz = f[i+2];
    if (tmpz < minz) minz = tmpz;
    if (tmpz > maxz) maxz = tmpz;
  }

  // initialize min/max values
  __m128 xmin4 = _mm_set_ps1(minx);
  __m128 xmax4 = _mm_set_ps1(maxx);
  __m128 ymin4 = _mm_set_ps1(miny);
  __m128 ymax4 = _mm_set_ps1(maxy);
  __m128 zmin4 = _mm_set_ps1(minz);
  __m128 zmax4 = _mm_set_ps1(maxz);

  for (; i<(end-11); i+=12) {
    // aligned load of four consecutive 3-element vectors into
    // three 4-element vectors
    __m128 x0y0z0x1 = _mm_load_ps(&f[i  ]);
    __m128 y1z1x2y2 = _mm_load_ps(&f[i+4]);
    __m128 z2x3y3z3 = _mm_load_ps(&f[i+8]);

    // convert rgb3f AOS format to 4-element SOA vectors using shuffle instructions
    __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
    __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
    __m128 x        = _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)); // x0x1x2x3
    __m128 y        = _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0)); // y0y1y2y3
    __m128 z        = _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3, 1)); // z0y1z2z3

    // compute mins and maxes
    xmin4 = _mm_min_ps(xmin4, x);
    xmax4 = _mm_max_ps(xmax4, x);
    ymin4 = _mm_min_ps(ymin4, y);
    ymax4 = _mm_max_ps(ymax4, y);
    zmin4 = _mm_min_ps(zmin4, z);
    zmax4 = _mm_max_ps(zmax4, z);
  }

  minx = fmin_m128(xmin4);
  miny = fmin_m128(ymin4);
  minz = fmin_m128(zmin4);

  maxx = fmax_m128(xmax4);
  maxy = fmax_m128(ymax4);
  maxz = fmax_m128(zmax4);
#endif

  // regular C code... 
  for (; i<end; i+=3) {
    float tmpx = f[i  ];
    if (tmpx < minx) minx = tmpx;
    if (tmpx > maxx) maxx = tmpx;

    float tmpy = f[i+1];
    if (tmpy < miny) miny = tmpy;
    if (tmpy > maxy) maxy = tmpy;

    float tmpz = f[i+2];
    if (tmpz < minz) minz = tmpz;
    if (tmpz > maxz) maxz = tmpz;
  }

  fmin[0] = minx;
  fmax[0] = maxx;
  fmin[1] = miny;
  fmax[1] = maxy;
  fmin[2] = minz;
  fmax[2] = maxz;
}


// Compute min/max values for a 16-byte-aligned array of float3s
// input value n3 is the number of 3-element vectors to process
int minmax_selected_3fv_aligned(const float *f, const int *on, const long n3, 
                                const long firstsel, const long lastsel,
                                float *fmin, float *fmax) {
  float minx, maxx, miny, maxy, minz, maxz;

  if ((n3 < 1) || (firstsel < 0) || (lastsel < firstsel) || (lastsel >= n3))
    return -1;

  // start at first selected atom
  long i=firstsel;
  minx=maxx=f[i*3L  ];
  miny=maxy=f[i*3L+1];
  minz=maxz=f[i*3L+2];

  long end=lastsel+1;

// printf("Starting array alignment: on[%d]: %p f[%d]: %p\n",
//        i, &on[i], i*3L, &f[i*3L]);

#if defined(VMDUSESSE) && defined(__SSE2__)
  // since we may not be on a 16-byte boundary, when we start, we roll 
  // through the first few items with plain C until we get to one.
  for (; i<end; i++) {
    long ind3 = i * 3L;

#if 1
    // exit if/when we reach a 16-byte boundary for the coordinate array only,
    // for now we'll do unaligned loads of the on array since there are cases
    // where we get differently unaligned input arrays and they'll never 
    // line up at a 16-byte boundary at the same time
    if (is_16byte_aligned(&f[ind3])) {
      break;
    }
#else
    // exit if/when we reach a 16-byte boundary for both arrays
    if (is_16byte_aligned(&on[i]) && is_16byte_aligned(&f[ind3])) {
// printf("Found alignment boundary: on[%d]: %p f[%d]: %p\n",
//        i, &on[i], ind3, &f[ind3]);
      break;
    }
#endif

    if (on[i]) {
      float tmpx = f[ind3  ];
      if (tmpx < minx) minx = tmpx;
      if (tmpx > maxx) maxx = tmpx;

      float tmpy = f[ind3+1];
      if (tmpy < miny) miny = tmpy;
      if (tmpy > maxy) maxy = tmpy;

      float tmpz = f[ind3+2];
      if (tmpz < minz) minz = tmpz;
      if (tmpz > maxz) maxz = tmpz;
    }
  }

  // initialize min/max values to results from scalar loop above
  __m128 xmin4 = _mm_set_ps1(minx);
  __m128 xmax4 = _mm_set_ps1(maxx);
  __m128 ymin4 = _mm_set_ps1(miny);
  __m128 ymax4 = _mm_set_ps1(maxy);
  __m128 zmin4 = _mm_set_ps1(minz);
  __m128 zmax4 = _mm_set_ps1(maxz);

  for (; i<(end-3); i+=4) {
#if 1
    // XXX unaligned load of four selection flags, since there are cases
    //     where the input arrays can't achieve alignment simultaneously
    __m128i on4 = _mm_loadu_si128((__m128i*) &on[i]);
#else
    // aligned load of four selection flags
    __m128i on4 = _mm_load_si128((__m128i*) &on[i]);
#endif

    // compute atom selection mask
    __m128i mask = _mm_cmpeq_epi32(_mm_set1_epi32(1), on4);
    if (!hor_m128i(mask))
      continue; // no atoms selected

    // aligned load of four consecutive 3-element vectors into
    // three 4-element vectors
    long ind3 = i * 3L;
    __m128 x0y0z0x1 = _mm_load_ps(&f[ind3+0]);
    __m128 y1z1x2y2 = _mm_load_ps(&f[ind3+4]);
    __m128 z2x3y3z3 = _mm_load_ps(&f[ind3+8]);

    // convert rgb3f AOS format to 4-element SOA vectors using shuffle instructions
    __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
    __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
    __m128 x        = _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0)); // x0x1x2x3
    __m128 y        = _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0)); // y0y1y2y3
    __m128 z        = _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3, 1)); // z0y1z2z3

    // compute mins and maxes
    xmin4 = _mm_sel_ps(xmin4, _mm_min_ps(xmin4, x), (__m128) mask);
    xmax4 = _mm_sel_ps(xmax4, _mm_max_ps(xmax4, x), (__m128) mask);
    ymin4 = _mm_sel_ps(ymin4, _mm_min_ps(ymin4, y), (__m128) mask);
    ymax4 = _mm_sel_ps(ymax4, _mm_max_ps(ymax4, y), (__m128) mask);
    zmin4 = _mm_sel_ps(zmin4, _mm_min_ps(zmin4, z), (__m128) mask);
    zmax4 = _mm_sel_ps(zmax4, _mm_max_ps(zmax4, z), (__m128) mask);
  }

  minx = fmin_m128(xmin4);
  miny = fmin_m128(ymin4);
  minz = fmin_m128(zmin4);

  maxx = fmax_m128(xmax4);
  maxy = fmax_m128(ymax4);
  maxz = fmax_m128(zmax4);
#endif

  // regular C code... 
  for (; i<end; i++) {
    if (on[i]) {
      long ind3 = i * 3L;
      float tmpx = f[ind3  ];
      if (tmpx < minx) minx = tmpx;
      if (tmpx > maxx) maxx = tmpx;

      float tmpy = f[ind3+1];
      if (tmpy < miny) miny = tmpy;
      if (tmpy > maxy) maxy = tmpy;

      float tmpz = f[ind3+2];
      if (tmpz < minz) minz = tmpz;
      if (tmpz > maxz) maxz = tmpz;
    }
  }

  fmin[0] = minx;
  fmax[0] = maxx;
  fmin[1] = miny;
  fmax[1] = maxy;
  fmin[2] = minz;
  fmax[2] = maxz;

  return 0;
}


