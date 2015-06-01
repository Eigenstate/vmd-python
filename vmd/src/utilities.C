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
 *	$RCSfile: utilities.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.169 $	$Date: 2015/05/13 19:21:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * General utility routines and definitions.
 *
 ***************************************************************************/

#define VMDUSESSE 1
// #define VMDUSEAVX 1
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
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#if defined(__irix)
#include <bstring.h>
#endif

#if defined(__hpux)
#include <time.h>
#endif // HPUX
#endif // _MSC_VER

#if defined(AIXUSEPERFSTAT)
#include <libperfstat.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "utilities.h"

// given an argc, argv pair, take all the arguments from the Nth one on
// and combine them into a single string with spaces separating words.  This
// allocates space for the string, which must be freed by the user.
char *combine_arguments(int argc, const char **argv, int n) {
  char *newstr = NULL;

  if(argc > 0 && n < argc && n >= 0) {
    int i, sl = 0;
    // find out the length of the words we must combine
    for(i=n; i < argc; i++)
      sl += strlen(argv[i]);

    // combine the words together
    if(sl) {
      newstr = new char[sl + 8 + argc - n];	// extra buffer added
      *newstr = '\0';
      for(i=n; i < argc; i++) {
        if(i != n)
          strcat(newstr," ");
        strcat(newstr, argv[i]);
      }
    }
  }

  // return the string, or NULL if a problem occurred
  return newstr;
}


// duplicate a string using c++ new call
char *stringdup(const char *s) {
  char *rs;

  if(!s)
    return NULL;

  rs = new char[strlen(s) + 1];
  strcpy(rs,s);

  return rs;
}


// convert a string to upper case
char *stringtoupper(char *s) {
  if (s != NULL) {
    int i;
    int sz = strlen(s);
    for(i=0; i<sz; i++)
      s[i] = toupper(s[i]);
  }

  return s;
}

void stripslashes(char *str) {
  while (strlen(str) > 0 && str[strlen(str) - 1] == '/') {
    str[strlen(str) - 1] = '\0';
  }
}

// do upper-case comparison
int strupcmp(const char *a, const char *b) {
  char *ua, *ub;
  int retval;

  ua = stringtoupper(stringdup(a));
  ub = stringtoupper(stringdup(b));

  retval = strcmp(ua,ub);

  delete [] ub;
  delete [] ua;

  return retval;
}


// do upper-case comparison, up to n characters
int strupncmp(const char *a, const char *b, int n) {
#if defined(ARCH_AIX3) || defined(ARCH_AIX4) || defined(_MSC_VER)
   while (n-- > 0) {
      if (toupper(*a) != toupper(*b)) {
	 return toupper(*b) - toupper(*a);
      }
      if (*a == 0) return 0;
      a++; b++;
   }
   return 0;
#else
   return strncasecmp(a, b, n);
#endif
}


// break a file name up into path + name, returning both in the specified
//	character pointers.  This creates storage for the new strings
//	by allocating space for them.
void breakup_filename(const char *full, char **path, char **name) {
  const char *namestrt;
  int pathlen;

  if(full == NULL) {
    *path = *name = NULL;
    return;
  } else if (strlen(full) == 0) {
    *path = new char[1];
    *name = new char[1];
    (*path)[0] = (*name)[0] = '\0';
    return;
  }

  // find start of final file name
  if((namestrt = strrchr(full,'/')) != NULL && strlen(namestrt) > 0) {
    namestrt++;
  } else {
    namestrt = full;
  }

  // make a copy of the name
  *name = stringdup(namestrt);

  // make a copy of the path
  pathlen = strlen(full) - strlen(*name);
  *path = new char[pathlen + 1];
  strncpy(*path,full,pathlen);
  (*path)[pathlen] = '\0';
} 

// break a configuration line up into tokens.
char *str_tokenize(const char *newcmd, int *argc, char *argv[]) {
  char *cmd; 
  const char *cmdstart;
  cmdstart = newcmd;

  // guarantee that the command string we return begins on the first
  // character returned by strtok(), otherwise the subsequent delete[]
  // calls will reference invalid memory blocks
  while (cmdstart != NULL &&
         (*cmdstart == ' '  ||
          *cmdstart == ','  ||
          *cmdstart == ';'  ||
          *cmdstart == '\t' ||
          *cmdstart == '\n')) {
    cmdstart++; // advance pointer to first command character
  } 

  cmd = stringdup(cmdstart);
  *argc = 0;

  // initialize tokenizing calls
  argv[*argc] = strtok(cmd, " ,;\t\n");

  // loop through words until end-of-string, or comment character, found
  while(argv[*argc] != NULL) {
    // see if the token starts with '#'
    if(argv[*argc][0] == '#') {
      break;                    // don't process any further tokens
    } else {
      (*argc)++;		// another token in list
    }
    
    // scan for next token
    argv[*argc] = strtok(NULL," ,;\t\n");
  }

  return (*argc > 0 ? argv[0] : (char *) NULL);
}


// get the time of day from the system clock, and store it (in seconds)
double time_of_day(void) {
#if defined(_MSC_VER)
  double t;
 
  t = GetTickCount(); 
  t = t / 1000.0;

  return t;
#else
  struct timeval tm;
  struct timezone tz;

  gettimeofday(&tm, &tz);
  return((double)(tm.tv_sec) + (double)(tm.tv_usec)/1000000.0);
#endif
}


int vmd_check_stdin(void) {
#if defined(_MSC_VER)
  if (_kbhit() != 0)
    return TRUE;
  else
    return FALSE;
#else
  fd_set readvec;
  struct timeval timeout;
  int ret, stdin_fd;

  timeout.tv_sec = 0;
  timeout.tv_usec = 0;
  stdin_fd = 0;
  FD_ZERO(&readvec);
  FD_SET(stdin_fd, &readvec);

#if !defined(ARCH_AIX3)
  ret = select(16, &readvec, NULL, NULL, &timeout);
#else
  ret = select(16, (int *)(&readvec), NULL, NULL, &timeout);
#endif
 
  if (ret == -1) {  // got an error
    if (errno != EINTR)  // XXX: this is probably too lowlevel to be converted to Inform.h
      printf("select() error while attempting to read text input.\n");
    return FALSE;
  } else if (ret == 0) {
    return FALSE;  // select timed out
  }
  return TRUE;
#endif
}


// return the username of the currently logged-on user
char *vmd_username(void) {
#if defined(_MSC_VER)
  char username[1024];
  unsigned long size = 1023;

  if (GetUserName((char *) &username, &size)) {
    return stringdup(username);
  }
  else { 
    return stringdup("Windows User");
  }
#else
#if defined(ARCH_FREEBSD) || defined(ARCH_FREEBSDAMD64) || defined(__APPLE__) || defined(__linux)
  return stringdup(getlogin());
#else
  return stringdup(cuserid(NULL));
#endif 
#endif
}

int vmd_getuid(void) {
#if defined(_MSC_VER)
  return 0;
#else
  return getuid(); 
#endif
}


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


#if defined(VMDUSESSE) || defined(VMDUSEAVX) || defined(VMDUSENEON)

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

// Aligment test routine for x86 16-byte SSE vector instructions
static int is_16byte_aligned(const void *ptr) {
  return (((myintptrtype) ptr) == (((myintptrtype) ptr) & (~0xf)));
}

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


static __m128i _mm_sel_m128i(const __m128i &a, const __m128i &b, const __m128i &mask) {
  // (((b ^ a) & mask)^a)
  return _mm_xor_si128(a, _mm_and_si128(mask, _mm_xor_si128(b, a)));
}


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


// Compute min/max values for a 16-byte-aligned array of floats
void minmax_1fv_aligned(const float *f, int n, float *fmin, float *fmax) {
  if (n < 1)
    return;

#if defined(VMDUSESSE) && defined(__SSE2__)
  int i=0;
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
#elif defined(VMDUSENEON) && defined(__ARM_NEON__)
  int i=0;
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
  for (int i=1; i<n; i++) {
    if (f[i] < min1) min1 = f[i];
    if (f[i] > max1) max1 = f[i];
  }
  *fmin = min1;
  *fmax = max1;
#endif
}


// Compute min/max values for a 16-byte-aligned array of float3s
// input value n3 is the number of 3-element vectors to process
void minmax_3fv_aligned(const float *f, const int n3, float *fmin, float *fmax) {
  float minx, maxx, miny, maxy, minz, maxz;
  const int end = n3*3;

  if (n3 < 1)
    return;

  int i=0;
  minx=maxx=f[i  ];
  miny=maxy=f[i+1];
  minz=maxz=f[i+2];

#if defined(VMDUSESSE) && defined(__SSE2__)
  // Since we may not be on a 16-byte boundary when we start, we roll 
  // through the first few items with plain C until we get to one.
  for (; i<end; i+=3) {
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
int minmax_selected_3fv_aligned(const float *f, const int *on, const int n3, 
                                const int firstsel, const int lastsel,
                                float *fmin, float *fmax) {
  float minx, maxx, miny, maxy, minz, maxz;

  if ((n3 < 1) || (firstsel < 0) || (lastsel < firstsel) || (lastsel >= n3))
    return -1;

  // start at first selected atom
  int i=firstsel;
  minx=maxx=f[i*3  ];
  miny=maxy=f[i*3+1];
  minz=maxz=f[i*3+2];

  int end=lastsel+1;

// printf("Starting array alignment: on[%d]: %p f[%d]: %p\n",
//        i, &on[i], i*3, &f[i*3]);

#if defined(VMDUSESSE) && defined(__SSE2__)
  // since we may not be on a 16-byte boundary, when we start, we roll 
  // through the first few items with plain C until we get to one.
  for (; i<end; i++) {
    int ind3 = i * 3;

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
    int ind3 = i * 3;
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
      int ind3 = i * 3;
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


// take three 3-vectors and compute x2 cross x3; with the results
// in x1.  x1 must point to different memory than x2 or x3
// This returns a pointer to x1
float * cross_prod(float *x1, const float *x2, const float *x3)
{
  x1[0] =  x2[1]*x3[2] - x3[1]*x2[2];
  x1[1] = -x2[0]*x3[2] + x3[0]*x2[2];
  x1[2] =  x2[0]*x3[1] - x3[0]*x2[1];
  return x1;
}

// normalize a vector, and return a pointer to it
// Warning:  it changes the value of the vector!!
float * vec_normalize(float *vect) {
  float len2 = vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2];

  // prevent division by zero
  if (len2 > 0) {
    float rescale = 1.0f / sqrtf(len2);
    vect[0] *= rescale;
    vect[1] *= rescale;
    vect[2] *= rescale;
  }

  return vect;
}


// find and return the norm of a 3-vector
float norm(const float *vect) {
  return sqrtf(vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2]);
}


// determine if a triangle is degenerate or not
int tri_degenerate(const float * v0, const float * v1, const float * v2) {
  float s1[3], s2[3], s1_length, s2_length;

  /*
   various rendering packages have amusingly different ideas about what
   constitutes a degenerate triangle.  -1 and 1 work well.  numbers
   below 0.999 and -0.999 show up in OpenGL
   numbers as low as 0.98 have worked in POVRay with certain models while
   numbers as high as 0.999999 have produced massive holes in other
   models
         -matt 11/13/96
  */

  /**************************************************************/
  /*    turn the triangle into 2 normalized vectors.            */
  /*    If the dot product is 1 or -1 then                      */
  /*   the triangle is degenerate                               */
  /**************************************************************/
  s1[0] = v0[0] - v1[0];
  s1[1] = v0[1] - v1[1];
  s1[2] = v0[2] - v1[2];

  s2[0] = v0[0] - v2[0];
  s2[1] = v0[1] - v2[1];
  s2[2] = v0[2] - v2[2];

  s1_length = sqrtf(s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2]);
  s2_length = sqrtf(s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2]);

  /**************************************************************/
  /*                   invert to avoid divides:                 */
  /*                         1.0/v1_length * 1.0/v2_length      */
  /**************************************************************/

  s2_length = 1.0f / (s1_length*s2_length);
  s1_length = s2_length * (s1[0]*s2[0] + s1[1]*s2[1] + s1[2]*s2[2]);

  // and add it to the list if it's not degenerate
  if ((s1_length >= 1.0 ) || (s1_length <= -1.0)) 
    return 1;
  else
    return 0;
}


// compute the angle (in degrees 0 to 180 ) between two vectors a & b
float angle(const float *a, const float *b) {
  float ab[3];
  cross_prod(ab, a, b);
  float psin = sqrtf(dot_prod(ab, ab));
  float pcos = dot_prod(a, b);
  return 57.2958f * (float) atan2(psin, pcos);
}


// Compute the dihedral angle for the given atoms, returning a value between
// -180 and 180.
// faster, cleaner implementation based on atan2
float dihedral(const float *a1,const float *a2,const float *a3,const float *a4)
{
  float r1[3], r2[3], r3[3], n1[3], n2[3];
  vec_sub(r1, a2, a1);
  vec_sub(r2, a3, a2);
  vec_sub(r3, a4, a3);
  
  cross_prod(n1, r1, r2);
  cross_prod(n2, r2, r3);
  
  float psin = dot_prod(n1, r3) * sqrtf(dot_prod(r2, r2));
  float pcos = dot_prod(n1, n2);

  // atan2f would be faster, but we'll have to workaround the lack
  // of existence on some platforms.
  return 57.2958f * (float) atan2(psin, pcos);
}
 
// compute the distance between points a & b
float distance(const float *a, const float *b) {
  return sqrtf(distance2(a,b));
}

char *vmd_tempfile(const char *s) {
  char *envtxt, *TempDir;

  if((envtxt = getenv("VMDTMPDIR")) != NULL) {
    TempDir = stringdup(envtxt);
  } else {
#if defined(_MSC_VER)
    if ((envtxt = getenv("TMP")) != NULL) {
      TempDir = stringdup(envtxt);
    }
    else if ((envtxt = getenv("TEMP")) != NULL) {
      TempDir = stringdup(envtxt);
    }
    else {
      TempDir = stringdup("c:\\\\");
    }
#else
    TempDir = stringdup("/tmp");
#endif
  }
  stripslashes(TempDir); // strip out ending '/' chars.

  char *tmpfilebuf = new char[1024];
 
  // copy in temp string
  strcpy(tmpfilebuf, TempDir);
 
#if defined(_MSC_VER)
  strcat(tmpfilebuf, "\\");
  strncat(tmpfilebuf, s, 1022 - strlen(TempDir));
#else
  strcat(tmpfilebuf, "/");
  strncat(tmpfilebuf, s, 1022 - strlen(TempDir));
#endif
 
  tmpfilebuf[1023] = '\0';
 
  delete [] TempDir;

  // return converted string
  return tmpfilebuf;
}


int vmd_delete_file(const char * path) {
#if defined(_MSC_VER)
  if (DeleteFile(path) == 0) 
    return -1;
  else 
    return 0;  
#else
  return unlink(path);
#endif
}

void vmd_sleep(int secs) {
#if defined(_MSC_VER)
  Sleep(secs * 1000);
#else 
  sleep(secs);
#endif
}

void vmd_msleep(int msecs) {
#if defined(_MSC_VER)
  Sleep(msecs);
#else 
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = 1000 * msecs;
  select(0, NULL, NULL, NULL, &timeout);
#endif // _MSC_VER
}

int vmd_system(const char* cmd) {
   return system(cmd);
}


/// portable random number generation, NOT thread-safe however
/// XXX we should replace these with our own thread-safe random number 
/// generator implementation at some point.
long vmd_random(void) {
#ifdef _MSC_VER
  return rand();
#else
  return random();
#endif
}

void vmd_srandom(unsigned int seed) {
#ifdef _MSC_VER
  srand(seed);
#else
  srandom(seed);
#endif
}

/// Slow but accurate standard distribution random number generator
/// (variance = 1)
float vmd_random_gaussian() {
  static bool cache = false;
  static float cached_value;
  const float RAND_FACTOR = 2.f/VMD_RAND_MAX;
  float r, s, w;
  
  if (cache) {
    cache = false;
    return cached_value;
  }
  do {
    r = RAND_FACTOR*vmd_random()-1.f; 
    s = RAND_FACTOR*vmd_random()-1.f;
    w = r*r+s*s;
  } while (w >= 1.f);
  w = sqrtf(-2.f*logf(w)/w);
  cached_value = s * w;
  cache = true;
  return (r*w);
}


/// routine to query the OS and find out how many MB of physical memory 
/// is installed in the system
long vmd_get_total_physmem_mb(void) {
#if defined(_MSC_VER)
  MEMORYSTATUS memstat;
  GlobalMemoryStatus(&memstat);
  if (memstat.dwLength != sizeof(memstat))
    return -1; /* memstat result is wrong size! */
  return memstat.dwTotalPhys/(1024 * 1024);
#elif defined(__linux)
  FILE *fp;
  char meminfobuf[1024], *pos;
  size_t len;

  fp = fopen("/proc/meminfo", "r");
  if (fp != NULL) {
    len = fread(meminfobuf,1,1024, fp);
    meminfobuf[1023] = 0;
    fclose(fp);
    if (len > 0) {
      pos=strstr(meminfobuf,"MemTotal:");
      if (pos == NULL) 
        return -1;
      pos += 9; /* skip tag */;
      return strtol(pos, (char **)NULL, 10)/1024L;
    }
  } 
  return -1;
#elif defined(AIXUSEPERFSTAT) && defined(_AIX)
  perfstat_memory_total_t minfo;
  perfstat_memory_total(NULL, &minfo, sizeof(perfstat_memory_total_t), 1);
  return minfo.real_total*(4096/1024)/1024;
#elif defined(_AIX)
  return (sysconf(_SC_AIX_REALMEM) / 1024);
#elif defined(_SC_PAGESIZE) && defined(_SC_PHYS_PAGES)
  /* SysV Unix */
  long pgsz = sysconf(_SC_PAGESIZE);
  long physpgs = sysconf(_SC_PHYS_PAGES);
  return ((pgsz / 1024) * physpgs) / 1024;
#elif defined(__APPLE__)
  /* MacOS X uses BSD sysctl */
  /* use hw.memsize, as it's a 64-bit value */
  int rc;
  uint64_t membytes;
  size_t len = sizeof(membytes);
  if (sysctlbyname("hw.memsize", &membytes, &len, NULL, 0)) 
    return -1;
  return (membytes / (1024*1024));
#else
  return -1; /* unrecognized system, no method to get this info */
#endif
}



/// routine to query the OS and find out how many MB of physical memory 
/// is actually "free" for use by processes (don't include VM/swap..)
long vmd_get_avail_physmem_mb(void) {
#if defined(_MSC_VER)
  MEMORYSTATUS memstat;
  GlobalMemoryStatus(&memstat);
  if (memstat.dwLength != sizeof(memstat))
    return -1; /* memstat result is wrong size! */ 
  return memstat.dwAvailPhys / (1024 * 1024);
#elif defined(__linux)
  FILE *fp;
  char meminfobuf[1024], *pos;
  size_t len;
  long val;

  fp = fopen("/proc/meminfo", "r");
  if (fp != NULL) {
    len = fread(meminfobuf,1,1024, fp);
    meminfobuf[1023] = 0;
    fclose(fp);
    if (len > 0) {
      val = 0L;
      pos=strstr(meminfobuf,"MemFree:");
      if (pos != NULL) {
        pos += 8; /* skip tag */;
        val += strtol(pos, (char **)NULL, 10);
      }
      pos=strstr(meminfobuf,"Buffers:");
      if (pos != NULL) {
        pos += 8; /* skip tag */;
        val += strtol(pos, (char **)NULL, 10);
      }
      pos=strstr(meminfobuf,"Cached:");
      if (pos != NULL) {
        pos += 8; /* skip tag */;
        val += strtol(pos, (char **)NULL, 10);
      }
      return val/1024L;
    } else {
      return -1;
    }
  } else {
    return -1;
  }
#elif defined(AIXUSEPERFSTAT) && defined(_AIX)
  perfstat_memory_total_t minfo;
  perfstat_memory_total(NULL, &minfo, sizeof(perfstat_memory_total_t), 1);
  return minfo.real_free*(4096/1024)/1024;
#elif defined(_SC_PAGESIZE) && defined(_SC_AVPHYS_PAGES)
  /* SysV Unix */
  long pgsz = sysconf(_SC_PAGESIZE);
  long avphyspgs = sysconf(_SC_AVPHYS_PAGES);
  return ((pgsz / 1024) * avphyspgs) / 1024;
#elif defined(__APPLE__)
#if 0
  /* BSD sysctl */
  /* hw.usermem isn't really the amount of free memory, it's */
  /* really more a measure of the non-kernel memory          */
  int rc;
  int membytes;
  size_t len = sizeof(membytes);
  if (sysctlbyname("hw.usermem", &membytes, &len, NULL, 0)) 
    return -1;
  return (membytes / (1024*1024));
#else
  return -1;
#endif
#else
  return -1; /* unrecognized system, no method to get this info */
#endif
}


/// return integer percentage of physical memory available
long vmd_get_avail_physmem_percent(void) {
  double total, avail;
  total = (double) vmd_get_total_physmem_mb();
  avail = (double) vmd_get_avail_physmem_mb();
  if (total > 0.0 && avail >= 0.0)
    return (long) (avail / (total / 100.0));

  return -1; /* return an error */
}


