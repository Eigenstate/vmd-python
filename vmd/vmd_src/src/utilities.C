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
 *	$RCSfile: utilities.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.173 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * General utility routines and definitions.
 *
 ***************************************************************************/

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
  if ((s1_length >= 1.0f ) || (s1_length <= -1.0f)) 
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


