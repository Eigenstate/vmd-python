/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: msmpot.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2010/06/03 20:07:08 $
 *
 ***************************************************************************/

#include "msmpot_internal.h"

#undef  NELEMS
#define NELEMS(a)  ((int)(sizeof(a)/sizeof(a[0])))


/* these strings correspond to the return codes in msmpot.h,
 * where the "unknown error" is used for out-of-range retcode */
static const char *ERROR_STRING[] = {
  "success",
  "assertion failed",
  "memory allocation error",
  "illegal parameter",
  "unsupported request",
  "CUDA device request failed",
  "CUDA memory allocation error",
  "CUDA memory copy error",
  "CUDA kernel execution failed",
  "CUDA kernel does not support request",
  "unknown error",
};

/* assume that the "unknown" error is listed last */
const char *Msmpot_error_string(int retcode) {
  if (retcode < 0 || retcode >= NELEMS(ERROR_STRING)) {
    retcode = NELEMS(ERROR_STRING) - 1;
  }
  return ERROR_STRING[retcode];
}


#ifdef MSMPOT_DEBUG

/* When debugging report error to stderr stream, return "err".
 * Note that we can't use Msmpot_error_message() since 
 * we might not have the Msmpot * available to us. */
int Msmpot_report_error(int err, const char *msg, const char *fn, int ln) {
  if (msg) {
    fprintf(stderr, "MSMPOT ERROR (%s,%d): %s: %s\n",
        fn, ln, Msmpot_error_string(err), msg);
  }
  else {
    fprintf(stderr, "MSMPOT ERROR (%s,%d): %s\n",
        fn, ln, Msmpot_error_string(err));
  }
  return err;
}

#endif


Msmpot *Msmpot_create(void) {
  Msmpot *msm = (Msmpot *) calloc(1, sizeof(Msmpot));
  if (NULL == msm) return NULL;
#ifdef MSMPOT_CUDA
  msm->msmcuda = Msmpot_cuda_create();
  if (NULL == msm->msmcuda) {
    Msmpot_destroy(msm);
    return NULL;
  }
#endif
  Msmpot_set_defaults(msm);
  return msm;
}


void Msmpot_destroy(Msmpot *msm) {
#ifdef MSMPOT_CUDA
  if (msm->msmcuda) Msmpot_cuda_destroy(msm->msmcuda);
#endif
  Msmpot_cleanup(msm);
  free(msm);
}


int Msmpot_use_cuda(Msmpot *msm, const int *devlist, int listlen,
    int cuda_optional) {
  if (NULL == devlist || listlen <= 0) {
    return ERROR(MSMPOT_ERROR_PARAM);
  }
#ifdef MSMPOT_CUDA
  msm->devlist = devlist;
  msm->devlistlen = listlen;
  msm->cuda_optional = cuda_optional;
  msm->use_cuda = 1;  /* attempt to use CUDA */
  return MSMPOT_SUCCESS;
#else
  return ERRMSG(MSMPOT_ERROR_SUPPORT,
      "CUDA support is not available in this build");
#endif
}
