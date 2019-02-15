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
 *      $RCSfile: OpenCLUtils.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2019/01/17 21:38:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   OpenCL utility functions for use in VMD
 *
 ***************************************************************************/

#if defined(__APPLE__)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int vmd_cl_print_platform_info(void);

cl_platform_id vmd_cl_get_platform_index(int i);

int vmd_cl_context_num_devices(cl_context clctx);

cl_command_queue vmd_cl_create_command_queue(cl_context clctx, int dev);

cl_kernel vmd_cl_compile_kernel(cl_context clctx, const char *kernname,
                                 const char *srctext, const char *flags, 
                                 cl_int *clerr, int verbose);


