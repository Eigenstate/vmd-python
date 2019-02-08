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
 *      $RCSfile: Benchmark.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.6 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Various CPU/memory subsystem benchmarking routines.
 * The peak performance numbers achieved within a VMD build can be 
 * used to determine how well the VMD build was optimized, the 
 * performance of the host CPU/memory systems, SMP scaling efficiency, etc.
 *
 * The streaming memory bandwidth tests are an alternative implementation 
 * of McCalpin's STREAM benchmark.
 *
 ***************************************************************************/


/// built-in minor variation of the STREAM benchmark
int stream_bench(int N, double *time, double *mbsec);



