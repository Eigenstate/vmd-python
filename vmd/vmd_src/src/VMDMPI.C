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
 *      $RCSfile: VMDMPI.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.23 $      $Date: 2019/02/07 21:41:51 $
 *
 ***************************************************************************/

#ifdef VMDMPI
#include <mpi.h>

// Check to see if we have to pass the MPI_IN_PLACE flag
// for in-place allgather reductions (same approach as Tachyon)
#if !defined(USE_MPI_IN_PLACE)
#if (MPI_VERSION >= 2) || defined(MPI_IN_PLACE)
#define USE_MPI_IN_PLACE 1
#endif
#endif

#include <stdio.h>
#include <string.h>
#include <Inform.h>
#include <utilities.h>
#include <WKFThreads.h>
#include "VMDMPI.h"

typedef struct {
  int numcpus;               /* number of processors on this node       */
  int numgpus;               /* number of GPUs on this node             */
  float cpuspeed;            /* relative speed of cpus on this node     */
  float nodespeed;           /* relative speed index for this node      */
  char machname[512];        /* machine/node name                       */
  long corefree;             /* available physical memory               */
  long corepcnt;             /* available physical memory (percentage)  */
} nodeinfo;

int vmd_mpi_init(int *argc, char ***argv) {
  int numnodes=0, mynode=0;

#if defined(VMDTHREADS)
  int provided=0;
  MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
  if (provided != MPI_THREAD_SERIALIZED) {
    msgWarn << "MPI not providing thread-serial access." << sendmsg;
  }
#else
  MPI_Init(argc, argv);
#endif
  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  MPI_Comm_size(MPI_COMM_WORLD, &numnodes);

  // mute console output for all VMD instances other than the root node
  if (mynode != 0) {
    msgInfo.mute();
    msgWarn.mute();
    msgErr.mute();
  }

  return 0;
}

int vmd_mpi_barrier() {
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

int vmd_mpi_fini() {
  vmd_mpi_barrier(); 
  msgInfo << "All nodes have reached the MPI shutdown barrier." << sendmsg;

  MPI_Finalize();

  return 0;
}

int vmd_mpi_nodescan(int *noderank, int *nodecount, 
                     char *nodename, int maxnodenamelen, 
                     int localgpucount) {
  int numnodes=0, mynode=0;
  int namelen;
  char namebuf[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
  msgInfo << "Initializing parallel VMD instances via MPI..." << sendmsg;

  nodeinfo *nodes;
  nodes = (nodeinfo *) malloc(numnodes * sizeof(nodeinfo));
  nodes[mynode].numcpus = wkf_thread_numprocessors();
  nodes[mynode].numgpus = localgpucount;
  nodes[mynode].cpuspeed = 1.0;
  nodes[mynode].nodespeed = nodes[mynode].numcpus * nodes[mynode].cpuspeed;
  nodes[mynode].corefree = vmd_get_avail_physmem_mb();
  nodes[mynode].corepcnt = vmd_get_avail_physmem_percent();

  MPI_Get_processor_name((char *) &namebuf, &namelen);

  // prepare for all-to-all gather
  strncpy((char *) &nodes[mynode].machname, namebuf,
          (((namelen + 1) < 511) ? (namelen+1) : 511));

  // provide to caller
  strncpy(nodename, namebuf,
    (((namelen + 1) < (maxnodenamelen-1)) ? (namelen+1) : (maxnodenamelen-1)));

#if defined(USE_MPI_IN_PLACE)
  // MPI >= 2.x implementations (e.g. NCSA/Cray Blue Waters)
  MPI_Allgather(MPI_IN_PLACE, sizeof(nodeinfo), MPI_BYTE,
                &nodes[     0], sizeof(nodeinfo), MPI_BYTE,
                MPI_COMM_WORLD);
#else
  // MPI 1.x
  MPI_Allgather(&nodes[mynode], sizeof(nodeinfo), MPI_BYTE,
                &nodes[     0], sizeof(nodeinfo), MPI_BYTE,
                MPI_COMM_WORLD);
#endif

  if (mynode == 0) {
    char msgtxt[1024];
    float totalspeed = 0.0;
    int totalcpus = 0;
    int totalgpus = 0;
    int i;

    for (i=0; i<numnodes; i++) {
      totalcpus  += nodes[i].numcpus;
      totalgpus  += nodes[i].numgpus;
      totalspeed += nodes[i].nodespeed;
    }

    sprintf(msgtxt, "Found %d VMD MPI node%s containing a total of %d CPU%s and %d GPU%s:",
            numnodes, (numnodes > 1) ? "s" : "",
            totalcpus, (totalcpus > 1) ? "s" : "",
            totalgpus, (totalgpus != 1) ? "s" : "");
    msgInfo << msgtxt << sendmsg;

    for (i=0; i<numnodes; i++) {
      sprintf(msgtxt,
              "%4d: %3d CPUs, %4.1fGB (%2ld%%) free mem, "
              "%d GPUs, "
//              "CPU Speed %4.2f, Node Speed %6.2f "
              "Name: %s",
              i, nodes[i].numcpus,
              nodes[i].corefree / 1024.0f, nodes[i].corepcnt,
              nodes[i].numgpus,
//            nodes[i].cpuspeed, nodes[i].nodespeed,
              nodes[i].machname);
      msgInfo << msgtxt << sendmsg;;
    }
  }

  // wait for node 0 console output to complete before peers 
  // continue with startup process
  MPI_Barrier(MPI_COMM_WORLD);

  *noderank=mynode;
  *nodecount=numnodes;

  free(nodes);
  return 0;
}

#endif
