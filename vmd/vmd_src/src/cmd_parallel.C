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
 *      $RCSfile: cmd_parallel.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.34 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   MPI-related text commands
 *
 * These commands are not logged or otherwise sent through the normal
 * VMD command queue as they are latency sensitive, and there would be
 * no obvious reason one would want to log them to files or to the console.
 *
 ***************************************************************************/

#if defined(VMDMPI)
#include <mpi.h>    // XXX this is a short term hack for testing only! 
                    // We should not be making MPI calls directly here,
                    // but rather calling via VMDApp, so that we can
                    // implement the needed MPI functionality in a dynamically
                    // loaded plugin, rather than being hard-compiled in...

// Check to see if we have to pass the MPI_IN_PLACE flag
// for in-place allgather reductions (same approach as Tachyon)
#if !defined(USE_MPI_IN_PLACE)
#if (MPI_VERSION >= 2) || defined(MPI_IN_PLACE)
#define USE_MPI_IN_PLACE 1
#endif
#endif

#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tcl.h>
#include "config.h"
#include "CommandQueue.h"
#include "Command.h"
#include "VMDApp.h"
#include "Inform.h" // debugging
#include "WKFThreads.h"

#define VMD_MPI_TAG_ALLREDUCE_ARGLENGTH 1
#define VMD_MPI_TAG_ALLREDUCE_PAYLOAD   2
#define VMD_MPI_TAG_FOR_REQUEST         3



//
// XXX hack to provide Swift/T with access to VMD's communicator
//
#if defined(VMDMPI)
//
// XXX global variable hack that needs to go away...
// MPI communicator made available to Swift/T during startup
//
MPI_Comm turbine_adlb_comm;
#endif

extern int swift_mpi_init(Tcl_Interp *interp) {
#if defined(VMDMPI)
  if (getenv("VMDNOSWIFTCOMM") == NULL) {
    if (MPI_SUCCESS == MPI_Comm_dup(MPI_COMM_WORLD, &turbine_adlb_comm)) {
      Tcl_Obj* TURBINE_ADLB_COMM = Tcl_NewStringObj("TURBINE_ADLB_COMM", -1);
      // XXX this is another gross hack.  This passes the MPI communicator pointer 
      //     as if it were a long, to make it available through Tcl, 
      //     but there MUST be a better way.  This is copied from what was done in NAMD.
      Tcl_Obj* adlb_comm_ptr = Tcl_NewLongObj((long) &turbine_adlb_comm);
      Tcl_ObjSetVar2(interp, TURBINE_ADLB_COMM, NULL, adlb_comm_ptr, 0);
    }
  }
#endif

  return 0;
}


//
// MPI dynamic work scheduler function
//
#if defined(VMDMPI)

typedef struct {
  int numnodes;
  wkf_tasktile_t loop;
  wkf_shared_iterator_t iter;
} parallel_for_parms;

extern "C" void *vmd_mpi_parallel_for_scheduler(void *voidparms) {
  parallel_for_parms *parfor = (parallel_for_parms *) voidparms;

  // Run the for loop management code on node zero.
  // Do the work on all the other nodes...
#if defined(VMDTHREADS)
  int i;
  wkf_tasktile_t curtile;
  while (wkf_shared_iterator_next_tile(&parfor->iter, 1, &curtile) != WKF_SCHED_DONE) {
    i = curtile.start;
#else
  int i;
  for (i=parfor->loop.start; i<parfor->loop.end; i++) {
#endif
    int reqnode;
    MPI_Status rcvstat;
    MPI_Recv(&reqnode, 1, MPI_INT, MPI_ANY_SOURCE, VMD_MPI_TAG_FOR_REQUEST, 
             MPI_COMM_WORLD, &rcvstat); 
    MPI_Send(&i, 1, MPI_INT, reqnode, VMD_MPI_TAG_FOR_REQUEST, 
             MPI_COMM_WORLD);
  }

  // tell all nodes we're done with all of the work
  int node;
  for (node=1; node<parfor->numnodes; node++) {
    int reqnode;
    MPI_Status rcvstat;
    MPI_Recv(&reqnode, 1, MPI_INT, MPI_ANY_SOURCE, VMD_MPI_TAG_FOR_REQUEST, 
             MPI_COMM_WORLD, &rcvstat); 

    i=-1; // indicate that the for loop is completed
    MPI_Send(&i, 1, MPI_INT, reqnode, VMD_MPI_TAG_FOR_REQUEST, 
             MPI_COMM_WORLD);
  }

  return NULL;
}

#endif



int text_cmd_parallel(ClientData cd, Tcl_Interp *interp, int argc, const char *argv[]) {
  VMDApp *app = (VMDApp *)cd;

  if(argc<2) {
    Tcl_SetResult(interp,
      (char *)
      "Parallel job query commands:\n"
      "  parallel nodename\n"
      "  parallel noderank\n"
      "  parallel nodecount\n"
      "Parallel collective operations (all nodes MUST participate):\n"
      "  parallel allgather <object>\n"
      "  parallel allreduce <tcl reduction proc> <object>\n"
      "  parallel barrier\n"
      "  parallel for <startcount> <endcount> <tcl callback proc> <user data>",
      TCL_STATIC);
    return TCL_ERROR;
  }

  // XXX hack to make Swift/T cooperate with VMD when using VMD's MPI
  // communicator
  if (!strcmp(argv[1], "swift_clone_communicator")) {
    swift_mpi_init(interp);
    return TCL_OK;
  }

  // return the MPI node name
  if (!strcmp(argv[1], "nodename")) {
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(app->par_name(), strlen(app->par_name())));
    Tcl_SetObjResult(interp, tcl_result);
    return TCL_OK;
  }

  // return the MPI node rank
  if (!strcmp(argv[1], "noderank")) {
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(app->par_rank()));
    Tcl_SetObjResult(interp, tcl_result);
    return TCL_OK;
  }

  // return the MPI node count
  if (!strcmp(argv[1], "nodecount")) {
    Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
    Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewIntObj(app->par_size()));
    Tcl_SetObjResult(interp, tcl_result);
    return TCL_OK;
  }

  // execute an MPI barrier
  if(!strupncmp(argv[1], "barrier", CMDLEN) && argc==2) {
    app->par_barrier();
    return TCL_OK;
  }


  // Execute a parallel for loop across all nodes
  //
  //  parallel for <startcount> <endcount> <callback proc> <user data>",
  //
  if (!strupncmp(argv[1], "for", CMDLEN)) {
    int isok = (argc == 6);
    int N = app->par_size();
    int start, end;

    if (Tcl_GetInt(interp, argv[2], &start) != TCL_OK ||
        Tcl_GetInt(interp, argv[3], &end) != TCL_OK) {
      isok = 0;
    }

    //
    // If there's only one node, short-circuit the parallel for
    //
    if (N == 1) {
      if (!isok) {
        Tcl_SetResult(interp, (char *) "invalid parallel for, missing parameter", TCL_STATIC);
        return TCL_ERROR;
      }

      // run for loop on one node...
      int i;
      for (i=start; i<=end; i++) { 
        char istr[128];
        sprintf(istr, "%d", i);
        if (Tcl_VarEval(interp, argv[4], " ", istr, " {",
                        argv[5], "} ", NULL) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "error occured during parallel for", TCL_STATIC);
        }
      }

      return TCL_OK;
    }

#if defined(VMDMPI)
    int allok = 0;

    // Check all node result codes before we continue with the reduction
    MPI_Allreduce(&isok, &allok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    // XXX we may want to verify that all nodes are going to call the same
    // reduction proc here before continuing further.

    if (!allok) {
      Tcl_SetResult(interp, (char *) "invalid parallel for, missing parameter on one or more nodes", TCL_STATIC);
      return TCL_ERROR;
    }

    // Run the for loop management code on node zero.
    // Do the work on all the other nodes...
    int i;
    if (app->par_rank() == 0) {
      // use multithreaded code path
      parallel_for_parms parfor;
      memset(&parfor, 0, sizeof(parfor));
      parfor.numnodes = N;
      parfor.loop.start=start;
      parfor.loop.end=end+1;
      wkf_shared_iterator_init(&parfor.iter);
      wkf_shared_iterator_set(&parfor.iter, &parfor.loop);

#if defined(VMDTHREADS)
      // run the MPI scheduler in a new child thread
      wkf_thread_t pft;
      wkf_thread_create(&pft, vmd_mpi_parallel_for_scheduler, &parfor);

      // run the Tcl in the main thread
      wkf_tasktile_t curtile;
      while (wkf_shared_iterator_next_tile(&parfor.iter, 1, &curtile) != WKF_SCHED_DONE) {
        i = curtile.start;
        char istr[128];
        sprintf(istr, "%d", i);
        if (Tcl_VarEval(interp, argv[4], " ", istr, " {",
                        argv[5], "} ", NULL) != TCL_OK) {
          Tcl_SetResult(interp, (char *) "error occured during parallel for", TCL_STATIC);
        }
      }

      // join up with the MPI scheduler thread
      wkf_thread_join(pft, NULL);
#else
      // if no threads, node zero only runs the scheduler and doesn't do work
      vmd_mpi_parallel_for_scheduler(&parfor);
#endif

      wkf_shared_iterator_destroy(&parfor.iter);
    } else {
      char istr[128];
      int done=0;
      int mynode=app->par_rank();
      while (!done) {
        MPI_Send(&mynode, 1, MPI_INT, 0, VMD_MPI_TAG_FOR_REQUEST, 
                 MPI_COMM_WORLD);

        MPI_Status rcvstat;
        MPI_Recv(&i, 1, MPI_INT, MPI_ANY_SOURCE, VMD_MPI_TAG_FOR_REQUEST, 
                 MPI_COMM_WORLD, &rcvstat); 
        if (i == -1) {
          done = 1;
        } else {
          sprintf(istr, "%d", i);
          if (Tcl_VarEval(interp, argv[4], " ", istr, " {",
                          argv[5], "} ", NULL) != TCL_OK) {
            Tcl_SetResult(interp, (char *) "error occured during parallel for", TCL_STATIC);
          }
        }
      }
    }
#endif

    return TCL_OK;
  }


  // Execute an allgather producing a Tcl list of the per-node contributions
  //
  // parallel allgather <object>
  //
  if (!strupncmp(argv[1], "allgather", CMDLEN)) {
    int isok = (argc == 3);
    int N = app->par_size();

    // when running on a single-node or MPI is disabled at runtime, 
    // bypass all MPI calls with the single-node fast path, which makes 
    // the code both faster and implicitly handles the case where 
    // MPI support is compiled and linked in, but disabled at runtime,
    // e.g., when running a VMD binary on a Cray login node.
    if (N == 1) {
      if (!isok) {
        Tcl_SetResult(interp, (char *) "invalid parallel gather, missing parameter on one or more nodes", TCL_STATIC);
        return TCL_ERROR;
      }

      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(argv[2], strlen(argv[2])));
      Tcl_SetObjResult(interp, tcl_result);
      return TCL_OK;
    } 
#if defined(VMDMPI)
    else {
      int allok = 0;
      int i;

      // Check all node result codes before we continue with the gather
      MPI_Allreduce(&isok, &allok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

      if (!allok) {
        Tcl_SetResult(interp, (char *) "invalid parallel gather, missing parameter on one or more nodes", TCL_STATIC);
        return TCL_ERROR;
      }

      // Collect parameter size data so we can allocate result buffers
      // before executing the gather
      int *szlist = new int[N];
      szlist[app->par_rank()] = strlen(argv[2])+1;

#if defined(USE_MPI_IN_PLACE)
      // MPI >= 2.x implementations (e.g. NCSA/Cray Blue Waters)
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT,
                    &szlist[0], 1, MPI_INT, MPI_COMM_WORLD);
#else
      // MPI 1.x
      MPI_Allgather(&szlist[app->par_rank()], 1, MPI_INT,
                    &szlist[0], 1, MPI_INT, MPI_COMM_WORLD);
#endif

      int totalsz = 0;
      int *displist = new int[N];
      for (i=0; i<N; i++) {
        displist[i]=totalsz;
        totalsz+=szlist[i];
      }

      char *recvbuf = new char[totalsz];
      memset(recvbuf, 0, totalsz);

      // Copy this node's data into the correct array position
      strcpy(&recvbuf[displist[app->par_rank()]], argv[2]);

      // Perform the parallel gather 
#if defined(USE_MPI_IN_PLACE)
      // MPI >= 2.x implementations (e.g. NCSA/Cray Blue Waters)
      MPI_Allgatherv(MPI_IN_PLACE, szlist[app->par_rank()], MPI_BYTE,
                     &recvbuf[0], szlist, displist, MPI_BYTE, MPI_COMM_WORLD);
#else
      // MPI 1.x
      MPI_Allgatherv(&recvbuf[displist[app->par_rank()]], szlist[app->par_rank()], MPI_BYTE,
                     &recvbuf[0], szlist, displist, MPI_BYTE, MPI_COMM_WORLD);
#endif

      // Build Tcl result from the array of results
      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      for (i=0; i<N; i++) {
        Tcl_ListObjAppendElement(interp, tcl_result, Tcl_NewStringObj(&recvbuf[displist[i]], szlist[i]-1));
      }
      Tcl_SetObjResult(interp, tcl_result);

      delete [] recvbuf;
      delete [] displist;
      delete [] szlist;
      return TCL_OK;
    }
#endif
  }


  //
  // Execute an All-Reduce across all of the nodes.
  // The user must provide a Tcl proc that performs the appropriate reduction
  // operation for a pair of data items, resulting in a single item.
  // Since the user may pass floating point data or perform reductions 
  // that give very slightly different answers depending on the order of
  // operations, the architecture or the host, or whether reductions on 
  // a given host are occuring on the CPU or on a heterogeneous accelerator 
  // or GPU of some kind, we must ensure that all nodes get a bit-identical
  // result.  When heterogeneous accelerators are involved, we can really 
  // only guarantee this by implementing the All-Reduce with a 
  // Reduce-then-Broadcast approach, where the reduction collapses the
  // result down to node zero, which then does a broadcast to all peers. 
  //
  // parallel allreduce <tcl reduction proc> <object>
  //
  if (!strupncmp(argv[1], "allreduce", CMDLEN)) {
    int isok = (argc == 4);
    int N = app->par_size();

    //
    // If there's only one node, short-circuit the full parallel reduction
    //
    if (N == 1) {
      if (!isok) {
        Tcl_SetResult(interp, (char *) "invalid parallel reduction, missing parameter", TCL_STATIC);
        return TCL_ERROR;
      }

      // return our result, no other reduction is necessary
      Tcl_SetObjResult(interp, Tcl_NewStringObj(argv[3], strlen(argv[3])));
      return TCL_OK;
    }

#if 1 && defined(VMDMPI)
    //
    // All-Reduce implementation based on a ring reduction followed by a 
    // broadcast from node zero.  This implementation gaurantees strict
    // ordering and will properly handle the case where one or more nodes
    // perform their reduction with slightly differing floating point 
    // rounding than others (e.g. using GPUs, heterogeneous nodes, etc),
    // and it works with any number of nodes.  While NOT latency-optimal,
    // this implementation is close to bandwidth-optimal which is helpful
    // for workstation clusters on non-switched networks or networks with
    // switches that cannot operate in a fully non-blocking manner.
    //
    int allok = 0;

    // Check all node result codes before we continue with the reduction
    MPI_Allreduce(&isok, &allok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    // XXX we may want to verify that all nodes are going to call the same
    // reduction proc here before continuing further.

    if (!allok) {
      Tcl_SetResult(interp, (char *) "invalid parallel reduction, missing parameter on one or more nodes", TCL_STATIC);
      return TCL_ERROR;
    }

    // copy incoming data into initial "result" object
    Tcl_Obj *resultobj = Tcl_NewStringObj((const char *) argv[3], strlen(argv[3])+1);

    // A ring-based all-reduce implementation which should be 
    // close to bandwidth-optimal, at the cost of additional latency.
    int src=app->par_rank(); // src node is this node
    int Ldest = (N + src + 1) % N; // compute left peer
    int Rdest = (N + src - 1) % N; // compute right peer
    MPI_Status status;

    if (src != 0) {
      int recvsz = 0;

      // Post blocking receive for data size
      MPI_Recv(&recvsz, 1, MPI_INT, Ldest,
               VMD_MPI_TAG_ALLREDUCE_ARGLENGTH, MPI_COMM_WORLD, &status);

      // Allocate or resize receive buffer 
      char * recvbuf = (char *) malloc(recvsz);

      // Post non-blocking receive for data
      MPI_Recv(recvbuf, recvsz, MPI_BYTE, Ldest,
               VMD_MPI_TAG_ALLREDUCE_PAYLOAD, MPI_COMM_WORLD, &status);

      // Perform reduction
      // Perform the reduction operation on our existing and incoming data.
      // We build a Tcl command string with the user-defined proc, this 
      // node's previous resultand, and the incoming data, and evaluate it.
      if (Tcl_VarEval(interp, argv[2], " ", 
                      Tcl_GetString(resultobj), " ", 
                      recvbuf, NULL) != TCL_OK) {
        printf("Error occured during reduction!\n");    
      }

      // Prep for next reduction step.  Set result object to result of
      // the latest communication/reduction phase.
      resultobj = Tcl_GetObjResult(interp);

      // Free the receive buffer
      free(recvbuf);
    } 
  
    //
    // All nodes
    //  
    char *sendbuf = Tcl_GetString(resultobj);
    int sendsz = strlen(sendbuf)+1;

    // Post blocking send for data size
    MPI_Send(&sendsz, 1, MPI_INT, Rdest,
             VMD_MPI_TAG_ALLREDUCE_ARGLENGTH, MPI_COMM_WORLD);

    // Post blocking send for data
    MPI_Send(sendbuf, sendsz, MPI_BYTE, Rdest,
             VMD_MPI_TAG_ALLREDUCE_PAYLOAD, MPI_COMM_WORLD);

    if (src == 0) {
      int recvsz = 0;

      // Post blocking receive for data size
      MPI_Recv(&recvsz, 1, MPI_INT, Ldest,
               VMD_MPI_TAG_ALLREDUCE_ARGLENGTH, MPI_COMM_WORLD, &status);

      // Allocate or resize receive buffer 
      char * recvbuf = (char *) malloc(recvsz);

      // Post non-blocking receive for data
      MPI_Recv(recvbuf, recvsz, MPI_BYTE, Ldest,
               VMD_MPI_TAG_ALLREDUCE_PAYLOAD, MPI_COMM_WORLD, &status);

      // Perform reduction
      // Perform the reduction operation on our existing and incoming data.
      // We build a Tcl command string with the user-defined proc, this 
      // node's previous result and the incoming data, and evaluate it.
      if (Tcl_VarEval(interp, argv[2], " ", 
                      Tcl_GetString(resultobj), " ", 
                      recvbuf, NULL) != TCL_OK) {
        printf("Error occured during reduction!\n");    
      }

      // Prep for next reduction step.  Set result object to result of
      // the latest communication/reduction phase.
      resultobj = Tcl_GetObjResult(interp);

      // Free the receive buffer
      free(recvbuf);
    } 

    //
    // Broadcast final result from root to peers
    //
    if (src == 0) {
      // update send buffer for root node before broadcast
      sendbuf = Tcl_GetString(resultobj);
      sendsz = strlen(sendbuf)+1;
      MPI_Bcast(&sendsz, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(sendbuf, sendsz, MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
      int recvsz = 0;
      MPI_Bcast(&recvsz, 1, MPI_INT, 0, MPI_COMM_WORLD);

      // Allocate or resize receive buffer 
      char * recvbuf = (char *) malloc(recvsz);

      MPI_Bcast(recvbuf, recvsz, MPI_BYTE, 0, MPI_COMM_WORLD);

      // Set the final Tcl result if necessary
      Tcl_SetObjResult(interp, Tcl_NewStringObj(recvbuf, recvsz-1));

      // Free the receive buffer
      free(recvbuf);
    }

    return TCL_OK;

#elif defined(VMDMPI)

    //
    // Power-of-two-only hypercube/butterfly/recursive doubling 
    // All-Reduce implementation.  This implementation can't be used
    // in the case that we have either a non-power-of-two node count or
    // in the case where we have heterogeneous processing units that may
    // yield different floating point rounding.  For now we leave this
    // implementation in the code for performance comparisons until we work
    // out the changes necessary to make it closer to bandwidth-optimal,
    // heterogeneous-safe, and non-power-of-two capable.
    //
    int allok = 0;
    int i;

    // Check all node result codes before we continue with the reduction
    MPI_Allreduce(&isok, &allok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    // XXX we may want to verify that all nodes are going to call the same
    // reduction proc here before continuing further.

    if (!allok) {
      Tcl_SetResult(interp, (char *) "invalid parallel reduction, missing parameter on one or more nodes", TCL_STATIC);
      return TCL_ERROR;
    }

    // Calculate number of reduction phases required
    int log2N;
    for (log2N=0; N>1; N>>=1) {
      log2N++;

      // XXX bail out of we don't have a power-of-two node count, 
      //     at least until we implement 3-2 reduction phases
      if ((N & 1) && (N > 1)) {
        Tcl_SetResult(interp, (char *) "parallel allreduce only allowed for even power-of-two node count", TCL_STATIC);
        return TCL_ERROR;
      }
    }
    N = app->par_size();

    // copy incoming data into initial "result" object
    Tcl_Obj *resultobj = Tcl_NewStringObj((const char *) argv[3], strlen(argv[3])+1);

    // An all-reduce tree with hypercube connectivity with 
    // log2(N) communication/reduction phases.  At each phase, we compute
    // the peer/destination node we will communicate with using an XOR of
    // our node ID with the current hypercube dimension.  If we have an
    // incomplete hypercube topology (e.g. non-power-of-two node count), 
    // we have to do special 3-2 communication rounds (not implemented yet).
    // The current implementation requires that all existing nodes 
    // participate, and that they contribute a valid data item.
    // If we wish to support reductions where a node may not contribute,
    // we would need to handle that similarly to a peer node that doesn't 
    // exist, but we would likely determine this during the parameter length
    // exchange step.
    int src=app->par_rank(); // src node is this node
    for (i=0; i<log2N; i++) {
      int mask = 1 << i;     // generate bitmask to use in the XOR
      int dest = src ^ mask; // XOR src node with bitmask to find dest node
      Tcl_Obj *oldresultobj = resultobj; // track old result 

      // Check to make sure dest node exists for non-power-of-two 
      // node counts (an incomplete hypercube).  If not, skip to the next
      // communication/reduction phase.
      if (dest < N) {
        char *sendbuf = Tcl_GetString(oldresultobj);
        int sendsz = strlen(sendbuf)+1;
        int recvsz = 0;
        MPI_Request handle;
        MPI_Status status;

        //
        // Exchange required receive buffer size for data exchange with peer
        //

        // Post non-blocking receive for data size
        MPI_Irecv(&recvsz, 1, MPI_INT, dest,
                  VMD_MPI_TAG_ALLREDUCE_ARGLENGTH, MPI_COMM_WORLD, &handle);

        // Post blocking send for data size
        MPI_Send(&sendsz, 1, MPI_INT, dest,
                 VMD_MPI_TAG_ALLREDUCE_ARGLENGTH, MPI_COMM_WORLD);

        // Wait for non-blocking receive of data size to complete
        MPI_Wait(&handle, &status); 

// printf("src[%d], dest[%d], value '%s', recvsz: %d\n", src, dest, sendbuf, recvsz);

        // Allocate or resize receive buffer 
        char * recvbuf = (char *) malloc(recvsz);

        //
        // Exchange the data payload
        // 

        // Post non-blocking receive for data
        MPI_Irecv(recvbuf, recvsz, MPI_BYTE, dest,
                  VMD_MPI_TAG_ALLREDUCE_PAYLOAD, MPI_COMM_WORLD, &handle);
        
        // Post blocking send for data
        MPI_Send(sendbuf, sendsz, MPI_BYTE, dest,
                 VMD_MPI_TAG_ALLREDUCE_PAYLOAD, MPI_COMM_WORLD);

        // Wait for receive of data
        MPI_Wait(&handle, &status); 

        // Perform the reduction operation on our existing and incoming data.
        // We build a Tcl command string with the user-defined proc, this 
        // node's previous result and the incoming data, and evaluate it.
        if (Tcl_VarEval(interp, argv[2], " ", 
                        sendbuf, " ", recvbuf, NULL) != TCL_OK) {
          printf("Error occured during reduction!\n");    
        }

        // Free the receive buffer
        free(recvbuf);

        // Prep for next reduction step.  Set result object to result of
        // the latest communication/reduction phase.
        resultobj = Tcl_GetObjResult(interp);
      }
    }

    // Set the final Tcl result if necessary
    Tcl_SetObjResult(interp, resultobj);

    return TCL_OK;
#endif
  }

  Tcl_SetResult(interp, (char *) "invalid parallel subcommand.", TCL_STATIC);

  return TCL_ERROR;
}

