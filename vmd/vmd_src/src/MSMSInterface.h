#ifndef MSMSINTERFACE_H
#define MSMSINTERFACE_H

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
 *	$RCSfile: MSMSInterface.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Communicate with the MSMS surface generation program.  For more
 * information about MSMS, please see:
 * http://www.scripps.edu/pub/olson-web/people/sanner/html/msms_home.html
 *
 ***************************************************************************/

// MSMS surface generation process:
//   Find a free port.
//   Start off MSMS server at that port (run process in background)
//   Connect to MSMS server
//   Send coords and get back surface information
//   Pass this data to MSMS server:
//     index, x, y, z, radius, use_atom
//   MSMS server returns this data:
//     face list containing 3 vertex points each
//     atomid, as mapped to input values
//     position list containing x, y, z
//     norm list containing normx, normy, normz

#include "ResizeArray.h"

/// structure containing MSMS vertex coordinates
struct MSMSCoord {
  float x[3];       ///< floating point xyz coordinates 
  int operator==(const MSMSCoord& c) {
    return !memcmp(x, c.x, 3L*sizeof(float));
  }
};

/// structure containing MSMS facet information
struct MSMSFace {
  int vertex[3];    ///< 1-based vertex indices

  int surface_type; ///< 1 - triangle in a toric reentrant face
                    ///< 2 - triangle in a sphereic reentrant face
                    ///< 3 - triangle in a contact face   

  int anaface;      ///< 1-based face number in the analytical description
                    ///< of the solvent excluded surface

  int component;    ///< which surface is it in?

  int operator==(const MSMSFace &f) {
    return (!memcmp(vertex, f.vertex, 3L*sizeof(float)) &&
                    surface_type==f.surface_type && anaface==f.anaface &&
                    component==f.component);
  }
};

/// Manages communication with the MSMS surface generation program
/// Can only use this class once!
class MSMSInterface {
private:
  void *msms_server;               ///< socket handle for msms connection.
  static int find_free_port(void); ///< find a free port, or return 0

  /// what is the MSMSSERVER name? (uses env. variable MSMSSERVER or "msms")
  static const char *server_name(void);

  static int start_msms(int port); ///< start the MSMS process running

  /// connect to MSMS (return 0 on error, or socket handle)
  static void *conn_to_service_port(int portnum);

  /// wait for n seconds.  If timeout reached, close msms_server
  /// and return 0, else return 1 .  Try num_reps times
  int check_for_input(int timeout_in_seconds, int num_reps, int stage);

  /// send information to MSMS (msms_server must be set by now)
  int call_msms(float probe_radius, float density,
		int n, float *xyzr, int *flgs);

  /// what is in the next information block?  (Or is this the end?)
  int msms_ended(void);

  /// close the server, and set an error, if there was a problem
  void close_server(int erno = 0);

  /// get triangulation information back
  void get_triangulated_ses(int component);

  /// get information from the connection
  /// will effectively timeout after a few seconds (returns -1 in that case)
  /// otherwise returns the number of bytes read
  int get_blocking(char *str, int nbytes);

  /// read data into a buffer (must be of size 256 or greater!)
  /// returns the buffer pointer
  char *get_message(char *buffer);

public:
  int err;                         ///< was there an error?
  MSMSInterface(void) { err = 0; }
  ResizeArray<int>       atomids;  ///< mapping of vertices to msms/vmd atoms
  ResizeArray<MSMSFace>  faces;    ///< solvent excluded surface facet list
  ResizeArray<MSMSCoord> coords;   ///< vertex list referenced by facet list
  ResizeArray<MSMSCoord> norms;    ///< normal list referenced by facet list

  /// return 1 on success
  enum {BAD_RANGE = -2, NO_PORTS = -3, NO_CONNECTION = -4,
	NO_INITIALIZATION = -5, MSMS_DIED = -6, COMPUTED = 1};
  int compute_from_socket(float probe_radius, float density,
	      int n, int *ids, float *xyzr, int *flgs, int component = 0);

  /// free memory in the ResizeArray's.
  void clear();

  // use file interface instead of sockets
  int compute_from_file(float probe_radius, float density,
	      int n, int *ids, float *xyzr, int *flgs, int component = 0);
};
  
#endif

