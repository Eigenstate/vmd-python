/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 *cr Some of these lines may be copyright Michel Sanner or Scripps as
 *cr they come from example code of the MSMS distribution
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: MSMSInterface.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.52 $	$Date: 2010/12/16 04:08:21 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Start the MSMS server and talk to it.  For more information about
 * MSMS, see 
 * http://www.scripps.edu/pub/olson-web/people/sanner/html/msms_home.html
 *
 ***************************************************************************/

#include "utilities.h"
#include "vmdsock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#if defined(__irix)
#include <bstring.h>
#endif

#if defined(__hpux)
#include <time.h>
#endif

#include "MSMSInterface.h"
#include "Inform.h"

#define MIN_PORT 1357
#define MAX_PORT 9457

// find port for connecting to MSMS; return port
int MSMSInterface::find_free_port(void) {

  void *sock = vmdsock_create(); 
  if (!sock) return 0;

  int port = 0;
  // search for a free port
  for (int i=MIN_PORT; i<=MAX_PORT; i++) {
    if (vmdsock_bind(sock, i) == 0) {
      port = i;
      break;
    }
  }
  vmdsock_destroy(sock); 

  if (port == 0) {
    msgErr << "Could not find an available port between " << 
           MIN_PORT << " and " << MAX_PORT << "." << sendmsg;
  }
  return port; // return port number or 0 if none available
}


const char *MSMSInterface::server_name(void) {
  const char *msms = getenv("MSMSSERVER");
  if (!msms) {
#ifdef _MSC_VER
    msms = "msms.exe";
#else
    msms = "msms";
#endif
  }

  return msms;
}


// returns 1
int MSMSInterface::start_msms(int port) {
  const char *msms = server_name();
  char *s = new char[strlen(msms) + 100];
  sprintf(s, "%s -no_area -socketPort %d &", msms, port);
  msgInfo << "Starting MSMS with: '" << s << "'" << sendmsg;
  vmd_system(s);
  delete [] s;
  return 1;
}



// return 0 on failure, or vmdsock handle 
void *MSMSInterface::conn_to_service_port(int port) {
  void *sock;

  sock = vmdsock_create();
  if (!sock) return NULL;
  if (vmdsock_connect(sock, "localhost", port)) {
    vmdsock_destroy(sock);
    sock = NULL;
  }
  return sock;
}


int MSMSInterface::compute_from_socket(float probe_radius, float density,
                                       int n, int *ids, float *xyzr, int *flgs,
                                       int /* component */) {
  if (xyzr == NULL) {
    err = BAD_RANGE;
    return err;
  }

  // Find a free port
  int port = find_free_port();
  if (port == 0) {
    err = NO_PORTS;
    return err;
  }

  // spawn an MSMS server pointed at that port
  start_msms(port);

  // try to connect up to 100 times with a linear backoff
  for (int loop=0; loop<100; loop++) {
    msms_server = conn_to_service_port(port);
    if (!msms_server) {
      if (loop!=0 && !(loop%50)) {
        msgInfo << "Waiting for MSMS server ..." << sendmsg;
      }
      vmd_msleep(loop);
    } else {
      break;
    }
  }

  if (msms_server == 0) {
    msgErr << "Could not connect to MSMS server.  " <<
           "Please check that the program '" << server_name() << 
           "' exists and is executable, or set the environment variable " <<
           "MSMSSERVER to point to the correct binary." << sendmsg;
    err = NO_CONNECTION; // (try to) kill server now?
    return err;
  }

  // from here on, error is set by close_server();

  // send info to server
  if (!call_msms(probe_radius, density, n, xyzr, flgs)) {
    return err;
  }

  // get data back
  int t;
  int surface_count = 0;
  do {
    t = msms_ended();
    //    printf("t is %d\n", t); fflush(stdout);
    switch (t) {
    case 5: get_triangulated_ses(surface_count++); break;
    case 1: close_server(COMPUTED); break;
    default: break;
    }
  } while (t!=1 && msms_server != 0);

  if (err != COMPUTED) {
    return err;
  }

  // map the atom ids as (and if) requested
  // atomids is an array of MSMS atoms indices which correspond to the
  // MSMS vertices. (one array element for each MSMS vertex)
  // ids is an array of VMD atom indices which correspond to the
  // MSMS atoms. (one array element for each MSMS atom)
  if (ids != NULL) {
    for (int i=0; i<atomids.num(); i++) {
      atomids[i] = ids[atomids[i]];
    }
  }
  return COMPUTED; // it all worked!
}


int MSMSInterface::check_for_input(int secs, int reps, int stage) {
  for (int count = 0; count < reps; count++) {
    int ret = vmdsock_selread(msms_server, secs);
    if (ret > 0) {  // got something
      return 1;  // return success
    }

    if (ret == 0) { 
      // select timeout reached
      msgInfo << "Waiting for data from MSMS(" << stage << ") ..." << sendmsg;
    } else {
      // select returned an error.
      msgErr << "Unknown error " << ret << "with MSMS interface!" << sendmsg;
      perror("Did you press Control-C? : ");
      break;
    }
  }

  vmdsock_destroy(msms_server); // never got anything, close the connection and
  return 0;                     // return failure (never got anything)
}


char *MSMSInterface::get_message(char *buffer) {
  int i,j;
  char *car;

  if (!check_for_input(1, 10, 1)) {
    return NULL;
  }

  for (i=0,car=buffer; car-buffer<255; car++) {
    j=vmdsock_read(msms_server, car, 1);
    if (j!=-1) {
      i++;
      if (*car=='\n') {*car='\0';break;}
    }
  }

  if (*car!='\0') 
    buffer[255]='\0';

  return(buffer);
}


int MSMSInterface::call_msms(float probe_radius, float density,
                             int n, float *xyzr, int *flgs) {
  int mask1 = 0; 
  int mask2 = 1;
  int  flag = 1;

  char buffer[256];
  if (!get_message(buffer)) {
    msgErr << "Couldn't send initialization to MSMS" << sendmsg;
    close_server(NO_INITIALIZATION);
    return 0; // return error
  }

  vmdsock_write(msms_server,(char *)&probe_radius, sizeof(float)); 
  vmdsock_write(msms_server,(char *)&density, sizeof(float)); 
  vmdsock_write(msms_server,(char *)&mask1, sizeof(int));
  vmdsock_write(msms_server,(char *)&mask2, sizeof(int));
  vmdsock_write(msms_server,(char *)&n, sizeof(int));
  for (int i=0; i<n; i++) {
    vmdsock_write(msms_server,(char *)(xyzr+4*i) , 4*sizeof(float));

    // If no flags passed, set a default
    if (flgs) {
      vmdsock_write(msms_server, (char *)(flgs+i), sizeof(int));
    } else {
      vmdsock_write(msms_server, (char *)&flag, sizeof(int));
    }
  }

  return 1; // return success
}


int MSMSInterface::msms_ended(void) {
  char buffer[256];
  int nread = 0;
  char c = '\0';
  int i;
  //int fl;
  //fl = fcntl(msms_server, F_GETFL, 0);         // get current setting
  //fcntl(msms_server, F_SETFL, O_NDELAY | fl);  // don't block

  if (!check_for_input(10, 12, 2)) { // two minutes (compute could be long)
    msgErr << "No information from MSMS.. giving up." << sendmsg;
    close_server(MSMS_DIED);
    return 1; // for a premature end
  }

  while (1) {
    // once I am getting data, I had better get it
    if (!check_for_input(1, 4, 3)) {
      msgErr << "No data from MSMS.. giving up." << sendmsg;
      close_server(MSMS_DIED);
      return 1; // for a premature end
    }

    i = vmdsock_read(msms_server, &c, 1);              // get a character
    if (i != -1 && i > 0) {
      buffer[nread] = c;
      nread++;
      if (c == '\n') {         // read new line, so end of header
        //fcntl(msms_server, F_SETFL, ~O_NDELAY & fl); // restore setting
        buffer[nread-1] = '\0';
        if (strcmp(buffer,"MSMS END")==0)       return(1);
        if (strcmp(buffer,"MSMS RS")==0)        return(2);
        if (strcmp(buffer,"MSMS CS")==0)        return(3);
        if (strcmp(buffer,"MSMS SEND DOTS")==0) return(4);
        if (strcmp(buffer,"MSMS RCF")==0)       return(5);

        msgErr << "Unknown MSMS message: " << buffer << sendmsg;
        return 1; // for a premature end
      }
    } else {
      // Why did I get a -1 or 0?  
      // Probably bacause the server crashed, or didn't flush its output
      // on exit?
      if (nread == 0)
        msgErr << "No data from MSMS.. giving up." << sendmsg;

      close_server(COMPUTED); // Should return MSMS_DIED, once error handling
                              // is fixed and we know msms is working right.
      return 1;
    }
  }
}


int MSMSInterface::get_blocking(char *str, int nbytes) {
  int i=0, to_be_read;
  char *cptr=str;
  
  // 30 seconds (3 seconds at a time)
  if (!check_for_input(3, 10, 4)) {
    msgErr << "Failed in MSMSInterface::get_blocking" << sendmsg;
    return -1; // return failure
  }; 

  for (to_be_read = nbytes; to_be_read > 0; cptr += i) {
    i = vmdsock_read(msms_server, cptr, to_be_read);

    if (i==-1) 
      return(nbytes - to_be_read);  // connection died?

    to_be_read -= i;
  }

  return nbytes; // return number of bytes read
}



void MSMSInterface::get_triangulated_ses(int component) {
  int    i, i1, i2, nf, ns;
  char  *buffer;
  float *bf;
  int   *bi, s1, s2, max;

  // get number of facets
  i1=get_blocking((char *)&nf, sizeof(int));
  if (i1 < 0) { close_server(MSMS_DIED); return; }

  // number of vertices
  i1=get_blocking((char *)&ns, sizeof(int));
  if (i1 < 0) { close_server(MSMS_DIED); return; }

  s1 = nf*5*sizeof(int);
  s2 = ns*(6*sizeof(float) + sizeof(int));
  
  max = (s2>s1) ? s2:s1;
  buffer = (char *)malloc(max*sizeof(char) + 1);
  if (buffer==NULL) {
    msgErr << "MSMS: allocation failed for buffer" << sendmsg;
    return;
  }

  // read in facet list
  i1 = get_blocking(buffer, s1*sizeof(char));
  if (i1 < 0) { close_server(MSMS_DIED); free(buffer); return; }
  bi = (int *)buffer;
  for (i=0;i<nf;i++) {
    MSMSFace face;
    face.vertex[0]=*bi++;
    face.vertex[1]=*bi++;
    face.vertex[2]=*bi++;
    face.surface_type=*bi++;
    face.anaface=*bi++;
    face.component = component;
    faces.append(face);
  }

  // read in vertex and normal lists
  i2 = get_blocking(buffer,s2*sizeof(char));
  if (i2 < 0) { close_server(MSMS_DIED); free(buffer); return; }
  bf = (float *)buffer;
  for (i=0; i<ns; i++) {
    MSMSCoord norm, coord;
    norm.x[0]=*bf++;
    norm.x[1]=*bf++;
    norm.x[2]=*bf++;
    coord.x[0]=*bf++;
    coord.x[1]=*bf++;
    coord.x[2]=*bf++;
    norms.append(norm);
    coords.append(coord);
  }
 
  // read in atomid mappings for each vertex 
  bi = (int *)bf;
  for (i=0;i<ns;i++) {
    atomids.append(*bi++);
  }

  free(buffer);
}



void MSMSInterface::close_server(int erno) {
  if (msms_server != 0) {
    err = erno;
    vmdsock_destroy(msms_server);
    msms_server = 0;
  }
}

void MSMSInterface::clear() {
  atomids.clear();
  faces.clear();
  coords.clear();
  norms.clear();
}

// mark a file for destruction when the object goes out of scope
class VMDTempFile {
private:
  const char *m_filename;
public:
  VMDTempFile(const char *fname) {
    m_filename = stringdup(fname);
  }
  ~VMDTempFile() {
    vmd_delete_file(m_filename);
  }
};


int MSMSInterface::compute_from_file(float probe_radius, float density,
                                     int n, int *ids, float *xyzr, int *flgs,
                                     int /* component */) {
  const char *msmsbin = server_name();

  // generate output file name and try to create
  char *dirname = vmd_tempfile("");
  char *ofilename = new char[strlen(dirname) + 100];
  sprintf(ofilename, "%svmdmsms.u%d.%d", 
      dirname, vmd_getuid(), (int)(vmd_random() % 999));
  delete [] dirname;
  FILE *ofile = fopen(ofilename, "wt");
  if (!ofile) {
    delete [] ofilename;
    msgErr << "Failed to create MSMS atom radii input file" << sendmsg;
    return 0;  // failure
  }

  // MSMS-generated input files append .vert and .face to what you give
  // it, so might as well use the same name as the output file.
  char *facetfilename = new char[strlen(ofilename) + 6];
  char *vertfilename = new char[strlen(ofilename) + 6];
  sprintf(facetfilename, "%s.face", ofilename);
  sprintf(vertfilename, "%s.vert", ofilename);

  // temporary files we want to make sure to clean up
  VMDTempFile otemp(ofilename);
  VMDTempFile ftemp(facetfilename);
  VMDTempFile vtemp(vertfilename);

  //
  // write atom coordinates and radii to the file we send to MSMS 
  //
  for (int i=0; i<n; i++) {
    fprintf(ofile, "%f %f %f %f\n", xyzr[4*i], xyzr[4*i+1], xyzr[4*i+2],
        xyzr[4*i+3]);
  }
  fclose(ofile);

  //
  // call MSMS to calculate the surface for the given atoms
  //
  {
    char *msmscmd = new char[2*strlen(ofilename) + strlen(msmsbin) + 100];
    sprintf(msmscmd, "\"%s\" -if %s -of %s -probe_radius %5.3f -density %5.3f -no_area -no_header", msmsbin, ofilename, ofilename, probe_radius, density);
    vmd_system(msmscmd);    
    delete [] msmscmd;
  }
  
  // read facets
  FILE *facetfile = fopen(facetfilename, "r");
  if (!facetfile) {
    msgErr << "Cannot read MSMS facet file: " << facetfilename << sendmsg;
    // Return cleanly, deleting temp files and so on. 
    return 0;  // failed
  }
  MSMSFace face;
  while (fscanf(facetfile, "%d %d %d %d %d",
        face.vertex+0, face.vertex+1, face.vertex+2, &face.surface_type,
        &face.anaface) == 5) {
    face.component = 0;  // XXX Unused by VMD, so why store?
    face.vertex[0]--;
    face.vertex[1]--;
    face.vertex[2]--;
    faces.append(face);
  }
  fclose(facetfile);

  // read verts
  FILE *vertfile = fopen(vertfilename, "r");
  if (!vertfile) {
    msgErr << "Cannot read MSMS vertex file: " << vertfilename << sendmsg;
    return 0;  // failed
  }
  MSMSCoord norm, coord;
  int atomid;  // 1-based atom index
  int l0fa;    // number of of the level 0 SES face
  int l;       // SES face level? (1/2/3)
  while (fscanf(vertfile, "%f %f %f %f %f %f %d %d %d",
        coord.x+0, coord.x+1, coord.x+2, 
        norm.x+0, norm.x+1, norm.x+2, 
        &l0fa, &atomid, &l) == 9) {
    norms.append(norm);
    coords.append(coord);
    atomids.append(atomid-1);
  }
  fclose(vertfile);

  if (ids) {
    for (int i=0; i<atomids.num(); i++) {
      atomids[i] = ids[atomids[i]];
    }
  }
  return 1; // success
}

