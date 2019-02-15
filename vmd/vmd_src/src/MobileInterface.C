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
 *	$RCSfile: MobileInterface.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.49 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Mobile UI object, which maintains the current state of connected
 * smartmobile/tablet clients.
 *
 ***************************************************************************
 * TODO list:
 *      Pretty much everything          
 *
 ***************************************************************************/

/* socket stuff */
#if defined(_MSC_VER)
#include <winsock2.h>
#else
#include <stdio.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <time.h>
#include <netinet/in.h>
#endif

#include "MobileInterface.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"
#include "PickList.h"
#include "Animation.h"
#include "VMDApp.h"
#include "math.h"
#include "stdlib.h" // for getenv(), abs() etc.

// ------------------------------------------------------------------------
// maximum API version that we understand
#define CURRENTAPIVERSION       9 

// packet contains:
#define PACKET_ORIENT       1    //   device orientation (gyro, accel, etc)
#define PACKET_TOUCH        2    //   touchpad events (up, down, move, etc)
#define PACKET_HEARTBEAT    3    //   heartbeat.. likely every second
#define PACKET_CONNECT      4    //   information about new connection
#define PACKET_DISCONNECT   5    //   notice that a device has disconnected
#define PACKET_BUTTON       6    //   button state has changed.

// what type of event are we getting from client
#define EVENT_NON_TOUCH     -1
#define EVENT_TOUCH_DOWN     0
#define EVENT_TOUCH_UP       1
#define EVENT_TOUCH_MOVE     2
#define EVENT_TOUCH_SOMEUP   5
#define EVENT_TOUCH_SOMEDOWN 6
#define EVENT_COMMAND        7

// what type of event are we sending to mobile client
#define SEND_HEARTBEAT          0
#define SEND_ADDCLIENT          1
#define SEND_REMOVECLIENT       2
#define SEND_SETACTIVECLIENT    3
#define SEND_SETMODE            4
#define SEND_MESSAGE            5

// ------------------------------------------------------------------------
/* Only works with aligned 4-byte quantities, will cause a bus error */
/* on some platforms if used on unaligned data.                      */
static void swap4_aligned(void *v, long ndata) {
  int *data = (int *) v;
  long i;
  int *N;
  for (i=0; i<ndata; i++) {
    N = data + i;
    *N=(((*N>>24)&0xff) | ((*N&0xff)<<24) |
        ((*N>>8)&0xff00) | ((*N&0xff00)<<8));
  }
}

// ------------------------------------------------------------------------
void Mobile::prepareSendBuffer(const int eventType) {
  memset(statusSendBuffer, 0, sizeof(statusSendBuffer));
  int caretLoc = 0;
  int itmp=0;

  // set endianism flag
  // *(int*)(statusSendBuffer + caretLoc) = 1;    // endianism 
  itmp=1;
  memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
  caretLoc += sizeof(int);

  // set API version
  // *(int*)(statusSendBuffer + caretLoc) = CURRENTAPIVERSION;    // version
  itmp=CURRENTAPIVERSION;
  memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
  caretLoc += sizeof(int);

  // current mode in VMD (off, move, animate, tracker, etc)
  // *(int*)(statusSendBuffer + caretLoc) = moveMode;    // off/move/animate/etc
  itmp=moveMode; // off/move/animate/etc
  memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
  caretLoc += sizeof(int);

  // event code for what we are sending
  // *(int*)(statusSendBuffer + caretLoc) = eventType;    
  itmp=eventType;
  memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
  caretLoc += sizeof(int);

  // we are putting a placeholder in.. at position 16, where we will insert
  // whether or not this specific user is active.
  caretLoc += 4;

  // number of connections
  // *(int*)(statusSendBuffer + caretLoc) = clientNick.num();    // how many connections?
  itmp=clientNick.num();    // how many connections?
  memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
  caretLoc += sizeof(int);
//  fprintf(stderr, "num connections: %d\n", clientNick.num());

  // names of who is connected, in control
  for (int i=0; i<clientNick.num();i++) {
    int strLength = strlen((const char *)(*(clientNick)[i]));
    // *(int*)(statusSendBuffer + caretLoc) = strLength;
    itmp=strLength;    // how many connections?
    memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
    caretLoc += sizeof(int);
//    fprintf(stderr, "nick is '%s'\n", (const char *)(*(clientNick)[i]));

    memcpy((statusSendBuffer + caretLoc), (const char *)(*(clientNick)[i]), strLength);
    caretLoc += strLength;  

//    fprintf(stderr, "clientactive is '%d'\n", (clientActive[i] ? 1 : 0));
//    *(int*)(statusSendBuffer + caretLoc) = (clientActive[i] ? 1 : 0);    // Is this nick active?
    itmp=(clientActive[i] ? 1 : 0);    // Is this nick active?
    memcpy(statusSendBuffer + caretLoc, &itmp, sizeof(itmp)); 
    caretLoc += sizeof(int);
  }
   
  // XXX could send different configurations to different clients (client
  // in control might get different setup)
  // we need to send:
  // desired button states

  // caretLoc is also the length of the useful data in the packet
  statusSendBufferLength = caretLoc;
}


// ------------------------------------------------------------------------
typedef struct {
  /* socket management data */
  char buffer[1024];
  struct sockaddr_in sockaddr;
#if defined(_MSC_VER)
  SOCKET sockfd;
#else
  int sockfd;
#endif
  int fromlen;

  /* mobile state vector */
  int seqnum;
  int buttons;
  float rx;
  float ry;
  float rz;
  float tx;
  float ty;
  float tz;
  int padaction;
  int touchcnt;
  int upid;
  int touchid[16];
  float padx[16];
  float pady[16];
  float rotmatrix[9];
} mobilehandle;


// ------------------------------------------------------------------------
static void * mobile_listener_create(int port) {
  mobilehandle *ph = (mobilehandle *) calloc(1, sizeof(mobilehandle));
  if (ph == NULL)
    return NULL;

#if defined(_MSC_VER)
  // ensure that winsock is initialized
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR)
    return NULL;
#endif

  if ((ph->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("socket: ");
    free(ph);
    return NULL;
  }

  /* make socket non-blocking */
#if defined(_MSC_VER)
  u_long nonblock = 1;
  ioctlsocket(ph->sockfd, FIONBIO, &nonblock);
#else
  int sockflags;
  sockflags = fcntl(ph->sockfd, F_GETFL, 0);
  fcntl(ph->sockfd, F_SETFL, sockflags | O_NONBLOCK);
#endif

  memset(&ph->sockaddr, 0, sizeof(ph->sockaddr));
  ph->sockaddr.sin_family      = AF_INET;
  ph->sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  ph->sockaddr.sin_port        = htons(port);

  if (bind(ph->sockfd, (struct sockaddr *)&ph->sockaddr, sizeof(sockaddr)) < 0) {
    perror("bind: ");
    free(ph);
    return NULL;
  }

  return ph;
}


int getint32(char *bufptr) {
  int tmp=0;
  memcpy(&tmp, bufptr, sizeof(int));
  return tmp;
} 

float getfloat32(char *bufptr) {
  float tmp=0;
  memcpy(&tmp, bufptr, sizeof(float));
  return tmp;
} 


// ------------------------------------------------------------------------
static int mobile_listener_poll(void *voidhandle,
                        float &tx, float &ty, float &tz,
                        float &rx, float &ry, float &rz,
                        int &padaction, int &upid,
                        int &touchcnt, int *touchid,
                        float *padx, float *pady,
                        int &buttons, int &packtype, JString &incomingIP,
                        JString &currentNick, int &listenerPort, 
                        float &tranScal, float &rotScal, float &zoomScal,
                        JString &commandToSend) {
  mobilehandle *ph = (mobilehandle *) voidhandle;
  int offset = 0;

  memset(ph->buffer, 0, sizeof(ph->buffer));
#if defined(_MSC_VER)
  int fromlen=sizeof(ph->sockaddr);
#else
  socklen_t fromlen=sizeof(ph->sockaddr);
#endif
  int packlen=0;

  packlen=recvfrom(ph->sockfd, ph->buffer, sizeof(ph->buffer), 0, (struct sockaddr *)&ph->sockaddr, &fromlen);
  /* no packets read */
  if (packlen < 1) {
    return 0;
  }

  /* we now have info.  Decode it */
//  if (((int*)ph->buffer)[0] != 1) {
  if (getint32(ph->buffer) != 1) {
    swap4_aligned(ph->buffer, sizeof(ph->buffer) / 4);
  }
//  if (((int*)ph->buffer)[0] != 1) {
  if (getint32(ph->buffer) != 1) {
    printf("Received unrecognized mobile packet...\n");
    return 0;
  }

//  int endianism  = ((int*)ph->buffer)[offset++];     /* endianism.  Should be 1  */
//  int apiversion = ((int*)ph->buffer)[offset++];     /* API version */
  int endianism  = getint32(ph->buffer + offset*sizeof(int));
  offset++;

  int apiversion = getint32(ph->buffer + offset*sizeof(int));     /* API version */
  offset++;

  /* drop old format packets, or packets with incorrect protocol,   */
  /* corruption, or that aren't formatted correctly for some reason */
  if (endianism != 1 || (apiversion < 7 || apiversion > CURRENTAPIVERSION)) {
    msgWarn << "Dropped incoming mobile input packet from "
            << inet_ntoa((ph->sockaddr).sin_addr)
            << ", version: " << apiversion << sendmsg;
    return 0;
  }

  // there are now 16 bytes of data for the nickname
  char nickName[17];
  memcpy(nickName, ph->buffer+(offset*sizeof(int)), 16);  // this might not be null terminated
  nickName[16] = 0;   // so we'll put a null in the last element of the char*
  currentNick = nickName;
//fprintf(stderr, "currentNick is %s\n", (const char *)currentNick);
  offset += 4;

  if (apiversion >= 9) {
//    listenerPort = ((int*)ph->buffer)[offset++];     /* listener port on the client*/
//    rotScal = ((float*)ph->buffer)[offset++];     /* scale factor for rotate */
//    zoomScal = ((float*)ph->buffer)[offset++];     /* scale factor for zoom */
//    tranScal = ((float*)ph->buffer)[offset++];     /* scale factor for translate*/
    listenerPort = getint32(ph->buffer + offset*sizeof(float)); // listener port on the client
    offset++;
    rotScal = getfloat32(ph->buffer + offset*sizeof(float)); // scale factor for rotate
    offset++;
    zoomScal = getfloat32(ph->buffer + offset*sizeof(float)); // scale factor for zoom
    offset++;
    tranScal = getfloat32(ph->buffer + offset*sizeof(float)); // scale factor for translate
    offset++;
  } else {
    listenerPort = 4141;  /* default */
  }

  packtype   = ((int*)ph->buffer)[offset++];     /* payload description */
//  if (packtype == PACKET_HEARTBEAT) { fprintf(stderr,"{HB}"); }
  ph->buttons    = ((int*)ph->buffer)[offset++];     /* button state */
  ph->seqnum     = ((int*)ph->buffer)[offset++];     /* sequence number */


  buttons = ph->buttons;
  incomingIP = inet_ntoa((ph->sockaddr).sin_addr);

  // at this point, lets check to see if we have a command that needs
  // to be send to the script side.
  if (packtype == EVENT_COMMAND) {
     // XXX extend to allow data parameters to be retrieved from client.
     // 'buttons' and 'ph->seqnum' have been set.
     // 'buttons' stores the type of message that it is
     // and seqnum just stores the sequence number.  No big deal
     // now, let's read in any command parameters that have been sent
     int msgSize = ((int*)ph->buffer)[offset++];     /* msg length */

//fprintf(stderr, "packtype: %d, buttons: %d, seq: %d, msg size is %d\n", 
//                      packtype, buttons, ph->seqnum, msgSize);
     if (msgSize > 0) {
        char *tmpmsg = new char[msgSize+1];
        memcpy(tmpmsg, ph->buffer+(offset*sizeof(int)), msgSize);  
        tmpmsg[msgSize] = 0;           // can't assume it was null terminated

        commandToSend = tmpmsg; 
        delete [] tmpmsg;
     } else {
        commandToSend = ""; 
     }

     return 1;
  }


  padaction = EVENT_NON_TOUCH;

  // check to see if we need to go farther
  //if (packtype != PACKET_ORIENT && packtype != PACKET_TOUCH) {
  //    return 1;
  //  }

  // clear previous state from handle before decoding incoming packet
  ph->rx = 0;
  ph->ry = 0;
  ph->rz = 0;
  ph->tx = 0;
  ph->ty = 0;
  ph->tz = 0;
  ph->padaction = EVENT_NON_TOUCH;
  ph->upid = 0;
  memset(ph->touchid, 0, sizeof(ph->touchid));
  memset(ph->padx, 0, sizeof(ph->padx));
  memset(ph->pady, 0, sizeof(ph->pady));
  memset(ph->rotmatrix, 0, sizeof(9*sizeof(float)));

  // decode incoming packet based on packet type
  int i;

  switch (packtype) {
    case PACKET_ORIENT:
      // Android sensor/orientation packet
      // r[0]: Azimuth, rotation around the Z axis (0<=azimuth<360).
      //       0 = North, 90 = East, 180 = South, 270 = West
      // r[1]: Pitch, rotation around X axis (-180<=pitch<=180),
      //       with positive values when the z-axis moves toward the y-axis.
      // r[2]: Roll, rotation around Y axis (-90<=roll<=90),
      //       with positive values when the z-axis moves toward the x-axis.
      ph->rz         = ((float*)ph->buffer)[offset  ]; // orientation 0
      ph->rx         = ((float*)ph->buffer)[offset+1]; // orientation 1
      ph->ry         = ((float*)ph->buffer)[offset+2]; // orientation 2
      ph->tx         = ((float*)ph->buffer)[offset+3]; // accel 0
      ph->ty         = ((float*)ph->buffer)[offset+4]; // accel 1
      ph->tz         = ((float*)ph->buffer)[offset+5]; // accel 2

      /* 3x3 rotation matrix stored as 9 floats */
      for (i=0; i<9; i++)
        ph->rotmatrix[i] = ((float*)ph->buffer)[offset+6+i];
      break;

    case PACKET_TOUCH:  case PACKET_HEARTBEAT:
      float xdpi = ((float*)ph->buffer)[offset];    // X dots-per-inch
      float ydpi = ((float*)ph->buffer)[offset+1];  // Y dots-per-inch
//      int xsz    = ((int*)ph->buffer)[11];        // screen size in pixels
//      int ysz    = ((int*)ph->buffer)[12];        // screen size in pixels
      float xinvdpi = 1.0f / xdpi;
      float yinvdpi = 1.0f / ydpi;

      // For single touch, Actions are basically:  0:down, 2: move, 1: up.
      // for multi touch, the actions can indicate, by masking, which pointer
      // is being manipulated
      ph->padaction = ((int*) ph->buffer)[offset+4]; // action
      ph->upid      = ((int*) ph->buffer)[offset+5]; // UP, pointer id

      if (ph->padaction == 1) {
         ph->touchcnt  = touchcnt = 0;
      } else {
        ph->touchcnt  = ((int*) ph->buffer)[offset+6]; // number of touches
        touchcnt = ph->touchcnt;

        for (int i=0; i<ph->touchcnt; i++) {
          float px, py;
#if 0
          // not currently used
          int ptrid;
          ptrid = ((int*) ph->buffer)[offset+7+3*i];   // pointer id
#endif
          px  = ((float*) ph->buffer)[offset+8+3*i];   // X pixel
          py  = ((float*) ph->buffer)[offset+9+3*i];   // Y pixel

//          printf("PID:%2d, X:%4.3f, Y:%4.3f, ", ptrid, px, py);

           // scale coords to be in inches rather than pixels
           ph->padx[i] = px * xinvdpi;
           ph->pady[i] = py * yinvdpi;
         }
      }
//      printf("\n");

      break;
  } // end switch (packtype)


  if (packtype == PACKET_ORIENT) {
    rx = -ph->rx;
    ry = -(ph->rz-180); // Renormalize Android X angle from 0:360deg to -180:180
    rz =  ph->ry;
    tx = 0.0;
    ty = 0.0;
    tz = 0.0;
  }

  if (packtype == PACKET_TOUCH) {
    padaction = ph->padaction;
    upid = ph->upid;

    for (int i=0; i<ph->touchcnt; i++) {
      padx[i] = ph->padx[i];
      pady[i] = ph->pady[i];
    }
  }

#if 1
  // get absolute values of axis forces for use in
  // null region processing and min/max comparison tests
  float t_null_region = 0.01f;
  float r_null_region = 10.0f;
  float atx = fabsf(tx);
  float aty = fabsf(ty);
  float atz = fabsf(tz);
  float arx = fabsf(rx);
  float ary = fabsf(ry);
  float arz = fabsf(rz);

  // perform null region processing
  if (atx > t_null_region) {
    tx = ((tx > 0) ? (tx - t_null_region) : (tx + t_null_region));
  } else {
    tx = 0;
  }
  if (aty > t_null_region) {
    ty = ((ty > 0) ? (ty - t_null_region) : (ty + t_null_region));
  } else {
    ty = 0;
  }
  if (atz > t_null_region) {
    tz = ((tz > 0) ? (tz - t_null_region) : (tz + t_null_region));
  } else {
    tz = 0;
  }
  if (arx > r_null_region) {
    rx = ((rx > 0) ? (rx - r_null_region) : (rx + r_null_region));
  } else {
    rx = 0;
  }
  if (ary > r_null_region) {
    ry = ((ry > 0) ? (ry - r_null_region) : (ry + r_null_region));
  } else {
    ry = 0;
  }
  if (arz > r_null_region) {
    rz = ((rz > 0) ? (rz - r_null_region) : (rz + r_null_region));
  } else {
    rz = 0;
  }
#endif

  return 1;
} // end of mobile_listener_poll


// ------------------------------------------------------------------------

static int mobile_listener_destroy(void *voidhandle) {
  mobilehandle *ph = (mobilehandle *) voidhandle;

#if defined(_MSC_VER)
  closesocket(ph->sockfd); /* close the socket */
#else
  close(ph->sockfd); /* close the socket */
#endif
  free(ph);

  // all of our clients are, by definition, gone too
  return 0;
}

// ------------------------------------------------------------------------

// constructor
Mobile::Mobile(VMDApp *vmdapp) : UIObject(vmdapp) {
  mobile = NULL;
  port = 3141; // default UDP port to use

  packtimer = wkf_timer_create();
  wkf_timer_start(packtimer);

  statustimer = wkf_timer_create();
  wkf_timer_start(statustimer);

  // how often should we send the status to the clients
  statusSendSeconds = 5.0f;

  touchinprogress = 0;
  touchcount = 0;
  touchmode = ROTATE;
  touchscale = 1.0;
  touchscalestartdist = 0;
  touchrotstartangle = 0;
  touchdeltaX = 0;
  touchdeltaY = 0;
  buttonDown = 0;

  tranScaling = 1.0;
  rotScaling = 1.0;
  zoomScaling = 1.0;

  reset();
}


// ------------------------------------------------------------------------
Mobile::~Mobile(void) {
  wkf_timer_destroy(packtimer);
  wkf_timer_destroy(statustimer);

  for (int i=0; i<clientNick.num();i++)
  {
    delete clientNick[i];
    delete clientIP[i];
  }

  // clean up the client list
//  while (clientList.num() > 0) {
//     MobileClientList *ptr = clientList.pop();
//     delete ptr;
//  }

  if (mobile != NULL)
    mobile_listener_destroy(mobile);
}


// ------------------------------------------------------------------------
int send_dgram(const char *host_addr, int port, const unsigned char *buf, 
                                                        int buflen) {
  struct sockaddr_in addr;
  int sockfd;

//  printf("sending dgram of length %d\n", buflen);

  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    return -1;
  } 

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(host_addr);

#if defined(_MSC_VER)
  sendto(sockfd, (const char *) buf, buflen, 0, (struct sockaddr *)&addr, sizeof(addr));
  closesocket(sockfd);
#else
  sendto(sockfd, buf, buflen, 0, (struct sockaddr *)&addr, sizeof(addr));
  close(sockfd);
#endif 

  return 0;
}                     


// ------------------------------------------------------------------------
void Mobile::sendStatus(const int eventType) {
  prepareSendBuffer(eventType);
//    int port = 4141;
  // loop over connected clients specifically, and send them the status
  for (int i=0; i< clientIP.num(); i++) {
    // we need to insert whether or not this specific user is active
    // *(int*)(statusSendBuffer + 16) = (clientActive[i] ? 1 : 0);    // Is this nick active?
    int active = (clientActive[i] ? 1 : 0);    // Is this nick active?
    memcpy(statusSendBuffer + 16, &active, sizeof(int));

//  fprintf(stderr, "Sending '%s': %d a message.\n", (const char*)*(clientIP[i]), clientListenerPort[i]);
    send_dgram((const char *)*(clientIP[i]), clientListenerPort[i], 
               statusSendBuffer, statusSendBufferLength);
  }

  // broadcast the same to everyone that might be listening
  // need to determine how widely we want to reasonably broadcast. For 
  // our local case, we are on the ks subnet, but wireless devices aren't.
  // not likely we should spam every device in illinois.edu, though.
  // so, we have to be more intelligent about it.

  // now that we've done everything, reset the timer.
  wkf_timer_start(statustimer);
}


// ------------------------------------------------------------------------
void Mobile::checkAndSendStatus() {
  if (wkf_timer_timenow(statustimer) > statusSendSeconds) {
     sendStatus(SEND_HEARTBEAT);
  }
}



// ------------------------------------------------------------------------
bool Mobile::isInControl(JString* nick, JString* ip, const int port, const int packtype) {
//   fprintf(stderr, "isInControl.start: %s, %s\n", (const char*)*nick, (const char *)*ip);
  int i;
  for (i=0; i < clientNick.num(); i++) {
    if (*nick == *(clientNick[i]) && *ip == *(clientIP[i])) {
      // XXX update the timer here?
      break;
    }
  }

  if (i < clientNick.num()) { // we found them!
    // was this a disconnect?
    if (packtype == PACKET_DISCONNECT) {
      removeClient(i);
      return false;
    }
    return clientActive[i];
  } else {
    JString *tmpNick, *tmpIp;
    tmpNick = new JString(*nick);
    tmpIp = new JString(*ip);
//   fprintf(stderr, "isInControl: Adding %s, %s\n", (const char *)*tmpNick, (const char *)*tmpIp);
    // we didn't find this particular IP/nick combination.  Let's add it.
    if (clientNick.num() == 0) {  // there weren't any before
      addNewClient(tmpNick, tmpIp, port, true);  // make this client in control
      return true;
    } else {
      addNewClient(tmpNick, tmpIp, port, false);  // just a new client.. not in control
      return false;
    }
  }

  return false; // won't ever get here
}


// ------------------------------------------------------------------------
/////////////////////// virtual routines for UI init/display  /////////////
   
// reset the Mobile to original settings
void Mobile::reset(void) {
  // set the default motion mode and initialize button state
  move_mode(OFF);

  // set the maximum animate stride allowed to 20 by default
  set_max_stride(20);

  // set the default translation and rotation increments
  // these really need to be made user modifiable at runtime
  transInc = 1.0f / 25000.0f;
    rotInc = 1.0f /   200.0f;
  scaleInc = 1.0f / 25000.0f;
   animInc = 1.0f /     1.0f;
}


// ------------------------------------------------------------------------
// update the display due to a command being executed.  Return whether
// any action was taken on this command.
// Arguments are the command type, command object, and the 
// success of the command (T or F).
int Mobile::act_on_command(int type, Command *cmd) {
  return FALSE; // we don't take any commands presently
}


// ------------------------------------------------------------------------
// check for an event, and queue it if found.  Return TRUE if an event
// was generated.
int Mobile::check_event(void) {
  float tx, ty, tz, rx, ry, rz;
  int touchid[16];
  float padx[16], pady[16];
  
  int padaction, upid, buttons, touchcnt;
  int buttonchanged;
  int win_event=FALSE;
  int packtype=0;
  JString incIP, nick, commandToSend;
//  bool inControl = false;

  int clientPort=0;

  // for use in UserKeyEvent() calls
//  DisplayDevice::EventCodes keydev=DisplayDevice::WIN_KBD;

  // if enough time has passed, let's send a heartbeat to all connected
  // clients
  checkAndSendStatus();

  // explicitly initialize event state variables
  rx=ry=rz=tx=ty=tz=0.0f;
  buttons=padaction=upid=0;
  memset(touchid, 0, sizeof(touchid));
  memset(padx, 0, sizeof(padx));
  memset(pady, 0, sizeof(pady));
  touchcnt=0;

  // process as many events as we can to prevent a packet backlog 
  while (moveMode != OFF && 
         mobile_listener_poll(mobile, rx, ry, rz, tx, ty, tz, padaction, upid,
                              touchcnt, touchid, padx, pady, buttons, packtype,
                              incIP, nick, clientPort, tranScaling, rotScaling,
                              zoomScaling, commandToSend)) {
//fprintf(stderr, "inside while. %s, %s\n", (const char *)nick, (const char *)incIP);
    win_event = TRUE;

    // is this a command?  If so, we need to send it on to the script
    // side and let them deal with it.
    if (packtype == EVENT_COMMAND) {
       // the 'buttons' variable has the specific command stored in it
       //   incIP and nick are important, too
       char strTmp[11];
       sprintf(strTmp, "%d",buttons);
       JString jstr = "{" + nick + "} {" + incIP + "} {" + strTmp + 
                      "} {" + commandToSend + "}";
//       nick + " " + incIP + " " + strTmp;
//fprintf(stderr, "running %s\n", (const char *)jstr);
       runcommand(new MobileDeviceCommandEvent(jstr));
       break;
    }

    // let's figure out who this is, and whether or not they are in
    // control
    if (isInControl(&nick, &incIP, clientPort, packtype)) {
      DisplayDevice::EventCodes keydev=DisplayDevice::WIN_KBD;
//      inControl = true;

      // find which buttons changed state
      buttonchanged = buttons ^ buttonDown; 

      // XXX change hardcoded numbers and support >3 buttons
      if (buttonchanged) {
         // for normal buttons, we want the down event
        if (buttonchanged == (1<<0) && (buttonchanged & buttons)) {
           runcommand(new UserKeyEvent(keydev, '0', (int) DisplayDevice::AUX));
        }
        if (buttonchanged == (1<<1) && (buttonchanged & buttons)) {
           runcommand(new UserKeyEvent(keydev, '1', (int) DisplayDevice::AUX));
        }
        if (buttonchanged == (1<<2) && (buttonchanged & buttons)) {
           runcommand(new UserKeyEvent(keydev, '2', (int) DisplayDevice::AUX));
        }
        if (buttonchanged == (1<<3) && (buttonchanged & buttons)) {
           runcommand(new UserKeyEvent(keydev, '3', (int) DisplayDevice::AUX));
        }
      } // end if on buttonchanged

#if 0
      printf("Touchpad action: %d upid %d", padaction, upid);
      for (int i=0; i<touchcnt; i++) {
        printf("ID[%d] x: %.2f y: %.2f ",
               i, padx[i], pady[i]);
      }
      printf("\n");
#endif

      if (padaction != EVENT_NON_TOUCH) {
        // detect end of a touch event
        if (touchcnt < touchcount || 
             padaction == EVENT_TOUCH_UP || padaction == EVENT_TOUCH_SOMEUP) {
//           fprintf(stderr,"<(a:%d,b:%d)", touchcnt, touchcount);
          touchinprogress = 0;
          touchmode = ROTATE;
          touchcount = 0;
          touchstartX = 0;
          touchstartY = 0;
          touchdeltaX = 0;
          touchdeltaY = 0;
          touchscale = 1.0;
          touchscalestartdist = 0;
          touchrotstartangle = 0;
        }
    
        // detect a touch starting event 
        if (touchcnt > touchcount ||
             padaction == EVENT_TOUCH_DOWN ||
             padaction == EVENT_TOUCH_SOMEDOWN) {
//           fprintf(stderr,">(a:%d,b:%d)", touchcnt, touchcount);
          touchcount = touchcnt;
          touchstartX = 0;
          touchstartY = 0;
          touchdeltaX = 0;
          touchdeltaY = 0;
          touchscale = 1.0;
          touchscalestartdist = 0;
          touchrotstartangle = 0;

          // printf("Touchcount: %d\n", touchcount);
          if (touchcount == 1) {
            // printf("Start rotate..\n");
            touchinprogress=1;
            touchmode = ROTATE;
            touchstartX = padx[0];
            touchstartY = pady[0];
          } else if (touchcount == 2) {
            touchinprogress=1;
            touchstartX = (padx[0] + padx[1]) * 0.5f;
            touchstartY = (pady[0] + pady[1]) * 0.5f;

            float dx = padx[1] - padx[0];
            float dy = pady[1] - pady[0];
            touchscalestartdist = sqrtf(dx*dx + dy*dy) + 0.00001f;
            if (touchscalestartdist > 0.65f) { 
              touchrotstartangle = float(atan2(dx, -dy) + VMD_PI);
              // printf("Start scale.. dist: %.2f  angle: %.2f\n", touchscalestartdist, touchrotstartangle);
              touchmode = SCALEROTZ;
            } else {
              // printf("Start translate.. dist(%.2f)\n", touchscalestartdist);
              touchmode = TRANSLATE;
            }
          }
        }

        if (touchinprogress && padaction == EVENT_TOUCH_MOVE) {
          if (touchmode == ROTATE) {
            touchdeltaX = padx[0] - touchstartX;
            touchdeltaY = pady[0] - touchstartY;
          } else if (touchmode == SCALEROTZ) {
            // only move the structure if we're in move mode,
            // in animate mode we do nothing...
            if (moveMode == MOVE) {
              float dx = padx[1] - padx[0];
              float dy = pady[1] - pady[0];
              float dist = sqrtf(dx*dx + dy*dy);
     
              // Only scale if the scale changes by at least 1%
              float newscale = (dist / touchscalestartdist) / touchscale;
              if (fabsf(newscale - 1.0f) > 0.01f) {
                touchscale *= newscale;
                app->scene_scale_by((newscale - 1.0f) * zoomScaling + 1.0f);
              }

              // Only rotate if the angle update is large enough to make
              // it worthwhile, otherwise we get visible "jitter" from noise
              // in the touchpad coordinates.  Currently, we only rotate if
              // the rotation magnitude is greater than a quarter-degree
              float newrotangle = float(atan2(dx, -dy) + VMD_PI);
              float rotby = float((newrotangle-touchrotstartangle)*180.0f/VMD_PI);
              if (fabsf(rotby) > 0.25f) {
                app->scene_rotate_by(-rotScaling*rotby, 'z');
                touchrotstartangle=newrotangle;
              }
            }
          } else if (touchmode == TRANSLATE) {
            touchdeltaX = ((padx[0]+padx[1])*0.5f) - touchstartX;
            touchdeltaY = ((pady[0]+pady[1])*0.5f) - touchstartY;
          }
        }
      }

      // update button status for next time through
      buttonDown = buttons;
    }  // end of isInControl

    // restart last-packet timer
    wkf_timer_start(packtimer);
  }           // end while (moveMode != OFF && mobile_listener_poll())

  // XXX this next check really needs to be done on a per-client basis and 
  // non responsive clients should be kicked out
  // check for dropped packets or mobile shutdown and 
  // halt any ongoing events if we haven't heard from the
  // client in over 1 second.  
  if (!win_event && wkf_timer_timenow(packtimer) > 3.0) {
    touchinprogress = 0;
    touchmode = ROTATE;
    touchcount = 0;
    touchstartX = 0;
    touchstartY = 0;
    touchdeltaX = 0;
    touchdeltaY = 0;
    touchscalestartdist = 0;
    touchrotstartangle = 0;
  }

  if (touchinprogress) {
    if (moveMode == MOVE) {
      if (touchmode == ROTATE) {
        //         fprintf(stderr,"+");
        // Motion in Android "X" rotates around VMD Y axis...
        app->scene_rotate_by(touchdeltaY*rotScaling*0.5f, 'x');
        app->scene_rotate_by(touchdeltaX*rotScaling*0.5f, 'y');
      } else if (touchmode == TRANSLATE) {
        app->scene_translate_by(touchdeltaX*tranScaling*0.005f, -touchdeltaY*tranScaling*0.005f, 0.0f);
      }
    } else if (moveMode == ANIMATE) {
      if (fabsf(touchdeltaX) > 0.25f) {
#if 0
        // exponential input scaling
        float speed = fabsf(expf(fabsf((fabsf(touchdeltaX) * animInc) / 1.7f))) - 1.0f;
#else
        // linear input scaling
        float speed = fabsf(touchdeltaX) * animInc;
#endif

        if (speed > 0) {
          if (speed < 1.0)
            app->animation_set_speed(speed);
          else
            app->animation_set_speed(1.0f);

          int stride = 1;
          if (fabs(speed - 1.0) > (double) maxstride)
            stride = maxstride;
          else
            stride = 1 + (int) fabs(speed-1.0);
          if (stride < 1)
            stride = 1;
          app->animation_set_stride(stride);

          if (touchdeltaX > 0) {
            app->animation_set_dir(Animation::ANIM_FORWARD1);
          } else {
            app->animation_set_dir(Animation::ANIM_REVERSE1);
          }
        } else {
          app->animation_set_dir(Animation::ANIM_PAUSE);
          app->animation_set_speed(1.0f);
        }
      } else {
        app->animation_set_dir(Animation::ANIM_PAUSE);
        app->animation_set_speed(1.0f);
      }
    }
  } 
//  else { 
//     if (!inControl) fprintf(stderr,"-");
//     if (!touchinprogress) fprintf(stderr,"|");
//  } 

  if (win_event) {
    return TRUE;
  } else {
    return FALSE; // no events to report
  }
} // end of Mobile::check_event()



// ------------------------------------------------------------------------
///////////// public routines for use by text commands etc

const char* Mobile::get_mode_str(MoveMode mm) {
  const char* modestr;

  switch (mm) {
    default:
    case OFF:         modestr = "off";        break;
    case MOVE:        modestr = "move";       break;
    case ANIMATE:     modestr = "animate";    break;
    case TRACKER:     modestr = "tracker";    break;
    case USER:        modestr = "user";       break;
  }

  return modestr;
}

// ------------------------------------------------------------------------
int Mobile::get_port () {
  return port;
}

// ------------------------------------------------------------------------
int Mobile::get_APIsupported () {
  return CURRENTAPIVERSION;
}

// ------------------------------------------------------------------------
int Mobile::get_move_mode () {
  return moveMode;
}

// ------------------------------------------------------------------------
void  Mobile::get_client_list (ResizeArray <JString*>* &nick, 
                         ResizeArray <JString*>* &ip, ResizeArray <bool>* &active)
{
  nick = &clientNick;
  ip = &clientIP;
  active = &clientActive;
}

// ------------------------------------------------------------------------
int Mobile::sendMsgToClient(const char *nick, const char *ip, 
                            const char *msgType, const char *msg)
{
//   fprintf(stderr, "Sending %s (%s) msgtype %s, msg '%s'\n",nick,ip,msgType,msg);
   // find the user with the given nick and ip
  bool found = false;
  int i;
  for (i=0; i<clientNick.num();i++)
  {
    if (*(clientNick[i]) == nick) {
      // we've found the right nick.  Now let's check the IP
      if (*(clientIP[i]) == ip) {
        found = true;
        break;
      }
    }
  }
   
  if (found) {
    int msgTypeAsInt;
    if (EOF == sscanf(msgType, "%d", &msgTypeAsInt)) {
      return false;
    }

    sendMsgToIp(clientActive[i],   // 1 if active, else 0
                msgTypeAsInt,            // integer msg type
                msg,            // msg contents
                (const char *)*(clientIP[i]),       // string IP address
                clientListenerPort[i]);  // port to send to
    return true;
  } else {
    return false;
  }
}  // end of Mobile::sendMsgToClient


// ------------------------------------------------------------------------
void Mobile::sendMsgToIp(const bool isActive,
                   const int msgTypeAsInt,
                   const char *msg,
                   const char *ip,
                   const int port)
{
  // let's send them the msg
  prepareSendBuffer(SEND_MESSAGE);
  // we need to insert whether or not this specific user is active
  // *(int*)(statusSendBuffer + 16) = (isActive ? 1 : 0);    // Is this nick active?
  int active = (isActive ? 1 : 0);    // Is this nick active?
  memcpy(statusSendBuffer + 16, &active, sizeof(int));

  // pack the message. Right now it is length long.  We need to add to it.
  int length = statusSendBufferLength;

  // *(int*)(statusSendBuffer + length) = msgTypeAsInt;
  memcpy(statusSendBuffer + length, &msgTypeAsInt, sizeof(int));
  length += sizeof(int);

  int msgLength = strlen(msg);
//  printf("msg is '%s', msglength is %ld\n", msg, msgLength);
  // *(int*)(statusSendBuffer + length) = msgLength;
  memcpy((statusSendBuffer + length), &msgLength, sizeof(int));
  length += sizeof(int);

  memcpy((statusSendBuffer + length), msg, msgLength);
  length += msgLength;

  send_dgram(ip, port, statusSendBuffer, length);

}


// ------------------------------------------------------------------------
int Mobile::set_activeClient(const char *nick, const char *ip) {
  // fprintf(stderr, "in set_activeClient.  nick: %s, ip: %s\n", nick, ip);
  // find the user with the given nick and ip
  bool found = false;
  int i;
  for (i=0; i<clientNick.num();i++) {
    if (*(clientNick[i]) == nick) {
      // we've found the right nick.  Now let's check the IP
      if (*(clientIP[i]) == ip) {
        found = true;
        break;
      }
    }
  }
   
  if (found) { 
    // First, run through clientActive and turn everyone off.
    for (int j=0; j<clientActive.num();j++) {
      clientActive[j] = false;
    }

    // turn off any movements that might have been going on
    touchinprogress = 0;
    touchmode = ROTATE;
    touchcount = 0;
    touchstartX = 0;
    touchstartY = 0;
    touchdeltaX = 0;
    touchdeltaY = 0;
    touchscalestartdist = 0;
    touchrotstartangle = 0;

    // set this one client to active.
    clientActive[i] = true;

    sendStatus(SEND_SETACTIVECLIENT);
    return true;
  } else {
    return false;
  }
}  // end set active client


// ------------------------------------------------------------------------
void Mobile::get_tracker_status(float &tx, float &ty, float &tz,
                                float &rx, float &ry, float &rz, 
                                int &buttons) {
  tx =  trtx * transInc;
  ty =  trty * transInc;
  tz = -trtz * transInc;
  rx =  trrx * rotInc;
  ry =  trry * rotInc;
  rz = -trrz * rotInc;
  buttons = trbuttons;
}


// ------------------------------------------------------------------------
// set the Mobile move mode to the given state; return success
int Mobile::move_mode(MoveMode mm) {
  // change the mode now
  moveMode = mm;

  if (moveMode != OFF && !mobile) {
    mobile = mobile_listener_create(port);
    if (mobile == NULL) {
      msgErr << "Failed to open mobile port " << port 
             << ", move mode disabled" << sendmsg;
      moveMode = OFF;
    } else {
      msgInfo << "Opened mobile port " << port << sendmsg;
    }
  }

  // let's destroy the port binding since they've turned moving off
  if (moveMode == OFF && mobile) {
    mobile_listener_destroy(mobile);
    mobile = 0;
    removeAllClients();
  }

  // clear out any remaining tracker event data if we're not in that mode
  if (moveMode != TRACKER) {
    trtx=trty=trtz=trrx=trry=trrz=0.0f; 
    trbuttons=0;
  }
  // fprintf(stderr,"Triggering command due to mode change\n");
  runcommand(new MobileStateChangedEvent());
  sendStatus(SEND_SETMODE);

  return TRUE; // report success
} // end of Mobile::move_mode


// ------------------------------------------------------------------------
// set the incoming UDP port (closing the old one if needed)
int Mobile::network_port(int newport) {
  if (mobile != NULL) {
    mobile_listener_destroy(mobile);
    removeAllClients();
  }

  if (moveMode != OFF) {
    mobile = mobile_listener_create(newport);
    if (mobile == NULL) {
      msgErr << "Failed to open mobile port " << newport 
               << ", move mode disabled" << sendmsg;
      moveMode = OFF;
    } else {
      port = newport;
//fprintf(stderr,"Triggering command due to port change\n");
      msgInfo << "Opened mobile port " << port << sendmsg;
    }
    runcommand(new MobileStateChangedEvent());
  } else {
    port = newport;
  }

  return TRUE; // report success
} // end of Mobile::network_port


// ------------------------------------------------------------------------
int Mobile::addNewClient(JString* nick,  JString* ip, const int port, const bool active) {
//  fprintf(stderr, "Adding %s, %s, %d\n", (const char*)*nick, (const char*)*ip, active);
  clientNick.append(nick);
  clientIP.append(ip);
  clientListenerPort.append(port);
  clientActive.append(active);

//fprintf(stderr,"Triggering command due to addNewClient\n");
  runcommand(new MobileStateChangedEvent());
  sendStatus(SEND_ADDCLIENT);
  return 0;
} // end of Mobile::addNewClient


// ------------------------------------------------------------------------
void Mobile::removeAllClients() {
  while (clientNick.num() > 0) {
    removeClient(0);
  }
}


// ------------------------------------------------------------------------
int Mobile::removeClient(const int num) {
  delete clientNick[num];
  clientNick.remove(num);
  delete clientIP[num];
  clientIP.remove(num);
  clientActive.remove(num);
  clientListenerPort.remove(num);

  // let's see how many active clients are left?
  int iCount;
  for (iCount=0; iCount<clientActive.num(); iCount++) {
    if (clientActive[iCount]) {
      break;
    }
  }

  // did we make it all the way through the client list without
  // finding anyone that is active?  If so (and there are actually
  // still clients) set the first one to active
  if (iCount == clientActive.num() && clientActive.num() > 0) {
    clientActive[0] = true;
  }

  // fprintf(stderr,"Triggering command due to removeClient\n");
  runcommand(new MobileStateChangedEvent());
  sendStatus(SEND_REMOVECLIENT);

  return 0;
}



