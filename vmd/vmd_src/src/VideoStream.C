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
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.34 $         $Date: 2019/01/17 21:21:02 $
*
***************************************************************************
* DESCRIPTION:
*   VMD interface for video streaming
*
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "WKFUtils.h"
#include "Inform.h"
#include "VideoStream.h"
#include "vmdsock.h"
#include "VMDApp.h"
#include "DisplayDevice.h"
#include "TextEvent.h"      // event handling callbacks for keypresses


// XXX enable simulated codec
#if !defined(VMDNVPIPE)
#define VIDEOSTREAM_SIMENCODER 1
#endif

#if defined(VIDEOSTREAM_SIMENCODER)
typedef struct {
  int width;        
  int height;            
} simenc_handle;


static void * simenc_initialize(int width, int height, int Mbps, int tfps) {
  simenc_handle *sep = (simenc_handle *) calloc(sizeof(simenc_handle), 1);
  sep->width = width;
  sep->height = height;
  return sep;
}


static void simenc_destroy(void *voidhandle) {
  free(voidhandle);
}


static int simenc_reconfig(void *voidhandle, int width, int height, 
                           int bitrateMbps, int targetfps) {
  simenc_handle *sep = (simenc_handle *) calloc(sizeof(simenc_handle), 1);
  sep->width = width;
  sep->height = height;
  return 0;
}


static unsigned long simenc_encode_frame(void *voidhandle, 
                                         const unsigned char *rgba,
                                         int pitch, int width, int height, 
                                         unsigned char * & compbuf, 
                                         long compbufsz, bool forceIframe) {
  long sz = pitch * height;
  if (sz > compbufsz)
    return 0;

  memcpy(compbuf, rgba, sz); // no-op passthrough compression  
  return sz;
}


static unsigned long simenc_decode_frame(void *voidhandle, 
                                         unsigned char *compbuf,
                                         long compbufsz, unsigned char *rgba,
                                         int width, int height) {
  long sz = 4 * width * height;
  if (sz > compbufsz) {
    printf("\nsimenc: sz: %ld  compbufsz: %ld\n", sz, compbufsz);
    return 0;
  }
 
  memcpy(rgba, compbuf, sz); // no-op passthrough decompression  
  return sz;
}


#endif


//
// Support for NvPipe+NVEnc hardware video encoding 
//
#if defined(VMDNVPIPE)
#include <NvPipe.h>
#include <cuda_runtime_api.h>

typedef struct {
  NvPipe_Format format;           // BGRA or RGBA format
  NvPipe_Codec codec;             // NVPIPE_H264 or NVPIPE_HEVC (*)new GPUs  
  NvPipe_Compression compression; // NVPIPE_LOSSY or NVPIPE_LOSSLESS
  float bitrateMbps;              // 5-20 MBps is reasonable
  uint32_t targetFPS;             // 20 FPS is decent
  uint32_t width;        
  uint32_t height;            
  NvPipe *encoder;
  NvPipe *decoder;
} nvpipe_handle;


static void * nvpipe_initialize(int width, int height, int Mbps, int tfps) {
  nvpipe_handle *nvp = (nvpipe_handle *) calloc(sizeof(nvpipe_handle), 1);

  // XXX NvPipe_Format enumeration is different across platforms, 
  //     even in the case that byte ordering is the same
#if defined(ARCH_SUMMIT) || defined(ARCH_OPENPOWER)
  nvp->format = NVPIPE_RGBA32; // XXX why do ppc64le and x86 differ?!?!?!?
#else
  nvp->format = NVPIPE_BGRA32; // XXX why do ppc64le and x86 differ?!?!?!?
#endif

  nvp->codec = NVPIPE_H264;
  nvp->compression = NVPIPE_LOSSY;
  nvp->bitrateMbps = Mbps;
  nvp->targetFPS = tfps;
  nvp->width = width;
  nvp->height = height;

  nvp->encoder = NvPipe_CreateEncoder(nvp->format, nvp->codec, 
                                      nvp->compression,
                                      nvp->bitrateMbps * 1000 * 1000, 
                                      nvp->targetFPS);

  nvp->decoder = NvPipe_CreateDecoder(nvp->format, nvp->codec);

  if (nvp->encoder != NULL && nvp->decoder != NULL) {
    return nvp;
  }

  // if either encoder or decoder don't init, we have to bail entirely
  if (nvp->encoder != NULL)
    NvPipe_Destroy(nvp->encoder);
  if (nvp->decoder != NULL)
    NvPipe_Destroy(nvp->decoder);

  return NULL;
}


static void nvpipe_destroy(void *voidhandle) {
  nvpipe_handle *nvp = (nvpipe_handle *) voidhandle;
  if (nvp->encoder != NULL) {
    NvPipe_Destroy(nvp->encoder);
  }
  if (nvp->decoder != NULL) {
    NvPipe_Destroy(nvp->decoder);
  }
  free(nvp);
}


static int nvpipe_reconfig(void *voidhandle, int width, int height, 
                           int bitrateMbps, int targetfps) {
  nvpipe_handle *nvp = (nvpipe_handle *) voidhandle;

  if (width <= 0 || height <= 0) {
    printf("nvpipe_reconfig(): invalid resolution: %d x %d\n", width, height);
    return -1;
  }

  nvp->width = width;
  nvp->height = height;
  NvPipe_Destroy(nvp->encoder);
  nvp->encoder = NvPipe_CreateEncoder(nvp->format, nvp->codec, 
                                      nvp->compression,
                                      nvp->bitrateMbps * 1000 * 1000, 
                                      nvp->targetFPS);
  return 0;
}


static unsigned long nvpipe_encode_frame(void *voidhandle, 
                                         const unsigned char *rgba,
                                         int pitch, int width, int height, 
                                         unsigned char * & compbuf, 
                                         long compbufsz,
                                         bool forceIframe) {
  nvpipe_handle *nvp = (nvpipe_handle *) voidhandle;

#if 1
  if (width != nvp->width || height != nvp->height) {
    printf("nvpipe_encode_frame(): inconsistent resolution: %d x %d\n", width, height);
    printf("                         does not match config: %d x %d\n", nvp->width, nvp->height);
  }
#endif

  // encode the image...
  uint64_t compsz = NvPipe_Encode(nvp->encoder, rgba, pitch, compbuf, 
                                  compbufsz, width, height, forceIframe);
  if (compsz == 0) {
    printf("NVEnc: encode failed!: %s\n", NvPipe_GetError(nvp->encoder));
  }

  return compsz;
}


static unsigned long nvpipe_decode_frame(void *voidhandle, 
                                         unsigned char *compbuf,
                                         long compbufsz,
                                         unsigned char *rgba,
                                         int width, int height) {
  nvpipe_handle *nvp = (nvpipe_handle *) voidhandle;
  uint64_t r = NvPipe_Decode(nvp->decoder, compbuf, compbufsz, 
                             rgba, width, height);

  return (r == 0); // error if we get a zero size back
}

#endif


#define VS_PROTOCOL_VERSION     1
#define VS_HEADER_NUM_DATAUNION 6

typedef union {
    int   ival;
    float fval;
} eventdataunion;

typedef struct VSMsgHeader_t {
  int type;
  int len;
  int width;
  int height;

  int framecount;
  int eventtype;
  eventdataunion eventdata[VS_HEADER_NUM_DATAUNION];
} VSMsgHeader;

#define VS_HEADERSIZE       sizeof(VSMsgHeader)

static void swap4(char *data, int ndata) {
  int i;
  char *dataptr;
  char b0, b1;

  dataptr = data;
  for (i=0; i<ndata; i+=4) {
    b0 = dataptr[0];
    b1 = dataptr[1];
    dataptr[0] = dataptr[3];
    dataptr[1] = dataptr[2];
    dataptr[2] = b1;
    dataptr[3] = b0;
    dataptr += 4;
  }
}

static int vs_htonl(int h) {
  int n;
  ((char *)&n)[0] = (h >> 24) & 0x0FF;
  ((char *)&n)[1] = (h >> 16) & 0x0FF;
  ((char *)&n)[2] = (h >> 8) & 0x0FF;
  ((char *)&n)[3] = h & 0x0FF;
  return n;
}

typedef struct {
  unsigned int highest : 8;
  unsigned int high    : 8;
  unsigned int low     : 8;
  unsigned int lowest  : 8;
} netint;

static int vs_ntohl(int n) {
  int h = 0;
  netint net;
  net = *((netint *)&n);
  h |= net.highest << 24 | net.high << 16 | net.low << 8 | net.lowest;
  return h;
}

static void fill_header(VSMsgHeader *header, int type, int length,
                        int width = 0, int height = 0, int framecount = 0) {
  header->type       = vs_htonl((int)type);
  header->len        = vs_htonl(length);
  header->width      = vs_htonl(width);
  header->height     = vs_htonl(height);

  header->framecount = vs_htonl(framecount);
  header->eventtype  = 0;
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }
}


#if 0
static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                int ival0, int ival1) {
  header->type       = vs_htonl((int)type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }
  header->eventdata[0].ival = vs_htonl(ival0);
  header->eventdata[1].ival = vs_htonl(ival1);
}

static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                int ival0) {
  header->type       = vs_htonl((int)type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }
  header->eventdata[0].ival = vs_htonl(ival0);
}
#endif

static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                int ival0, int ival1, int ival2) {
  header->type       = vs_htonl((int)type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }
  header->eventdata[0].ival = vs_htonl(ival0);
  header->eventdata[1].ival = vs_htonl(ival1);
  header->eventdata[2].ival = vs_htonl(ival2);
}

static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                float fval0) {
  header->type       = vs_htonl(type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }

  // XXX treat the float/int union as a float, then as an int 
  //     for vs_htonl() handling, clean up this hack later
  header->eventdata[0].fval = fval0;
  header->eventdata[0].ival = vs_htonl(header->eventdata[0].ival);
}


static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                float fval0, int ival1) {
  header->type       = vs_htonl(type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }

  // XXX treat the float/int union as a float, then as an int 
  //     for vs_htonl() handling, clean up this hack later
  header->eventdata[0].fval = fval0;
  header->eventdata[0].ival = vs_htonl(header->eventdata[0].ival);
  header->eventdata[1].ival = vs_htonl(ival1);
}


static void fill_header_uievent(VSMsgHeader *header, int type, int eventtype, 
                                float fval0, float fval1, float fval2) {
  header->type       = vs_htonl(type);
  header->len        = 0;
  header->width      = 0;
  header->height     = 0;

  header->framecount = 0;
  header->eventtype  = vs_htonl(eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    header->eventdata[i].ival = 0;
  }

  // XXX treat the float/int union as a float, then as an int 
  //     for vs_htonl() handling, clean up this hack later
  header->eventdata[0].fval = fval0;
  header->eventdata[0].ival = vs_htonl(header->eventdata[0].ival);
  header->eventdata[1].fval = fval1;
  header->eventdata[1].ival = vs_htonl(header->eventdata[1].ival);
  header->eventdata[2].fval = fval2;
  header->eventdata[2].ival = vs_htonl(header->eventdata[2].ival);
}


static void swap_header(VSMsgHeader *header) {
  header->type       = vs_ntohl(header->type);
  header->len        = vs_ntohl(header->len);
  header->width      = vs_ntohl(header->width);
  header->height     = vs_ntohl(header->height);

  header->framecount = vs_ntohl(header->framecount);
  header->eventtype  = vs_ntohl(header->eventtype);
  for (int i=0; i<VS_HEADER_NUM_DATAUNION; i++) {
    int tmp = header->eventdata[i].ival;
    header->eventdata[i].ival = vs_ntohl(tmp);
  }
}




//
// VideoStream class methods
//

VideoStream::VideoStream(VMDApp *vmdapp) : UIObject(vmdapp) {
  ench = NULL;
  cli_socket = NULL;
  srv_socket = NULL;

  timer = wkf_timer_create();
  wkf_timer_start(timer);

  srv_get_one_ui_event = 0;

  // initialize last_xxx timers to now so we don't get warnings at startup
  double now = wkf_timer_timenow(timer);
  cli_lastmsgtime = now;
  cli_lastframe = now;
  srv_lastmsgtime = now;
  srv_lastframe = now;

  expave_fps = 0.0;
  imagesz_uncomp = 0.0;
  imagesz_comp = 0.0;
  lastmsgeventloops = 0;
  lastconsolemesg = wkf_timer_timenow(timer);

  // get video stream resolution from the active client display size
  vs_width = app->display->xSize;
  vs_height = app->display->ySize;

  // force window resize to video encoder block-multiple
  vs_width = ((vs_width + 15) / 16) * 16;
  vs_height = ((vs_height + 15) / 16) * 16;
  //  XXX can't call through VMDApp chain because we can rapidly 
  //      get into a deeply recursive call chain leading to a segfault.
//  app->display_set_size(vs_width, vs_height);
  app->display->resize_window(vs_width, vs_height);

  vs_bitrateMbps = 10;
  vs_targetFPS = 20;

#if defined(VIDEOSTREAM_SIMENCODER)
  ench = simenc_initialize(vs_width, vs_height, vs_bitrateMbps, vs_targetFPS);
#elif defined(VMDNVPIPE)
  ench = nvpipe_initialize(vs_width, vs_height, vs_bitrateMbps, vs_targetFPS);
  msgInfo << "VideoStream: codec initialized @ "
          << vs_width << "x" << vs_height << " res, " 
          << vs_targetFPS << " FPS @ " << vs_bitrateMbps << "Mbps."
          << sendmsg;
#endif

#if defined(VIDEOSTREAM_STATICBUFS)
  vs_imgbufsz = 8192 * 4320 * 4 + VS_HEADERSIZE; // max framebuffer sz
  vs_imgbuf = (unsigned char *) malloc(vs_cbufsz);

  vs_cbufsz = 8192 * 4320 * 4 + VS_HEADERSIZE; // max framebuffer sz
  vs_cbuf = (unsigned char *) malloc(vs_cbufsz);
#endif

  // clear all "pending" frame state variables
  vs_framepending = 0;
  vs_rgba_pend = NULL;
  vs_rgba_width = 0;
  vs_rgba_height = 0;

  // clear pending codec reconfig for now
  vs_codec_reconfig_pending = 0;
}


VideoStream::~VideoStream(void) {

#if defined(VIDEOSTREAM_STATICBUFS)
  if (vs_imgbuf) {
    free(vs_imgbuf); 
    vs_imgbuf = NULL;
  }

  if (vs_cbuf) {
    free(vs_cbuf); 
    vs_cbuf = NULL;
  }
#endif

  if (ench != NULL) {
#if defined(VIDEOSTREAM_SIMENCODER)
    simenc_destroy(ench);
#elif defined(VMDNVPIPE)
    nvpipe_destroy(ench);
#endif
    ench = NULL;
  }

  wkf_timer_destroy(timer);
}


//
// client-side APIs
//
int VideoStream::cli_listen(int port) {
  vmdsock_init(); // ensure socket interface is ready

  msgInfo << "VideoStream client: setting up incoming socket\n" << sendmsg;
  void *listen_socket = vmdsock_create();
  vmdsock_bind(listen_socket, port);

  msgInfo << "VideoStream client: Waiting for connection on port: " << port << sendmsg;
  vmdsock_listen(listen_socket);
  while (!cli_socket) {
    if (vmdsock_selread(listen_socket, 0) > 0) {
      cli_socket = vmdsock_accept(listen_socket);
      if (vs_recv_handshake(cli_socket)) {
        cli_socket = NULL;
      };
    }
  }

  vmdsock_destroy(listen_socket); // we are no longer listening...

  return 0;
}


int VideoStream::cli_connect(const char *hostname, int port) {
  vmdsock_init(); // ensure socket interface is ready
 
  cli_socket = vmdsock_create();
  if (cli_socket == NULL) {
    msgErr << "VideoStream client: could not create socket" << sendmsg;
    return -1;
  }
  int rc = vmdsock_connect(cli_socket, hostname, port);
  if (rc < 0) {
    msgErr << "VideoStream client: error connecting to " << hostname << " on port "<< port <<sendmsg;
    vmdsock_destroy(cli_socket);
    cli_socket = 0;
    return -1;
  }
  rc = vs_send_handshake(cli_socket);
  msgInfo << "VideoStream client: handshake return code: " << rc << sendmsg;  

  // initialize last_xxx timers to now so we don't get warnings at startup
  double now = wkf_timer_timenow(timer);
  cli_lastmsgtime = now;
  cli_lastframe = now;

  expave_fps = 0.0;
  imagesz_uncomp = 0.0;
 
  return rc;
}

int VideoStream::cli_wait_msg() {
  return -1;
}

int VideoStream::cli_disconnect() {
  if (cli_socket != NULL) {
    vs_send_disconnect(cli_socket);
    vmdsock_destroy(cli_socket);
    cli_socket = 0;
    return 0;
  }
  return -1;
}


//
// server-side APIs
//

int VideoStream::vs_encoder_reconfig() {
  int rc = -1;
#if defined(VIDEOSTREAM_SIMENCODER)
  rc = simenc_reconfig(ench, vs_width, vs_height, vs_bitrateMbps, vs_targetFPS);
#elif defined(VMDNVPIPE)
  rc = nvpipe_reconfig(ench, vs_width, vs_height, vs_bitrateMbps, vs_targetFPS);
#endif
  return rc;
}


int VideoStream::srv_listen(int port) {
  vmdsock_init(); // ensure socket interface is ready

  msgInfo << "VideoStream: setting up incoming socket\n" << sendmsg;
  void *listen_socket = vmdsock_create();
  vmdsock_bind(listen_socket, port);

  msgInfo << "VideoStream server: Waiting for connection on port: " << port << sendmsg;
  vmdsock_listen(listen_socket);
  while (!srv_socket) {
    if (vmdsock_selread(listen_socket, 0) > 0) {
      srv_socket = vmdsock_accept(listen_socket);
      if (vs_recv_handshake(srv_socket)) {
        srv_socket = NULL;
      };
    }
  }

  vmdsock_destroy(listen_socket); // we are no longer listening...

  return 0;
}


int VideoStream::srv_connect(const char *hostname, int port) {
  vmdsock_init(); // ensure socket interface is ready

  srv_socket = vmdsock_create();
  if (srv_socket == NULL) {
    msgErr << "VideoStream server: could not create socket" << sendmsg;
    return -1;
  }
  int rc = vmdsock_connect(srv_socket, hostname, port);
  if (rc < 0) {
    msgErr << "VideoStream server: error connecting to " << hostname << " on port "<< port <<sendmsg;
    vmdsock_destroy(srv_socket);
    srv_socket = 0;
    return -1;
  }
  rc = vs_send_handshake(srv_socket);
  msgInfo << "VideoStream servert: handshake return code: " << rc << sendmsg;

  // initialize last_xxx timers to now so we don't get warnings at startup
  double now = wkf_timer_timenow(timer);
  srv_lastmsgtime = now;
  srv_lastframe = now;

  return rc;
}


int VideoStream::srv_send_frame(const unsigned char *rgba, int pitch,   
                                int width, int height, int forceIFrame) {
  int rc = -1;
  if (ench) {
    long imgsz = pitch * height;
    long msgallocsz = imgsz + VS_HEADERSIZE; // original uncompressed msg size
#if defined(VIDEOSTREAM_STATICBUFS)
    unsigned char *cbuf = vs_cbuf;
#else
    unsigned char *cbuf = (unsigned char *) malloc(msgallocsz);
#endif
    unsigned char *imgbuf = cbuf + VS_HEADERSIZE; // offset past header

    unsigned long compsz=0;
#if defined(VIDEOSTREAM_SIMENCODER)
    compsz = simenc_encode_frame(ench, rgba, width * 4, width, height, 
                                 imgbuf, imgsz, false);
#elif defined(VMDNVPIPE)
    compsz = nvpipe_encode_frame(ench, rgba, width * 4, width, height, 
                                 imgbuf, imgsz, false);
#endif

    // if encoding succeeded, send it!
    if (compsz > 0) {
      // compute exponential moving average for exp(-1/10)
      imagesz_uncomp = (imagesz_uncomp * 0.90) + (imgsz * 0.10);
      imagesz_comp   = (imagesz_comp * 0.90)   + (compsz * 0.10);

      fill_header((VSMsgHeader *)cbuf, VS_IMAGE, compsz, width, height);

      long msgsz = compsz + VS_HEADERSIZE; // final actual compressed msg size
      rc = (vs_writen(srv_socket, (const char*) cbuf, msgsz) != msgsz);
    }
 
#if !defined(VIDEOSTREAM_STATICBUFS)
    free(cbuf);
#endif
  }

#if 1
  printf("NVEnc: %dx%d  raw:%.1fMB  comp:%.1fMB  ratio:%.1f:1  FPS:%.1f   \r",
          width, height, 
          imagesz_uncomp/(1024.0*1024.0), imagesz_comp/(1024.0*1024.0),
          imagesz_uncomp/imagesz_comp, expave_fps);
          fflush(stdout);
#endif
  
  // Throttle video throughput so we don't hammer the network.
  // Don't allow our actual FPS to exceed the user-defined setpoint.
  // If we arrive too early at this point, we stall the outbound video
  // frame by looping on the wall clock and/or making millisecond sleep
  // calls as needed.
  const double mintimeperframe = 1.0 / ((double) vs_targetFPS); 
  double nowtime = wkf_timer_timenow(timer);
  double timesincelastframe = fabs(nowtime - srv_lastframe) + 0.0001;
  while (timesincelastframe < mintimeperframe) {
    nowtime = wkf_timer_timenow(timer);
    timesincelastframe = fabs(nowtime - srv_lastframe) + 0.0001;
    if ((mintimeperframe - timesincelastframe) > 0.001) 
      vmd_msleep(1);  
  }

  // compute exponential moving average for exp(-1/10)
  double fps = 1.0 / timesincelastframe;
  expave_fps = (expave_fps * 0.90) + (fps * 0.10);
  
  srv_lastframe = nowtime; // update timestamp of last sent image

  return rc;
}


int VideoStream::srv_disconnect() {
  if (srv_socket != NULL) {
    vs_send_disconnect(srv_socket);
    vmdsock_destroy(srv_socket);
    cli_socket = 0;
    return 0;
  }
  return -1;
}


//
// Helper routines
//
int VideoStream::vs_readn(void *s, char *ptr, int n) {
  int nleft;
  int nread;

  nleft = n;
  while (nleft > 0) {
    if ((nread = vmdsock_read(s, ptr, nleft)) < 0) {
      if (errno == EINTR)
        nread = 0;         /* and call read() again */
      else
        return -1;
    } else if (nread == 0)
      break;               /* EOF */
    nleft -= nread;
    ptr += nread;
  }
  return n-nleft;
}


int VideoStream::vs_readn_discard(void *s, int discardsz) {
  // read+discard next discardsz bytes 
  while (discardsz > 0) {
    char buf[1024 * 1024];
    int readsz = (discardsz > sizeof(buf)) ? sizeof(buf) : discardsz;
    int n = vs_readn(cli_socket, buf, readsz);
    if (n < 0) {
      printf("VS: vs_readn_discard(): error reading message!\n");
      return -1;
    }
    discardsz -= n;
  } 
  return 0;
}


int VideoStream::vs_writen(void *s, const char *ptr, int n) {
  int nleft;
  int nwritten;

  nleft = n;
  while (nleft > 0) {
    if ((nwritten = vmdsock_write(s, ptr, nleft)) <= 0) {
      if (errno == EINTR)
        nwritten = 0;
      else
        return -1;
    }
    nleft -= nwritten;
    ptr += nwritten;
  }
  return n;
}


int VideoStream::vs_recv_handshake(void *s) {
  int buf;
  int type;

  /* Wait up to 5 seconds for the handshake to come */
  if (vmdsock_selread(s, 5) != 1) return -1;

  /* Check to see that a valid handshake was received */
  type = vs_recv_header_nolengthswap(s, &buf);
  if (type != VS_HANDSHAKE) return -1;

  /* Check its endianness, as well as the VS version. */
  if (buf == VS_PROTOCOL_VERSION) {
    if (!vs_send_go(s)) return 0;
    return -1;
  }
  swap4((char *)&buf, 4);
  if (buf == VS_PROTOCOL_VERSION) {
    if (!vs_send_go(s)) return 1;
  }

  /* We failed to determine endianness. */
  return -1;
}

int VideoStream::vs_send_handshake(void *s) {
  VSMsgHeader header;
  fill_header(&header, VS_HANDSHAKE, 0, 0, 0);
  header.len = VS_PROTOCOL_VERSION;   /* Not byteswapped! */
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_recv_header(void *s, VSMsgHeader &header) {
  if (vs_readn(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE)
    return VS_IOERROR;
  swap_header(&header);
  return (VSMsgType) header.type;
}

int VideoStream::vs_recv_header_nolengthswap(void *s, int *length) {
  VSMsgHeader header;
  if (vs_readn(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE)
    return VS_IOERROR;
  *length = header.len;
  swap_header(&header);
  return (VSMsgType) header.type;
}

int VideoStream::vs_send_disconnect(void *s) {
  VSMsgHeader header;
  fill_header(&header, VS_DISCONNECT, 0, 0, 0);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_send_pause(void *s) {
  VSMsgHeader header;
  fill_header(&header, VS_PAUSE, 0, 0, 0);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_send_go(void *s) {
  VSMsgHeader header;
  fill_header(&header, VS_GO, 0, 0, 0);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_send_heartbeat(void *s) {
  VSMsgHeader header;
  fill_header(&header, VS_HEARTBEAT, 0, 0, 0);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_send_resize(void *s, int width, int height) {
  VSMsgHeader header;
  fill_header(&header, VS_RESIZE, 0, width, height);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::vs_send_codec_reconfig(void *s, int bitrateMbps, int tFPS) {
  // XXX add a new reconfig-specific fill_header variant
  VSMsgHeader header;
  fill_header(&header, VS_CODEC_RECONFIG, 0, bitrateMbps, tFPS);
  return (vs_writen(s, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}


int VideoStream::cli_decode_frame(unsigned char *cbuf, long cbufsz,
                                  unsigned char *rgba, int width, int height) {
  int rc = -1;
#if defined(VIDEOSTREAM_SIMENCODER)
  rc = simenc_decode_frame(ench, cbuf, cbufsz, rgba, width, height);
#elif defined(VMDNVPIPE)
  rc = nvpipe_decode_frame(ench, cbuf, cbufsz, rgba, width, height);
#endif
  return rc;
}


int VideoStream::cli_send_rotate_by(float angle, char axis) {
  if (!cli_socket) return -1;

// printf("\nVS rotateby: %f  %d\n", angle, axis);

  VSMsgHeader header;
  fill_header_uievent(&header, VS_UIEVENT, VS_EV_ROTATE_BY, angle, (int)axis);
  return (vs_writen(cli_socket, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::cli_send_translate_by(float dx, float dy, float dz) {
  if (!cli_socket) return -1;

// printf("\nVS translateby: %f %f %f\n", dx, dy, dz);

  VSMsgHeader header;
  fill_header_uievent(&header, VS_UIEVENT, VS_EV_TRANSLATE_BY, dx, dy, dz);
  return (vs_writen(cli_socket, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::cli_send_scale_by(float scalefactor) {
  if (!cli_socket) return -1;

// printf("\nVS scaleby: %f \n", scalefactor);

  VSMsgHeader header;
  fill_header_uievent(&header, VS_UIEVENT, VS_EV_SCALE_BY, scalefactor);
  return (vs_writen(cli_socket, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::cli_send_keyboard(int dev, char val, int shift_state) {
  if (!cli_socket) return -1;

// printf("\nVS keyboard: %d %d %d\n", dev, val, shift_state);

  VSMsgHeader header;
  fill_header_uievent(&header, VS_UIEVENT, VS_EV_KEYBOARD,
                      (int) dev, (int) val, (int) shift_state);
  return (vs_writen(cli_socket, (char *)&header, VS_HEADERSIZE) != VS_HEADERSIZE);
}

int VideoStream::srv_check_ui_event() {
  srv_get_one_ui_event = 1;
  check_event();
  if (srv_last_event_type != VS_EV_NONE)
    return 1;

  return 0;
}

//
// UIObject virtual methods
//
int VideoStream::check_event() {
  // entry point to check for network events
  double curtime = wkf_timer_timenow(timer);

  // give status updates only once per 10 sec or so
  if ((curtime - lastconsolemesg) > 10) {
    int verbose = (getenv("VMDVIDEOSTREAMVERBOSE") != NULL);
    if (verbose) {
      printf("VS::check_event(): Cli: %s Srv: %s loops: %d\n",
             (cli_socket != NULL) ? "on" : "off",
             (srv_socket != NULL) ? "on" : "off",
             lastmsgeventloops);
    }
    lastconsolemesg = curtime;
    lastmsgeventloops = 0;
  }
  lastmsgeventloops++;

  //
  // VideoStream client-specific event handling implementation
  // 
  if (cli_socket) {
    // force main VMD event loop to run flat-out with no millisleep calls
    app->background_processing_set();

    // 
    // Process inbound mesgs from the video stream server
    //

    // loop to process all incoming data before continuing...
    int selectloops = 0;
    while (vmdsock_selread(cli_socket, 0) > 0) {
      VSMsgHeader header;
      int msgtype = vs_recv_header(cli_socket, header);
      int payloadsz = header.len;

      switch (msgtype) {
        case VS_HANDSHAKE:
          msgInfo << "VideoStream: received out-of-seq message type: " 
                  << msgtype << sendmsg;
          break;

        case VS_GO:
          // just eat the message, nothing specific to do presently
          break;

        case VS_HEARTBEAT:
#if 0
          printf("VS client received heartbeat message sz: %d\n", header.len);
#endif
          break;

        case VS_IMAGE:
          {
            // compute exponential moving average for exp(-1/10)
            double fps = 1.0 / (fabs(curtime - cli_lastframe) + 0.0001);
            expave_fps = (expave_fps * 0.90) + (fps * 0.10);
            printf("VS client recv image sz: %d, res: %dx%d  FPS:%.1f      \r", 
                   header.len, header.width, header.height, expave_fps);
            fflush(stdout);
#if defined(VIDEOSTREAM_STATICBUFS)
            unsigned char *compbuf = vs_cbuf;
#else
            unsigned char *compbuf = (unsigned char *) malloc(payloadsz);
#endif
            long imgsz = 4 * header.width * header.height;
            vs_readn(cli_socket, (char *) compbuf, payloadsz);

#if defined(VIDEOSTREAM_STATICBUFS)
            unsigned char *imgbuf = vs_imgbuf;
#else
            unsigned char *imgbuf = (unsigned char *) malloc(imgsz);
#endif
             
            unsigned long decodesz = 0;
#if defined(VIDEOSTREAM_SIMENCODER)
            decodesz = simenc_decode_frame(ench, compbuf, payloadsz, imgbuf, 
                                           header.width, header.height);
#elif defined(VMDNVPIPE)
            decodesz = nvpipe_decode_frame(ench, compbuf, payloadsz, imgbuf, 
                                           header.width, header.height);
#endif

            app->display->prepare3D(1);
            app->display->drawpixels_rgba4u(imgbuf, header.width, header.height);

#if 0
            { int ll, cnt;
              long inten=0;
              for (ll=0, cnt=0; ll<imgsz; ll++) {
                inten += imgbuf[ll];
                cnt += (imgbuf[ll] > 0);
              }
              printf("\nDecode: sz: %ld  Pixel stats: totalI: %ld, nonZero: %d \n", decodesz, inten, cnt);
            }
#endif

#if !defined(VIDEOSTREAM_STATICBUFS)
            free(imgbuf);
            free(compbuf);
#endif
            payloadsz = 0; // we've used all of the incoming data
          }
          cli_lastframe = curtime;
          break;

        case VS_DISCONNECT:
          printf("VS client received disconnect message sz: %d\n", header.len);
          vmdsock_destroy(cli_socket);
          cli_socket = 0;
          return 0; // bail all the way out since we no longer have a socket
          break;

        case VS_IOERROR:
          printf("VS client: I/O error, disconnecting!\n");
          vmdsock_destroy(cli_socket);
          cli_socket = 0;
          return 0; // bail all the way out since we no longer have a socket
          break;

        default:
          printf("VS client message type: %d  sz: %d\n", msgtype, header.len);
          break;
      }
      // read+discard next payloadsz bytes if any are left unused...
      if (payloadsz > 0) {
#if 0
        printf("VS server discarding payload, %d bytes\n", payloadsz);
#endif
        vs_readn_discard(cli_socket, payloadsz);
      }

      cli_lastmsgtime = curtime; 
      selectloops++;
    }
#if 0
    if (selectloops > 0) {
      printf("VideoStream: client select loops %d\n", selectloops);
    }
#endif

    // 
    // Send outbound mesgs to the video stream server
    //

    // check if the local display size has been changed
    int framesizechanged = 0;
    if (app->display->xSize != vs_width) {
      vs_width = app->display->xSize;
      framesizechanged = 1;
    }
    if (app->display->ySize != vs_height) {
      vs_height = app->display->ySize;
      framesizechanged = 1;
    }

    if (framesizechanged) {
      printf("\n");
      msgInfo << "VideoStream: client window resize: " 
              << vs_width << "x" << vs_height << sendmsg;
      vs_send_resize(cli_socket, vs_width, vs_height);
    } 

    // check whether codec parameters need reconfiguration
    if (vs_codec_reconfig_pending) {
      vs_send_codec_reconfig(cli_socket, vs_bitrateMbps, vs_targetFPS);
      vs_codec_reconfig_pending = 0;
    }

    // print warnings if we haven't gotten any messages for 5 sec
    if ((curtime - cli_lastmsgtime) > 5) {
      printf("VS: no mesgs for 5 seconds...\n");    
      cli_lastmsgtime = curtime; 
    }
  }


  //
  // VideoStream server-specific event handling implementation
  // 
  if (srv_socket) {
    // force main VMD event loop to run flat-out with no millisleep calls
    app->background_processing_set();

    // 
    // Process inbound mesgs from the video stream client
    //

    // loop to process all incoming data before continuing...
    int selectloops = 0;
    while (vmdsock_selread(srv_socket, 0) > 0) {
      VSMsgHeader header;
      int msgtype = vs_recv_header(srv_socket, header);
      int payloadsz = header.len;

      switch (msgtype) {
        case VS_HANDSHAKE:
          msgInfo << "VideoStream: received out-of-seq message type: " 
                  << msgtype << sendmsg;
          break;

        case VS_GO:
          // just eat the message, nothing specific to do presently
          break;

        case VS_HEARTBEAT:
#if 0
          printf("VS server received heartbeat, sz: %d\n", header.len);
#endif
          break;

        case VS_RESIZE:
          printf("\n");
          printf("VS server received resize, sz: %d new size: %dx%d\n", 
                 header.len, header.width, header.height);

          // force window resize to video encoder block-multiple
//          vs_width = ((header.width + 15) / 16) * 16;
//          vs_height = ((header.height + 15) / 16) * 16;
          vs_width = header.width;
          vs_height = header.height;

          //  XXX can't call through VMDApp chain because we can rapidly 
          //      get into a deeply recursive call chain leading to a segfault.
//          app->display_set_size(vs_width, vs_height);
          app->display->resize_window(vs_width, vs_height);

          vs_encoder_reconfig(); // apply new width/height to encoder
          break;

        case VS_CODEC_RECONFIG:
          // XXX add a new reconfig-specific fill_header variant
          printf("\n");
          printf("VS server received reconfig, sz: %d new Mbps: %d, FPS: %d\n", 
                 header.len, header.width, header.height);
          vs_bitrateMbps = header.width;
          vs_targetFPS = header.height;
          vs_encoder_reconfig(); // apply new width/height to encoder
          break;

        case VS_UIEVENT:
          {
#if 0
            printf("\nUIEvent: %d  [%d | %f]  [%d | %f]\n", 
                   header.eventtype, 
                   header.eventdata[0].ival, header.eventdata[0].fval,
                   header.eventdata[1].ival, header.eventdata[1].fval);
#endif

            srv_last_event_type = header.eventtype;
            switch (header.eventtype) {
              case VS_EV_ROTATE_BY:
                app->scene_rotate_by(header.eventdata[0].fval, header.eventdata[1].ival);
                srv_last_rotate_by_angle = header.eventdata[0].fval;
                srv_last_rotate_by_axis = header.eventdata[1].ival;
                break;

              case VS_EV_TRANSLATE_BY:
                app->scene_translate_by(header.eventdata[0].fval, header.eventdata[1].fval, header.eventdata[2].fval);
                srv_last_translate_by_vec[0] = header.eventdata[0].fval;
                srv_last_translate_by_vec[1] = header.eventdata[1].fval;
                srv_last_translate_by_vec[2] = header.eventdata[2].fval;
                break;

              case VS_EV_SCALE_BY:
                app->scene_scale_by(header.eventdata[0].fval);
                srv_last_scale_by_factor = header.eventdata[0].fval;
                break;

              case VS_EV_KEYBOARD:
               // forward keyboard events through normal command queues
                runcommand(new UserKeyEvent((DisplayDevice::EventCodes) header.eventdata[0].ival, 
                           (char) header.eventdata[1].ival, 
                           header.eventdata[2].ival));
                srv_last_key_dev = header.eventdata[0].ival;
                srv_last_key_val = header.eventdata[1].ival;
                srv_last_key_shift_state = header.eventdata[2].ival;
                break;

              default:
                printf("\nUnhandled UIEvent: %d  [%d | %f]\n", 
                       header.eventtype,
                       header.eventdata[0].ival, header.eventdata[0].fval);
                break;
            }

            if (srv_get_one_ui_event) {
              return 1; // early-exit so caller can catch events 1-at-a-time
            }
          }
          break;

        case VS_DISCONNECT:
          printf("VS server: received disconnect, sz: %d\n", header.len);
          vmdsock_destroy(srv_socket);
          srv_socket = 0;
          return 0; // bail all the way out since we no longer have a socket
          break;

        case VS_IOERROR:
          printf("VS server: I/O error, disconnecting!\n");
          vmdsock_destroy(srv_socket);
          srv_socket = 0;
          return 0; // bail all the way out since we no longer have a socket
          break;

        default:
          printf("VS server message type: %d  sz: %d\n", msgtype, header.len);
          break;
      }
      // read+discard next payloadsz bytes if any are left unused...
      if (payloadsz > 0) {
#if 0
        printf("VS server discarding payload, %d bytes\n", payloadsz);
#endif
        vs_readn_discard(srv_socket, payloadsz);
      }

      srv_lastmsgtime = curtime; 
      selectloops++;
    }
#if 0
    if (selectloops > 0) {
      printf("VideoStream: server select loops %d\n", selectloops);
    }
#endif

    // 
    // Send outbound mesgs to the video stream client
    //

    // check if the local display state has been changed
    if (vs_framepending) {
      unsigned char *img = NULL;
      if (vs_rgba_pend != NULL) {
        srv_send_frame(vs_rgba_pend, vs_rgba_width * 4, 
                       vs_rgba_width, vs_rgba_height, vs_forceIframe);
      } else {
        // if no frame was provided, we grab the GL framebuffer
        int xs, ys;
        img = app->display->readpixels_rgba4u(xs, ys);
        if (xs == 0 || ys == 0) {
          printf("VS: check_event() zero framebuffer dim!: %d x %d\n", xs, ys);
        }

        if (img != NULL) {
          srv_send_frame(img, xs * 4, xs, ys, vs_forceIframe);
          free(img);
        }
      }

      // clear any pending frame encoding flags
      vs_framepending = 0;
      vs_forceIframe = 0;
    }

    // send a heartbeat message if no other msg was sent in last second
    if ((curtime - srv_lastmsgtime) > 1) {
      vs_send_heartbeat(srv_socket);
      srv_lastmsgtime = curtime; 
    }
  } 
 
  return 0;
}

int VideoStream::act_on_command(int type, Command *cmd) {
  // take any appropriate actions based on incoming commands
  return 0;
}



