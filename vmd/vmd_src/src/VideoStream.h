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
*      $Revision: 1.21 $         $Date: 2019/01/17 21:21:02 $
*
***************************************************************************
* DESCRIPTION:
*   VMD interface for video streaming
*
***************************************************************************/

#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#define VIDEOSTREAM_SUCCESS  0
#define VIDEOSTREAM_ERROR   -1

// #define VIDEOSTREAM_STATICBUFS 1

#include <stdlib.h>
#include "UIObject.h"
#include "WKFUtils.h"

struct VSMsgHeader_t;

class VideoStream : public UIObject {
  public:
    VideoStream(VMDApp *);
    ~VideoStream(void);

    int cli_listen(int port);
    int cli_connect(const char *hostname, int port);
    int cli_wait_msg();
    int cli_disconnect();
    const char *cli_gethost() { return "Unknown"; };
    int cli_getport() { return 0; };
    int cli_connected() { return (cli_socket != NULL); }
    int cli_decode_frame(unsigned char *cbuf, long cbufsz,
                         unsigned char *rgba, int width, int height);

    // client-side UI event transmission routines
    int cli_send_rotate_by(float angle, char axis);
    int cli_send_translate_by(float dx, float dy, float dz);
    int cli_send_scale_by(float scalefactor);
    int cli_send_keyboard(int dev, char val, int shift_state);

    int srv_listen(int port);
    int srv_connect(const char *hostname, int port);
    int srv_send_frame(const unsigned char *rgba, int pitch, 
                       int width, int height, int forceIFrame);
    int srv_disconnect();
    const char *srv_gethost() { return "Unknown"; };
    int srv_getport() { return 0; };
    int srv_connected() { return (srv_socket != NULL); }

    int set_target_bitrate_Mbps(float Mbps) { 
      vs_bitrateMbps = Mbps;
      vs_codec_reconfig_pending = 1;
      return 0;
    }

    int set_target_frame_rate(float tfps) { 
      vs_targetFPS = tfps; 
      vs_codec_reconfig_pending = 1;
      return 0;
    }

    // Key virtual methods inherited from UIObject
    int check_event();
    int act_on_command(int, Command *);

    // methods used to inform videostream of graphics updates 
    // and/or to funnel in incoming image data
    void video_frame_pending(const unsigned char *ptr = NULL,
                             int pwidth=0, int pheight=0) { 
      vs_rgba_pend = ptr;
      vs_rgba_width = pwidth;
      vs_rgba_height = pheight;
      vs_framepending = 1; 
    }
    void video_frame_force_Iframe() { vs_forceIframe = 1; }

    enum VSEventDev {
      VS_EV_NONE,                       ///< no event occured; never sent
      VS_EV_ROTATE_BY,                  ///< rotation on cartesian axis by angle
      VS_EV_TRANSLATE_BY,               ///< 3DoF translation
      VS_EV_SCALE_BY,                   ///< uniform scaling operation
      VS_EV_KEYBOARD                    ///< key down event
    };

    // special method for use by OptiXRenderer
    int srv_check_ui_event();

    ///< return last UI event type we received, VS_EV_NONE if none was recvd
    int srv_process_all_events() { 
      srv_get_one_ui_event=0; 
      return 0;
    }

    ///< return last UI event type we received, VS_EV_NONE if none was recvd
    int srv_get_last_event_type(int &eventtype) { 
      eventtype = srv_last_event_type; 
      return 0;
    }

    ///< return last rotation event info.
    int srv_get_last_rotate_by(float &angle, int &axis) {
      angle = srv_last_rotate_by_angle;
      axis = srv_last_rotate_by_axis;

      srv_last_event_type = VS_EV_NONE; // clear event so we read it only once
      return 0;
    }

    ///< return last translation event info.
    int srv_get_last_translate_by(float &tx, float &ty, float &tz) {
      tx = srv_last_translate_by_vec[0];
      ty = srv_last_translate_by_vec[1];
      tz = srv_last_translate_by_vec[2];
 
      srv_last_event_type = VS_EV_NONE; // clear event so we read it only once
      return 0;
    }

    ///< return last scaling event info.
    int srv_get_last_scale_by(float &factor) {
      factor = srv_last_scale_by_factor;

      srv_last_event_type = VS_EV_NONE; // clear event so we read it only once
      return 0;
    }

    ///< return last keyboard event info.
    int srv_get_last_keyboard(int &dev, int &val, int &shift_state) {
      dev = srv_last_key_dev;
      val = srv_last_key_val;
      shift_state = srv_last_key_shift_state;

      srv_last_event_type = VS_EV_NONE; // clear event so we read it only once
      return 0;
    }

  private:
    int srv_get_one_ui_event;           ///< flag to early-exit main event loop
                                        ///< on server-side UI event arrival
    int srv_last_event_type;            ///< last event we received
    int srv_last_rotate_by_axis;        ///< last rotate_by axis
    float srv_last_rotate_by_angle;     ///< last rotate_by angle
    float srv_last_translate_by_vec[3]; ///< last translate by offset
    float srv_last_scale_by_factor;     ///< last scale_by factor
    int srv_last_key_dev;               ///< last keyboard event device
    int srv_last_key_val;               ///< last keyboard event value  
    int srv_last_key_shift_state;       ///< last keyboard event shift state

    int vs_framepending;                ///< a frame is ready for pull+encode 
    const unsigned char *vs_rgba_pend;  ///< ptr to pending frame if one exists
    int vs_rgba_width;                  ///< width of pending frame            
    int vs_rgba_height;                 ///< height of pending frame           

    int vs_forceIframe;                 ///< force next frame to be an I-frame
    int vs_width;                       ///< width, X dimension of video stream
    int vs_height;                      ///< height, Y dimension of video stream
    int vs_targetFPS;                   ///< target frame rate
    int vs_bitrateMbps;                 ///< target bitrate
    int vs_codec_reconfig_pending;      ///< encoder needs reconfig

    double imagesz_comp;                ///< exp ave compressed image size
    double imagesz_uncomp;              ///< exp ave uncompressed image size
    double expave_fps;                  ///< exponential average of FPS rate

    void *ench;                         ///< hardware-specific encoder ctx
    void *cli_socket;                   ///< inbound image stream
    void *srv_socket;                   ///< outbound image stream

#if defined(VIDEOSTREAM_STATICBUFS)
    unsigned char *vs_imgbuf;           ///< compressed framebuffer (cli+srv)
    int vs_imgbufsz;                    ///< max size of compressed framebuffer
    unsigned char *vs_cbuf;             ///< compressed framebuffer (cli+srv)
    int vs_cbufsz;                      ///< max size of compressed framebuffer
#endif

    wkf_timerhandle timer;              ///< message/heartbeat timer
    double lastconsolemesg;             ///< time since last console mesg
    double cli_lastmsgtime;             ///< time of most recent message arrival
    double cli_lastframe;               ///< time since last image frame recv
    double srv_lastmsgtime;             ///< time of most recent message send
    double srv_lastframe;               ///< time since last image frame sent
    int lastmsgeventloops;              ///< event loop count since last arrival

    enum VSMsgType {
        VS_HANDSHAKE,                   ///< endianism+version check message
        VS_GO,                          ///< start video streaming
        VS_PAUSE,                       ///< pause video streaming
        VS_CODEC_RECONFIG,              ///< reconfigure codec parameters
        VS_IMAGE,                       ///< compressed video images
        VS_RESIZE,                      ///< video stream resize event
#if 0
        VS_KILL,                        ///< kill the connection
        VS_TRATE,                       ///< set IMD update transmission rate
#endif
        VS_UIEVENT,                     ///< various mouse/keyboard/UI events
        VS_HEARTBEAT,                   ///< live connection heartbeat signal
        VS_DISCONNECT,                  ///< close video stream with VMD running
        VS_IOERROR                      ///< indicate an I/O error
    };

    int vs_encoder_reconfig();

    int vs_recv_handshake(void *sock);
    int vs_send_handshake(void *s);

    int vs_recv_header(void *s, VSMsgHeader_t &header);
    int vs_recv_header_nolengthswap(void *s, int *length);
    int vs_readn(void *s, char *ptr, int n);
    int vs_readn_discard(void *s, int n);
    int vs_writen(void *s, const char *ptr, int n);
    int vs_send_disconnect(void *s);
    int vs_send_pause(void *s);
    int vs_send_go(void *s); 
    int vs_send_heartbeat(void *s);
    int vs_send_resize(void *s, int w, int h);
    int vs_send_codec_reconfig(void *s, int bitrateMbps, int targetFPS);
};

#endif
