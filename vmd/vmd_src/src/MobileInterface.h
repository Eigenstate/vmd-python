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
 *	$RCSfile: MobileInterface.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.17 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Mobile UI object, which maintains the current state of the 
 * mobile phone, tablet, or other WiFi input or auxiliary display device.
 *
 ***************************************************************************/
#ifndef MOBILEINTERFACE_H
#define MOBILEINTERFACE_H

#include "UIObject.h"
#include "Command.h"
#include "JString.h"
#include "NameList.h"
#include "WKFUtils.h"

#if 0
class MobileClientList {
  public:
    MobileClientList() {}
    MobileClientList(const JString n, const JString i, const bool a) :
             nick(n), ip(i), active(a) {}
  private:
    JString nick;
    JString ip;
    bool active;
};
#endif

/// UIObject subclass for mobile/wireless phone/tablet motion control
class Mobile : public UIObject {
public:
  /// enum for Mobile movement modes
  enum MoveMode { OFF, MOVE, ANIMATE, TRACKER, USER };

  /// enum for multitouch motion mode
  enum TouchMode { ROTATE, TRANSLATE, SCALEROTZ };

  /// gets a string representing a mode's name
  static const char *get_mode_str(MoveMode mode);

private:
  void *mobile;
  int port;                  ///< UDP port to receive incoming packets on

  MoveMode moveMode;         ///< the current move mode
  int maxstride;             ///< maximum stride when in animate mode
  float transInc;            ///< increment for translation
  float rotInc;              ///< increment for rotation
  float scaleInc;            ///< increment for scaling
  float animInc;             ///< increment for animation
  int buttonDown;            ///< which buttons are down 

  int touchinprogress;       ///< multitouch event is in-progress
  int touchcount;            ///< number of concurrent touches
  int touchmode;             ///< multitouch motion mode
  wkf_timerhandle packtimer; ///< packet timer for multitouch updates
  float touchstartX;         ///< initial X location of first touch
  float touchstartY;         ///< initial Y location of first touch
  float touchdeltaX;         ///< current X delta from first touch location
  float touchdeltaY;         ///< current Y delta from first touch location
  float touchscale;          ///< current multitouch scale factor
  float touchscalestartdist; ///< multitouch scaling normalization (spread)
  float touchrotstartangle;  ///< initial multitouch rotation angle 

  float tranScaling;         ///< global translation scaling factor (1.0 def)
  float rotScaling;          ///< global rotation scaling factor (1.0 def)
  float zoomScaling;         ///< global zoom scaling factor (1.0 def)


  /// tracker data reported to MobileTracker
  float trtx;                ///< X translation
  float trty;                ///< Y translation
  float trtz;                ///< Z translation
  float trrx;                ///< X rotation
  float trry;                ///< Y rotation
  float trrz;                ///< Z rotation
  int trbuttons;             ///< button status 

  ResizeArray <JString *> clientNick;              ///< client nickname list
  ResizeArray <JString *> clientIP;                ///< client IP list
  ResizeArray <bool> clientActive;                 ///< client activity list
  ResizeArray <int> clientListenerPort;            ///< client port list
  ResizeArray <wkf_timerhandle> clientLastContact; ///< client heartbeat timer

  /// check if a particular client IP/port is controlling
  bool isInControl(JString* nick, JString* ip, const int port, 
                   const int packtype);

  float statusSendSeconds;          ///< client status send/update interval
  wkf_timerhandle statustimer;      ///< internal status update timer
  void checkAndSendStatus();        ///< send link heartbeat if interval reached
  void sendStatus(const int event); ///< send event to all active clients

  /// 65507 bytes is the max UDP packet length that can be sent, 
  /// http://wikipedia.org/wiki/User_Datagram_Protocol
  /// but such large packets would be fragmented, so it is better to 
  /// stick with a packet size that will fit into common ethernet MTUs.
  /// The same value is specified in UDPServer.java on the mobile side.
  unsigned char statusSendBuffer[1536]; ///< single-packet UDP 
  int statusSendBufferLength;

  /// prepare statusSendBuffer to notify of an event
  void prepareSendBuffer(const int eventType);

  /// send an event UDP datagram message to a target IP address 
  void sendMsgToIp(const bool isActive,
                   const int tmpInt,
                   const char *msg,
                   const char *ip,
                   const int port);

public:
  Mobile(VMDApp *);      ///< constructor
  virtual ~Mobile(void); ///< destructor
  
  //
  // virtual routines for UI init/display
  //
   
  /// reset the user interface (force update of all info displays)
  virtual void reset(void);
  
  /// update the display due to a command being executed.  Return whether
  /// any action was taken on this command.
  /// Arguments are the command type, command object, and the 
  /// success of the command (T or F).
  virtual int act_on_command(int, Command *); ///< command execute update
  
  /// check for and event, queue and return TRUE if one is found.  
  virtual int check_event(void);

  /// get the currently configured port number
  int get_port ();

  /// get the currently supported API level
  int get_APIsupported ();

  /// get the currently configured mode
  int get_move_mode ();

  /// get the current list of connected clients
  void get_client_list(ResizeArray <JString*>* &nick, 
                       ResizeArray <JString*>* &ip, 
                       ResizeArray <bool>* &active);

  /// set the Mobile move mode to the given state; return success
  int move_mode(MoveMode mm);

  /// set the active client, based on their nick and IP
  int set_activeClient(const char *nick, const char *ip);

  /// send a message to a single client
  int sendMsgToClient(const char *nick, const char *ip, const char *msgType, const char *msg);

  /// set the incoming UDP port (closing the old one if needed)
  int network_port(int newport);

  /// set the maximum animation stride allowed
  void set_max_stride(int ms) {
    maxstride = ms;
  }

  /// return the current orientation event data, 
  /// used by the UIVR MobileTracker interface
  void get_tracker_status(float &tx, float &ty, float &tz, 
                          float &rx, float &ry, float &rz, int &buttons);


  /// add a new client device connection, user nickname, IP, etc.
  int addNewClient(JString* nick, JString* ip, const int port, const bool active);

  /// remove a connected client device
  int  removeClient(const int num);

  /// disconnect all client devices
  void removeAllClients();
};


/// change the current mouse mode
/// This command doesn't generate an output text command, it is just
/// used to change the VMD internal state
class CmdMobileMode : public Command {
public:
  /// specify new mode and setting
  CmdMobileMode(int mm) : Command(MOBILE_MODE), mobileMode(mm) {}

  /// mode and setting for the mouse
  int mobileMode;
};


#endif

