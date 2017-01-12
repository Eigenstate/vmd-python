/***************************************************************************
 *cr
 *cr            (C) Copyright 2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
package edu.uiuc.ks.vmdmobile;


import android.app.Activity;
import android.app.AlertDialog;
import android.app.TabActivity;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.res.Configuration;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.view.View;
import android.view.ViewGroup;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.BaseAdapter;
import android.widget.FrameLayout;
import android.widget.Gallery;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TabWidget;
import android.widget.TabHost;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.EditText;
import android.widget.Button;

import org.metalev.multitouch.controller.MultiTouchController.PointInfo;

import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

public class VMDMobile extends TabActivity implements SensorEventListener {
    final String tag = "VMDMobile";
    final String prefFile = MainPreferences.prefFile; 

    // athine
//    public static final String DEFAULT_VMD_IP = "130.126.120.165";
    public static final String DEFAULT_VMD_IP = "127.0.0.1";
    public static final int    DEFAULT_VMD_PORT = 3141;
    public static final int    DEFAULT_LISTEN_PORT = 4141;
    public static final float  DEFAULT_ROTSCALE_FACTOR = 1.0f;
    public static final float  DEFAULT_ZOOMSCALE_FACTOR = 1.0f;
    public static final float  DEFAULT_TRANSCALE_FACTOR = 1.0f;

    // used to identify subactivities
    final int ACTIVITY_SERVER_CHOOSER = 0;

    public static final int API_VERSION = 9;

    private boolean m_bInGetSocket = false;
    private boolean m_bInSendPacket = false;
    private boolean m_bServerChooserCanceled = false;

    final int DEVICE_STATE = 1;
    final int TOUCH_STATE = 2;
    final int HEART_BEAT = 3;
    final int CONNECT_MSG = 4;
    final int DISCONNECT_MSG = 5;
    final int BUTTON_EVENT = 6;
    final int COMMAND_EVENT = 7;

    final int SERVER_HEARTBEAT=         0;
    final int SERVER_ADDCLIENT=         1;
    final int SERVER_REMOVECLIENT=      2;
    final int SERVER_SETACTIVECLIENT=   3;
    final int SERVER_SETMODE=           4;
    final int SERVER_MESSAGE=           5;

    final int COMMAND_TAKESNAPSHOT =           0;
    final int COMMAND_HASH         =           1;
    final int COMMAND_REQUEST_BUTTONCONFIG =   2;
    final int COMMAND_RESETVIEW =              3;

    private Hashtable m_htButtonList;
    private String m_strCurrentButtonGroup = "Default Aux";

    private boolean m_bServerSet = false;
    private String m_strCurrentServer;
    private int m_iCurrentPort;
    private int m_iCurrentListenPort;

   ArrayList<TabHost.TabSpec> listOfTabs = new ArrayList<TabHost.TabSpec>();   
    DatagramSocket skt;

    SensorManager sm = null;
    
//    ToggleButton toggleSensorSend = null;
//    ToggleButton toggleTouchSend = null;
//    ToggleButton button0 = null;

    MultiTouchPad touchPad = null;

    TextView console = null;
    TextView ipInfo = null;

    RelativeLayout mainLayout = null;

    private TabHost th;

    int imgTabNum = 0;

    final int redColor = 0xff000000 
              + 50 * 0x10000
              +  0 * 0x100
              +  0;
    final int yellowColor = 0xff000000 
              + 50 * 0x10000
              + 50 * 0x100
              +  0;
    final int greenColor = 0xff000000 
              +  0 * 0x10000
              + 50 * 0x100
              +  0;

    final int notConnectedColor = redColor;
    final int notInControlColor = yellowColor;
    final int inControlColor = greenColor;

    int currentColor = notConnectedColor;


    SharedPreferences mPrefs = null;
    float [] orientation;
    float [] accel;
    float [] magField;
    float [] gyroscope;
    float [] Rmat;
    float [] Imat;

    int i;

    int buttonState = 0;

    boolean sendingOrient = false;
    boolean sendingTouch = false;

    int screenWidth;
    int screenHeight;
    float xdpi;
    float ydpi;

    long lastTime = System.currentTimeMillis();

    private static final int INVALID_POINTER = -1;
    private int activePointer = INVALID_POINTER;

    // lengths in bytes
    private int headerLength;
    private int sensorPayloadLength;
    private int touchPayloadLength;
    private int touchPointsSupported;
    
    private int COMMAND_DATA_SIZE;
    private int BUTTON_DATA_SIZE;
    private int HEARTBEAT_DATA_SIZE;
    private int CONNECT_DATA_SIZE;
    private int DISCONNECT_DATA_SIZE;
    private int SENSOR_DATA_SIZE;
    private int TOUCH_DATA_SIZE;

    private static final int MAX_PACKET_SIZE = 2048;  // arbitrary
    private byte [] data;
    private byte [] buttonData;
    private byte [] commandData;
    private byte [] heartBeatData;
    private int sequenceNumber = 0;
    private DatagramPacket pkt = null;

    private HeartBeat m_heartBeat;
    private UDPServer m_udpServer;
    private int HEARTBEATMILLIS = 2000;

    private int m_iApiVersion = API_VERSION;

    private PointInfo oldTouchPoints = new PointInfo();

// -------------------------------------------------------------
    // handles up to 32 buttons
    private void setButtonState(final int id, final boolean value)
    {
       int oldState = buttonState;
       // buttonState is the int
       if (value)
       {
          buttonState |= (1 << id);
       } else {
          buttonState &= ~(1 << id);
       }
       if (oldState != buttonState)
       {
          sendButtonState();
       }
//       logConsole("button" + id + ",val: " + value +"," + buttonState);
    }  // end of setButtonState

   

   // ----------------------------------------------------------------
   public void vmdServerGone()
   {
      Toast.makeText(getApplicationContext(), "VMD server has gone away!",
                                                   Toast.LENGTH_LONG).show();
      mainLayout.setBackgroundColor(notConnectedColor);
      currentColor = notConnectedColor;
   } // end of vmdServerGone()

   // ----------------------------------------------------------------
   public void packetReceived(final byte [] datapkt, final int pktLength, final String kstrAddr)
   {
      // let's decode what we've received.
      ByteBuffer bb = ByteBuffer.wrap(datapkt);

      int offset = 0;
      int iEndian = bb.getInt(offset);
      if (iEndian != 1) {
         // let's try changing the byte order..
         if (bb.order() == ByteOrder.BIG_ENDIAN) {
            bb.order(ByteOrder.LITTLE_ENDIAN);
         } else {
            bb.order(ByteOrder.BIG_ENDIAN);
         }
         iEndian = bb.getInt(offset);
         if (iEndian != 1) {
            // we've failed.  Tell the user.
            logConsole("Dropped packet: Wrong format");
         }
      }
      offset += 4;

      int serverVersion = bb.getInt(offset);
      offset += 4;

      int mode = bb.getInt(offset);
      offset += 4;

      int event = bb.getInt(offset);
      offset += 4;

      int isActive = bb.getInt(offset);
      offset += 4;
      if (isActive == 1) {
         if (currentColor != inControlColor) {
            mainLayout.setBackgroundColor(inControlColor);
            currentColor = inControlColor;
         }
      } else {
         if (currentColor != notInControlColor) {
            mainLayout.setBackgroundColor(notInControlColor);
            currentColor = notInControlColor;
         }
      }

      // -----------------------------
      // next bit of code should print the connected clients
      // on another tab of the app
      int numConnections = bb.getInt(offset);
      offset += 4;

//      Log.d("UDPServer", "m:" + mode + ",a:" + isActive + ",n:" + numConnections);
      String strClients = "";
      // now we need to loop through connections.
      for (int i=0; i < numConnections; i++) {
         int nickLength = bb.getInt(offset);
         offset += 4;
//         Log.d("UDPServer", "Nick length:" + nickLength);

         // we need to find the end of the string.

         String str = new String(datapkt, offset, nickLength);
         offset += nickLength;

         int clientActive = bb.getInt(offset);
         offset += 4;
         strClients += str + "," + clientActive + ";";
      }

//      logConsole("m:" + mode + ",a:" + isActive + ",n:" +
//          numConnections + "," + strClients);
      // -----------------------------

//      Log.d(tag, "Packet:iEndian:" + iEndian + ",server version:" + serverVersion + ",mode:" + mode + ",event:" + event + ",isActive:" + isActive + ",numConn:" + numConnections + ",clients:" + strClients);

      // Now we need to process what we actually get.
      switch (event) {
         case SERVER_MESSAGE:
            int msgType = bb.getInt(offset);
            offset += 4;

            int msgLength = bb.getInt(offset);
            offset += 4;

            String str = new String(datapkt, offset, msgLength);
            offset += msgLength;

//            Log.d(tag, "MsgType: " + msgType + ", length: " + msgLength + ", str: '" + str + "'");
            switch (msgType) {
               case 0:                                        // snapshot
                  // we need to get the snapshot at http://server:5141/str
                  addSnapshot( "http://" + m_strCurrentServer
                               + ":5141/" +  str);
                  break;

               case 1:            // generic image.. likely a GUI window
                  // decode 'str'.  Will be:     ImageTabName URLstr  
//                  logConsole("str is '" + str + "'");
                  String [] arr = str.split(" ");

                  addImageTab( "http://" + m_strCurrentServer
                                                    + ":5141/" +  arr[1],
                               "snap" + imgTabNum++,
                               arr[0],
                               false);
                  break;

               case 2:    // this is a list of buttons we should be displaying
                          // format is:
//   buttonGroupHash(Timeline) = {Get 824242165}
//   buttonGroupHash(vcr)      = {Next 769396887} {Prev 777179245}
   // msg is:
//       2 Timeline 1 Get 824242165 vcr 2 Next 769396887 Prev 777179245
//                  Log.d(tag, "read string '" + str + "'");
                  arr = str.split(" ");
//                  Log.d(tag, "arr has " + arr.length + " elems");
                  Hashtable ht = new Hashtable();
                  int numButtonGroups = Integer.parseInt(arr[0]);
                  int iCount = 1;
                  for (int i=0; i < numButtonGroups; i++)
                  {
                     String strGroupName = arr[iCount++];
                     int iButtonsInGroup = Integer.parseInt(arr[iCount++]);
                     Vector buttonInfo = new Vector();
                     for (int j=0; j < iButtonsInGroup; j++) 
                     {
                        String strButtonName = arr[iCount++];
                        String strButtonHash = arr[iCount++];
                        buttonInfo.add(new Tuple(strButtonName, strButtonHash));
                     }

                     ht.put(strGroupName, buttonInfo);
                  }
                  processButtonMessage(ht);

                  // check to see if we need to update buttons that
                  // are displayed right now
//                  Log.d(tag, "ht is:" + ht);
                  break;

               default:
                  logConsole("Unknown msgType (" + msgType + ") received.");
            }

         break;  // end case SERVER_MESSAGE

      }
   }  // end of packetReceived

   // ----------------------------------------------------------------
   private void processButtonMessage(Hashtable ht) {
      // check to see if the currently shown collection of buttons has
      // changed.   
      Vector vNew = (Vector) ht.get(m_strCurrentButtonGroup);
      Vector vOld = (Vector) m_htButtonList.get(m_strCurrentButtonGroup);

      m_htButtonList = ht;
      
      if (vNew != vOld)  // it has changed.. Let's reconfigure
      {
         configureButtonBar(m_strCurrentButtonGroup);
      }
   }  // end of processButtonMessage

   // ----------------------------------------------------------------
   // ----------------------------------------------------------------
   class GetSocket extends AsyncTask<String, Void, Object>
   {

      protected Object doInBackground(String... parms)
      {
         try {
            logConsole("GetSocket.doInBackground: Start");
            InetAddress addr = InetAddress.getByName(m_strCurrentServer);
            skt = new DatagramSocket(m_iCurrentPort);
            pkt = new DatagramPacket( data, MAX_PACKET_SIZE, addr,
                          m_iCurrentPort);
            logConsole("GetSocket.doInBackground: End of attempt");
         } catch (Exception e) {
            logConsole("GetSocket exception:" + e);
         }
         m_bInGetSocket = false;
         return null;
      }
      protected void onPostExecute(Object o) {
         logConsole("Done creating addr, skt, and pkt");
      }
   }
   // ----------------------------------------------------------------
   // ----------------------------------------------------------------

   // ----------------------------------------------------------------
   private void openUDPSocket()
   {
      try {
//         InetAddress addr = InetAddress.getByName(m_strCurrentServer);
//         skt = new DatagramSocket( m_iCurrentPort);
         m_bInGetSocket = true;
         new GetSocket().execute();
         // broadcasting UDP doesn't seem to work on 130.126.120.255.
         // hard to say for positive that this is true, but it seems to
         // fail using the technique at:
         //   http://code.google.com/p/boxeeremote/wiki/AndroidUDP
//         skt.setBroadcast(true);
//         pkt = new DatagramPacket( data, MAX_PACKET_SIZE, addr,
//                          m_iCurrentPort);
//         Log.d(tag, "openUDPSocket: Opened socket and created packet properly");
         while (m_bInGetSocket) Thread.sleep(100);
         ipInfo.setText("VMD Server: " + m_strCurrentServer
               + ":" +  m_iCurrentPort);
      } catch (Exception e) {
         e.printStackTrace();
         logConsole("InitError: " + e);
         startServerChooserIntent();
//         ipInfo.setText(e.toString() + ": Invalid VMD Server: " + 
//                     m_strCurrentServer + ":" + 
 //              m_iCurrentPort);
//         Log.d(tag, "openUDPSocket:" + e.toString());
      }
   } // end of openUDPSocket

// -------------------------------------------------------------
   private void startSensorRecording()
   {
      if (!sendingOrient)
      {
         Sensor aSensor = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
         sm.registerListener(this,aSensor,SensorManager.SENSOR_DELAY_FASTEST);

//         Sensor oSensor = sm.getDefaultSensor(Sensor.TYPE_ORIENTATION);
//         sm.registerListener(this,oSensor,SensorManager.SENSOR_DELAY_FASTEST);

         Sensor mSensor = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
         sm.registerListener(this,mSensor,SensorManager.SENSOR_DELAY_FASTEST);

         Sensor gSensor = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
         sm.registerListener(this,gSensor,SensorManager.SENSOR_DELAY_FASTEST);

         sendingOrient = true;
      }
   } // end of startSensorRecording

// -------------------------------------------------------------
   private void startSending()
   {
//      logConsole("startSending. Begin");
      if (skt == null || skt.isClosed())
      {
         openUDPSocket();
      }

      if (skt == null || skt.isClosed())
      {
//         logConsole("startSending. skt was null or closed");
         startServerChooserIntent();
         // something didn't work
//         ipInfo.setText("Invalid VMD Server: " + 
//                  m_strCurrentServer+ ":" + 
//               m_iCurrentPort);
         return;
      }
   
      sendConnectMsg();

      if (m_heartBeat != null)
      {
         m_heartBeat.setContinue(false);
         try {
            m_heartBeat.interrupt();
         } catch(SecurityException e) { /* don't care */ }
         m_heartBeat = null;
      } 
      m_heartBeat = new HeartBeat(HEARTBEATMILLIS, this);
      m_heartBeat.start();
      sendingTouch = true;
   } // end of startSending()

// -------------------------------------------------------------
   private void stopAllSending()
   {
      // Send goodbye message
      sendDisconnectMsg();

//      Log.d(tag, "Stopping sending");
      if (m_heartBeat != null)
      {
         m_heartBeat.setContinue(false);
         try {
            m_heartBeat.interrupt();
         } catch(SecurityException e) { /* don't care */ }
         m_heartBeat = null;
      }

      if (m_udpServer != null)
      {
         m_udpServer.setContinue(false);
         try {
            m_udpServer.interrupt();
         } catch(SecurityException e) { /* don't care */ }
         m_udpServer = null;
      }


      stopSensorRecording();
      sendingOrient = sendingTouch = false;
//      if (toggleTouchSend != null) toggleTouchSend.setChecked(false);
//      if (toggleSensorSend != null) toggleSensorSend.setChecked(false);
      if (m_bInSendPacket == false)
      {
         if (skt != null && !skt.isClosed())
         {
            skt.close();
            skt = null;
         }
      }
   }   // end of stopAllSending()

// -------------------------------------------------------------
   private void stopSensorRecording()
   {
      sm.unregisterListener(this);
      sendingOrient = false;;
   }   // end of stopSensorRecording

   // -----------------------------------------------------------------
   // prepare 'data' for shipment 
   private void packDeviceState()
   {
       int caret = headerLength;

       // pack payload description (DEVICE_STATE, in this case)
       packInt(DEVICE_STATE, data, caret);
       caret += 4;   // sizeof(int)

       packInt(buttonState, data, caret);
       caret += 4;   // sizeof(int)

       // strictly monotonically increasing number
       packInt(sequenceNumber++, data, caret );
       caret += 4;   // sizeof(int)


       // orientation
       packFloatArray(orientation, data, caret );
       caret += orientation.length * 4;  // 4 is sizeof(float)

       packFloatArray(accel, data, caret);
       caret += accel.length * 4;  // 4 is sizeof(float)


       packFloatArray(Rmat, data, caret);
       caret += Rmat.length * 4;  // 4 is sizeof(float)

   } // end of packDeviceState

   // -----------------------------------------------------------------
    // prepare 'data' for shipment 
    private void packTouchState(PointInfo currTouchPoints, final int width, final int height)
    {
       int caret = headerLength;

       int numPoints = Math.min(touchPointsSupported, currTouchPoints.getNumTouchPoints());

       // pack payload description (DEVICE_STATE, in this case)
       packInt(TOUCH_STATE, data, caret);
       caret += 4;   // sizeof(int)

       packInt(buttonState, data, caret);
       caret += 4;   // sizeof(int)

       // strictly monotonically increasing number
       packInt(sequenceNumber++, data, caret );
       caret += 4;   // sizeof(int)

       packFloat(xdpi, data, caret);
       caret += 4;   // sizeof(float)

       packFloat(ydpi, data, caret);
       caret += 4;   // sizeof(float)

       packInt(width, data, caret);
       caret += 4;   // sizeof(int)

       packInt(height, data, caret);
       caret += 4;   // sizeof(int)

       packInt(currTouchPoints.getAction() & MotionEvent.ACTION_MASK, 
                                                          data, caret);
       caret += 4;   // sizeof(int)

       if ((currTouchPoints.getAction() & MotionEvent.ACTION_MASK) == 
                                         MotionEvent.ACTION_POINTER_UP)
       {
          packInt((currTouchPoints.getAction() & 
                   MotionEvent.ACTION_POINTER_INDEX_MASK) 
                                    >> MotionEvent.ACTION_POINTER_INDEX_SHIFT, 
                   data, caret); 
          caret += 4;   // sizeof(int)

       } else {
          packInt(0, data, caret); 
          caret += 4;   // sizeof(int)
       }

       packInt(numPoints, data, caret);
       caret += 4;   // sizeof(int)

       float[] xs = currTouchPoints.getXs();
       float[] ys = currTouchPoints.getYs();
       int[] pids = currTouchPoints.getPointerIds();

       for (int i = 0; i < numPoints; i++)
       {
          packInt(pids[i], data, caret);
          caret += 4;   // sizeof(int)

          packFloat(xs[i], data, caret);
          caret += 4;   // sizeof(float)

          packFloat(ys[i], data, caret);
          caret += 4;   // sizeof(float)
       }
    } // end of packTouchState

   // ----------------------------------------------------------------
   private void addSnapshot(final String url)  {
      logConsole("adding snapshot " + url);
      addImageTab(url,
                  "snap" + imgTabNum++,
                  "Snap",
                  false);
   }   // end of addSnapshot

   // ----------------------------------------------------------------
   private void addImageTab(final String url, final String tabName,
                    final String indicatorName, final boolean allowRotate) 
   {
      // get the current count of tabs
      final int iCount = th.getTabWidget().getTabCount();
//      Log.d(tag, "adding tab " + iCount);

      // deal with where we are going to store sharing if they decide
      // to do it.
      if (!Environment.MEDIA_MOUNTED.equals(
                                   Environment.getExternalStorageState())) 
      {
         Log.d(tag,"External media not mounted and writeable");
      }
      String tmpJpgFileDir =
                 Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES) + "/VMDremote";
      (new File(tmpJpgFileDir)).mkdirs();
      final String tmpJpgFile = tmpJpgFileDir + "/" +
              (new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date())) 
              + "img.jpg";

      TabHost.TabSpec spec = th.newTabSpec(tabName).setIndicator(
         indicatorName ).setContent( new TabHost.TabContentFactory() {
            public View createTabContent(String tagX)
            {
              //View view =
              RelativeLayout rl = (RelativeLayout) 
                 LayoutInflater.from(VMDMobile.this).inflate(R.layout.snapshot,
                                                         null);
                 // Setup the view here
              SnapshotView ssv = (SnapshotView)
                                      rl.findViewById(R.id.snapshotImage);
              ssv.allowRotate(allowRotate);
              ssv.setImage(url);

              final Bitmap bmp = ssv.getBitmap();
            
              Button buttonShare = (Button) rl.findViewById(R.id.shareButton);
              buttonShare.setOnTouchListener(new OnTouchListener() {
                  @Override
                  public boolean onTouch(View v, MotionEvent event) {
                      if ( event.getAction() == (MotionEvent.ACTION_UP)) {
                         // save a JPG
                         try {
                            FileOutputStream out=new FileOutputStream(
                                                               tmpJpgFile);
                            Log.d("ShareButton", 
                                      "to '" + tmpJpgFile + "'" +
                                      ",bmp wid: " + bmp.getWidth() + 
                                      ",bmp hei: " + bmp.getHeight()  );
                            bmp.compress(
                                        Bitmap.CompressFormat.JPEG, 80, out);
                            out.flush();
                            out.close();

                         } catch (Exception e) {
                            Log.d("ShareButton", 
                                      "Couldn't write jpg: " + e.getMessage());
                         }

                         Intent share = new Intent(Intent.ACTION_SEND);
                         share.setType("image/jpeg");

                         share.putExtra(Intent.EXTRA_STREAM,
                                        Uri.parse("file://" + tmpJpgFile));

                         startActivity(Intent.createChooser(
                                         share, "Share Snapshot"));

//                         (new File(tmpJpgFile)).delete();
                      }
                     return false;
                  }
              }); // end of setOnTouchListener

              Button buttonClose = (Button) rl.findViewById(R.id.closeButton);
              buttonClose.setOnTouchListener(new OnTouchListener() {
     @Override
     public boolean onTouch(View v, MotionEvent event) {
         if ( event.getAction() == (MotionEvent.ACTION_UP)) {
           // let's see if we need to delete a share file
           if ((new File(tmpJpgFile)).exists()) {
             (new File(tmpJpgFile)).delete();
           }
           // close this tab.  Set 'current tab' back to 0
//           TabWidget tw = th.getTabWidget();
//           Log.d(tag, "action: " + event.getAction() + ",tab count: " + tw.getTabCount());

           // find the tabspec with a name of 'tabName'
           for (int i = 0; i < listOfTabs.size(); i++) {
              TabHost.TabSpec s = listOfTabs.get(i);
              if (s.getTag().equals(tabName)) {
                 listOfTabs.remove(i);
                 break;
              }
           }
           th.setCurrentTab(0);
           th.clearAllTabs();
           for (TabHost.TabSpec spec : listOfTabs) {
              th.addTab(spec);
           }
         }
         return false;
       }
     });  // end of setOnTouchListener

              return rl;
            } /* end of createTabContent() */ } /* end of TabContentFactory */
         );  // end of setContent()

      th.addTab(spec);
      listOfTabs.add(spec);
   }   // end of addImageTab

   // ----------------------------------------------------------------
   private void setupTabs()
   {
      th = getTabHost();

      TabHost.TabSpec spec = th.newTabSpec("touch").setIndicator(
         "TouchPad").setContent( new TabHost.TabContentFactory() {
            public View createTabContent(String tag)
            {
              //View view =
              mainLayout = (RelativeLayout) 
                 LayoutInflater.from(VMDMobile.this).inflate(R.layout.touchpad,
                                                         null);

              // Setup the view here
              return mainLayout;
            }}
         );

      th.addTab(spec);
      listOfTabs.add(spec);

/*
      th.setOnTabChangedListener(new TabHost.OnTabChangeListener() {
         @Override
         public void onTabChanged(String tabId) {
            Log.d(tag,"TAB id " + tabId);
         }
      });
*/


      th.setCurrentTab(0);
   } // end of setupTabs

   // ----------------------------------------------------------------
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
//        Log.d(tag, "onCreate start");
       // get reference to SensorManager

        // let's set the default server and ports
//        selectNewServer(DEFAULT_VMD_IP, DEFAULT_VMD_PORT, DEFAULT_LISTEN_PORT);

        setContentView(R.layout.main);

        setupTabs();

        sm = (SensorManager) getSystemService(Context.SENSOR_SERVICE);

        i = 0;

        orientation = new float[3];
        accel = new float[3];
        gyroscope = new float[3];
        magField = new float[3];

        Rmat = new float[9];
        Imat = new float[9];
        data = new byte[MAX_PACKET_SIZE];
        heartBeatData = new byte[MAX_PACKET_SIZE];
        buttonData = new byte[MAX_PACKET_SIZE];
        commandData = new byte[MAX_PACKET_SIZE];

        console = (TextView) findViewById(R.id.console);
        ipInfo = (TextView) findViewById(R.id.ipinfo);

//        mainLayout = (RelativeLayout) findViewById(R.id.mainLayout);
        mainLayout.setBackgroundColor(notConnectedColor);

        mPrefs = getSharedPreferences(prefFile, MODE_PRIVATE);

        m_htButtonList = new Hashtable();
//        hasMulti = hasMultiTouch();

        configureGUIitems();
        
// --------------------

        // keep the screen on while this application is in the foreground
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);
        screenWidth = metrics.widthPixels;
        screenHeight = metrics.heightPixels;
        xdpi = metrics.xdpi;
        ydpi = metrics.ydpi;


        // let's set the server to the last one used...
        SharedPreferences serverList =
               getSharedPreferences(ServerChooser.SERVER_NAMES, MODE_PRIVATE);
        if (serverList.getInt("NumServers", 0) > 0) {
           selectNewServer(serverList.getString("server0", DEFAULT_VMD_IP),
                     serverList.getInt("port0", DEFAULT_VMD_PORT),
                     serverList.getInt("listen0", DEFAULT_LISTEN_PORT));
        }

        Toast.makeText(getApplicationContext(), R.string.welcomeMsg,
                                                   Toast.LENGTH_LONG).show();
    } // end of onCreate()

// -----------------------------------------------------------------
//    private boolean hasMultiTouch()
//    {
 //      try {
  //      MotionEvent.class.getMethod("getPointerCount", new Class[] {});
   //     MotionEvent.class.getMethod( "getPointerId", new Class[] { int.class });
    //    MotionEvent.class.getMethod( "getX", new Class[] { int.class });
     //   MotionEvent.class.getMethod( "getY", new Class[] { int.class });
//        return true;
 //      } catch (NoSuchMethodException nsme) {
  //      return false;
   //    }
//    }

// -----------------------------------------------------------------
    private void configureButtonBar(final String strGroup)
    {
        LinearLayout lay = (LinearLayout) findViewById(R.id.buttonBar);

        lay.removeAllViews();

        Button [] buttonArray = getButtonArrayFromList(strGroup);
        if (buttonArray== null) {
           buttonArray = getAuxButtonArray();
        } 

        for (int i=0; i < buttonArray.length; i++) {
           lay.addView(buttonArray[i]);
        }

    } // end of configureButtonBar

// -----------------------------------------------------------------
    private Button [] getButtonArrayFromList(final String strGroup)
    {
       Button [] buttonArray;
       Vector v = (Vector) m_htButtonList.get(strGroup);

       if (v != null && v.size() > 0)
       {
          m_strCurrentButtonGroup = strGroup;
          buttonArray = new Button[v.size()];

          for (int i=0; i < v.size(); i++)
          {
             Tuple t = (Tuple) v.get(i);

             Button button = new Button(this); button.setText(t.getKey());
             final String cmd = t.getValue();
             button.setOnTouchListener(new OnTouchListener() {
                @Override
                public boolean onTouch(View v, MotionEvent event) {
                   if ( event.getAction() == (MotionEvent.ACTION_UP)) {
                      sendSpecificCommand(COMMAND_HASH, cmd);
                   }
                   return false;
                }
             });
             buttonArray[i] = button;
          }
       } else { return null; }
       return buttonArray;
    }

// -----------------------------------------------------------------
    private Button [] getAuxButtonArray()
    {
       final int howMany = 4;
        Button [] buttonArray = new Button[howMany];

        for (int i=0; i < howMany; i++)
        {
           final int iCount = i;
           Button button = new Button(this); button.setText("Aux-" + i);
           button.setOnTouchListener(new OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if ( event.getAction() == (MotionEvent.ACTION_UP)) {
                   setButtonState(iCount, true);
                } else {
                   setButtonState(iCount, false);
                }
               return false;
            }
           });
           buttonArray[i] = button;
        }

       return buttonArray;
    }

// -----------------------------------------------------------------
    private void configureGUIitems()
    {
       configureButtonBar("Default Aux");

// ----------------------------------------
        touchPad = (MultiTouchPad) findViewById(R.id.touchPad);
        touchPad.registerApp(this);
/*
        touchPad.setOnTouchListener(new OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
               return touchPadActivity(event);
            }
        });
        */

    } // end of configureGUIitems

   // ----------------------------------------------------------------
   public boolean sendingTouch() { return sendingTouch; }
   // ----------------------------------------------------------------
   public void sendDisconnectMsg() {
      try {
//         logConsole("sendDisconnectMsg().start");
         packInt(DISCONNECT_MSG, data, headerLength);

         packInt(buttonState, data, headerLength+4);
         packInt(sequenceNumber++, data, headerLength+8 );
         
         packAndSend(data, DISCONNECT_DATA_SIZE);
      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
         logConsole( errstr.toString());
//         Log.d(tag, e.toString());
      }
   }
   // ----------------------------------------------------------------
   public void sendConnectMsg() {
      try {
         packInt(CONNECT_MSG, data, headerLength);
         packInt(buttonState, data, headerLength+4);
         packInt(sequenceNumber++, data, headerLength+8 );

         packAndSend(data, CONNECT_DATA_SIZE);
         requestButtonConfig();
      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
         logConsole( errstr.toString());
//         Log.d(tag, e.toString());
      }
   }

   // ----------------------------------------------------------------
   public void requestButtonConfig() {
      sendSpecificCommand(COMMAND_REQUEST_BUTTONCONFIG, "");
   }
   // ----------------------------------------------------------------
   public void sendHeartBeat() {
      try {
         // the HEARTBEAT msg identifier has already been packed
         // at position 'headerLength' of heartBeatData
//         packInt(buttonState, heartBeatData, headerLength+4);
//         packInt(sequenceNumber++, data, headerLength+8 );
         packTouchState(oldTouchPoints, touchPad.getWidth(), touchPad.getHeight());
         packInt(HEART_BEAT, data, headerLength);

         packAndSend(data, TOUCH_DATA_SIZE);
//         packAndSend(heartBeatData, HEARTBEAT_DATA_SIZE);
      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
         logConsole( errstr.toString());
//         Log.d(tag, e.toString());
      }
   }

   // ----------------------------------------------------------------
   // xxx: can see us wanting to send data parameters with a message
   // at some point in the future.
   public void sendSpecificCommand(final int cmd, final String msg) {
//      Log.d(tag, "Sending command " + cmd + " w msg " + msg);
      try {
         // the COMMAND_EVENT msg identifier has already been packed
         // at position 'headerLength' of commandData
         packInt(cmd, commandData, headerLength+4);
         packInt(sequenceNumber++, commandData, headerLength+8 );
         packInt(msg.length(), commandData, headerLength+12 );
         packString( msg, msg.length(), commandData, headerLength+16);

//         Log.d(tag, "Packed '" + cmd + "', seqNum: '" + sequenceNumber +
//                    "', length: '" + msg.length() + "', msg: '" + msg + "'");

         // I added the 4 bytes to make sure that we capture the rest
         // of the 4byte packet that the string is contained in.
         packAndSend(commandData, COMMAND_DATA_SIZE+msg.length() + 4);
      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
         logConsole( errstr.toString());
      }
   }

   // ----------------------------------------------------------------
   public void sendButtonState() {
      try {
         // the BUTTON_EVENT msg identifier has already been packed
         // at position 'headerLength' of buttonData
         packInt(buttonState, buttonData, headerLength+4);
         packInt(sequenceNumber++, buttonData, headerLength+8 );
         packAndSend(buttonData, BUTTON_DATA_SIZE);
      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
         logConsole( errstr.toString());
//         Log.d(tag, e.toString());
      }
   }
   // ----------------------------------------------------------------
   public boolean sendTouchEvent(PointInfo currTouchPoints)
   {
      try {

         packTouchState(currTouchPoints, touchPad.getWidth(), touchPad.getHeight());
         
         packAndSend(data, TOUCH_DATA_SIZE);
         if (++i%25 == 0)
         {
              int numPoints = Math.min(touchPointsSupported, currTouchPoints.getNumTouchPoints());
              logConsole("i:" + i + ",numPoints:" + 
                                   String.format("%4d", numPoints));
         }

      } catch (Exception e) {
         StringWriter errstr = new StringWriter();
         PrintWriter pw = new PrintWriter(errstr);
         e.printStackTrace(pw);
//         Log.d(tag, e.toString());

         logConsole( errstr.toString());
      }
      return false;

   }

   // ----------------------------------------------------------------
   public boolean registerTouch(PointInfo currTouchPoints)
   {
      oldTouchPoints = currTouchPoints;
      if (!sendingTouch) return true;
      return sendTouchEvent(currTouchPoints);
   }

   // ----------------------------------------------------------------
    public void onSensorChanged(SensorEvent event) {
//       int sensor = event.sensor.getType();
       if (!sendingOrient) return;
       try {
        synchronized (this) {
           i++;


//            if (sensor == SensorManager.SENSOR_ORIENTATION) {
//               System.arraycopy(event.values,0,orientation,0,3);
//            }
//            else 
            switch (event.sensor.getType())
            {
               case Sensor.TYPE_ACCELEROMETER:
                  System.arraycopy(event.values,0,accel,0,3);
                  break;
               case Sensor.TYPE_ORIENTATION:
                  System.arraycopy(event.values,0,orientation,0,3);
                  break;
               case Sensor.TYPE_GYROSCOPE:
                  System.arraycopy(event.values,0,orientation,0,3);
                  break;
               case Sensor.TYPE_MAGNETIC_FIELD:
                  System.arraycopy(event.values,0,magField,0,3);
                  break;
            }

//            if (sensor == Sensor.TYPE_ACCELEROMETER) {
//               System.arraycopy(event.values,0,accel,0,3);
//            }            
//            else if (sensor == Sensor.TYPE_MAGNETIC_FIELD) {
//               System.arraycopy(event.values,0,magField,0,3);
//            }            

            // rotation matrix, 
            // http://developer.android.com/reference/android/hardware/SensorManager.html#getRotationMatrix(float[],%20float[],%20float[],%20float[])
            SensorManager.getRotationMatrix(Rmat, Imat, accel, magField);

//            SensorManager.getOrientation(Rmat, orientation);

           if (i%100 == 0)
           {
//              float r2d = (float)(180.0f/Math.PI);
//              logConsole("o:" + (int)(r2d*orientation[1]) + "," +
//                                   (int)(r2d*orientation[2]) + "," +
//                                   (int)(r2d*orientation[0]));
              logConsole("i:" + i + ",o:" + 
                                   String.format("%.2f", orientation[1]) + "," +
                                   String.format("%.2f", orientation[2]) + "," +
                                   String.format("%.2f", orientation[0]));

           }

           packDeviceState();
           packAndSend(data, SENSOR_DATA_SIZE);

        }
       } catch (Exception e) {
          StringWriter errstr = new StringWriter();
          PrintWriter pw = new PrintWriter(errstr);
          e.printStackTrace(pw);

          logConsole("" + accel[0] + "," +
                               accel[1] + "," +
                               accel[2] + "," + errstr.toString());
//          Log.d(tag, "onSensorChanged:" + e.toString());
       }
    } // onSensorChanged(SensorEvent event) {
    
   // ----------------------------------------------------------------
    public void onAccuracyChanged(int sensor, int accuracy) {
//      Log.d(tag,"onAccuracyChanged: " + sensor + ", accuracy: " + accuracy);
    }
    @Override public void onAccuracyChanged(Sensor sensor, int accuracy) {
    }
   // ----------------------------------------------------------------
    public void logConsole(final String str) {
       Log.d(tag, "Logging " + str);
       if (mPrefs.getBoolean("showDebug",false)) console.setText(str);
    }

   // ----------------------------------------------------------------
   // figure out which version of the API we want to support
   private void configureApiVersion() 
   {
      String strApiVersion = mPrefs.getString("apiVersion", "Latest");
      if (strApiVersion.equals("Latest")) {
         m_iApiVersion = API_VERSION;
      } else {
         int i = Integer.parseInt(strApiVersion);
         switch (i) {
            case 9:
            case 8:
               m_iApiVersion = i;
               logConsole("Sending API Version " + i);
               break;
            default:
               m_iApiVersion = API_VERSION;
               logConsole("Don't support API Version " + i + ". Using Latest");
               break;
         }
      }
   } // end of configureApiVersion

   // ----------------------------------------------------------------
   // configure all of the data packets, based on the API version
   private void setDataPacketSizes() 
   {
      switch (m_iApiVersion) {
         case 9: headerLength = 40; break;
         case 8: headerLength = 24; break;
      }
         
      sensorPayloadLength = 36 + 36;
      touchPayloadLength = 40;
      touchPointsSupported = 5;
    
      BUTTON_DATA_SIZE = headerLength + 12;
      COMMAND_DATA_SIZE = headerLength + 16;
      HEARTBEAT_DATA_SIZE = headerLength + 12;
      CONNECT_DATA_SIZE = headerLength + 12;
      DISCONNECT_DATA_SIZE = headerLength + 12;
      SENSOR_DATA_SIZE = headerLength + sensorPayloadLength;
      TOUCH_DATA_SIZE = headerLength + touchPayloadLength
                                                  + touchPointsSupported * 12;
   } // end of setDataPacketSizes

   // ----------------------------------------------------------------
   private void startServerChooserIntent()
   {
      Intent i = new Intent(this, ServerChooser.class);
      // not actually sending data to serverchooser right now
//      Bundle b = new Bundle();
//      b.put("app", this);
//      i.putExtras(b);
      startActivityForResult(i,ACTIVITY_SERVER_CHOOSER);
   }

   // ----------------------------------------------------------------
   protected void onActivityResult(int requestCode, int resultCode, Intent data)
   {
      logConsole("onActivityResult.  resultCode=" + resultCode + 
               ", requestCode=" + requestCode);
      
      if (resultCode == Activity.RESULT_CANCELED) {  // user hit the back button
         m_bServerChooserCanceled = true;
      }

      if (resultCode == Activity.RESULT_OK && requestCode == ACTIVITY_SERVER_CHOOSER)
      {
         // can we just call selectNewServer here?
         if (data != null && data.hasExtra("serverName")) {
            selectNewServer(data.getStringExtra("serverName"),
                            data.getIntExtra("port", DEFAULT_VMD_PORT),
                            data.getIntExtra("listenPort",DEFAULT_LISTEN_PORT));
//            selectNewServer(data.getExtras().getString("serverName"),
//                            data.getExtras().getInt("port"),
//                            data.getExtras().getInt("listenPort"));
         }
      }
   }

   // ----------------------------------------------------------------
   public void selectNewServer(final String name, final int port,
                               final int listenPort)
   {
      // were we set to something else?
      if (m_bServerSet)
      {  // we need to shut down everything..
         stopAllSending();
      }
      if (skt != null && !skt.isClosed())
      {
         skt.close();
      }
      skt = null;

      logConsole("selectNewServer: name:" + name + ", port: " + port +  
                 ", listen: " + listenPort);
      m_bServerSet = true;
      m_strCurrentServer = name;  // ipEntry; DEFAULT_VMD_IP
      m_iCurrentPort = port;    // portEntry ; DEFAULT_VMD_PORT
      m_iCurrentListenPort = listenPort; // listenportEntry; DEF_LISTEN_PRT

   }

   // ----------------------------------------------------------------
    @Override
    protected void onResume() {
        super.onResume();
        console.setText("");
        Log.d(tag, "onResume start");
        if (m_udpServer != null) m_udpServer.setContinue(false);
        m_udpServer  = new UDPServer(this, m_iCurrentListenPort);
        m_udpServer.start();

//        logConsole("Waking up. slept " + (System.currentTimeMillis() -
//                                               lastTime)/1000.0);

        ConnectivityManager connManager = (ConnectivityManager)
                               getSystemService(CONNECTIVITY_SERVICE);
        NetworkInfo mWifi = connManager.getNetworkInfo(
                             ConnectivityManager.TYPE_WIFI);

//        configureApiVersion();
        setDataPacketSizes();

        initialPack(data);
        initialPack(heartBeatData);
        initialPack(buttonData);
        initialPack(commandData);
        packInt(HEART_BEAT, heartBeatData, headerLength);
        packInt(BUTTON_EVENT, buttonData, headerLength);
        packInt(COMMAND_EVENT, commandData, headerLength);

        // foobar
        if (!m_bServerSet) {
          logConsole("Server hasn't been set yet");
          if (!m_bServerChooserCanceled)
          {
             m_bServerChooserCanceled = false;
             startServerChooserIntent();
          } else {
             ipInfo.setText("Server Selection canceled.");

          }
        } else {
           ipInfo.setText("VMD Server: " + 
               m_strCurrentServer + ":" + m_iCurrentPort);
           startSending();
        }
        if (!(mWifi.isConnected())) {
           ipInfo.setText(ipInfo.getText() + "; WI-FI is not connected.  Please connect.");
        }

    }
    
   // ----------------------------------------------------------------
    @Override
    public void onConfigurationChanged(Configuration nc) {
//       logConsole("configuration has changed");
       super.onConfigurationChanged(nc);
       DisplayMetrics metrics = new DisplayMetrics();
       getWindowManager().getDefaultDisplay().getMetrics(metrics);
       screenWidth = metrics.widthPixels;
       screenHeight = metrics.heightPixels;
       xdpi = metrics.xdpi;
       ydpi = metrics.ydpi;

    }    
   // ----------------------------------------------------------------
    @Override
    protected void onDestroy() {
        Log.d(tag, "onDestroy start");
       lastTime = System.currentTimeMillis();
       stopAllSending();
       super.onDestroy();
    }    
   // ----------------------------
    @Override
    protected void onPause() {
        Log.d(tag, "onPause start");
       lastTime = System.currentTimeMillis();
       stopAllSending();
        super.onPause();
    }    
   // -----------------------------
    @Override
    protected void onStop() {
        Log.d(tag, "onStop start");
       lastTime = System.currentTimeMillis();
       stopAllSending();
       super.onStop();
    }    
   // ----------------------------------------------------------------
   @Override
   public boolean onCreateOptionsMenu(Menu menu){
//      logConsole("Menu has been selected");
      MenuInflater inflater = getMenuInflater();
      inflater.inflate(R.menu.menu, menu);
      return true;
   }

   // ----------------------------------------------------------------
   @Override
   public boolean onPrepareOptionsMenu(Menu menu){

/*
      if (sendingTouch)
      {
         menu.findItem(R.id.touchPadToggle).setIcon(R.drawable.ic_menu_stop);
         menu.findItem(R.id.touchPadToggle).setTitle(R.string.sendTouchStop);
      } else {
         menu.findItem(R.id.touchPadToggle).setIcon(R.drawable.ic_menu_play_clip);
         menu.findItem(R.id.touchPadToggle).setTitle(R.string.sendTouchStart);
      }
*/
      return super.onPrepareOptionsMenu(menu);
   }

   // ----------------------------------------------------------------
   @Override
   public boolean onOptionsItemSelected(MenuItem item) {
      // Handle item selection
      switch (item.getItemId()) {
      case R.id.chooseServer:
         startServerChooserIntent();
         return true;
      case R.id.settings:
         startActivity(new Intent(this, MainPreferences.class));
         return true;
/*
      case R.id.touchPadToggle:
         if (sendingTouch)  // then we need to turn it off
         {
            sendingTouch = false;
         } else {           // then we need to turn it on
            sendingTouch = true;
         }

         // look at current state, change icon, title, actually do it
         return true;
*/
      case R.id.takeSnapshot:
         // xxx: send the command to take a snapshot
         sendSpecificCommand(COMMAND_TAKESNAPSHOT, "");

         return true;
      case R.id.chooseButtonList:
         // we need to launch the popup.
         CharSequence [] it = new CharSequence[m_htButtonList.size() + 1 ];
         it[0] = "Default Aux";
         Enumeration e = m_htButtonList.keys();
         int i=1;
         while (e.hasMoreElements()) {
            it[i++] = (String) e.nextElement();
         }

         final CharSequence[] items = it;

         AlertDialog.Builder builder = new AlertDialog.Builder(this);
         builder.setTitle(R.string.buttonChooserMenu);
         builder.setItems(items, new DialogInterface.OnClickListener() {
              public void onClick(DialogInterface dialog, int item) {
                       // let's redraw the buttons.. even if they already
                       // have this button set selected.  This lets them
                       // manually update an group.

                       configureButtonBar(items[item].toString());
                       Toast.makeText(getApplicationContext(), items[item],
                                                   Toast.LENGTH_SHORT).show();
              }
         });
         AlertDialog alert = builder.create();
         alert.show();

         return true;
      case R.id.resetVMDView:
         // send reset view event
         sendSpecificCommand(COMMAND_RESETVIEW, "");
//         setButtonState(31,true);
//         setButtonState(31,false);
         return true;
//      case R.id.exit:
//         logConsole("quit chosen");
//         return true;
      default:
         logConsole("item " + item.getItemId() + "chosen." 
	 );
//	 + R.id.settings + "," + R.id.exit);
         return super.onOptionsItemSelected(item);
      }
   }

   // ----------------------------------------------------------------
   // ---------------- HELPER FCTS -----------------------------------
   // ----------------------------------------------------------------
    private void packInt(final int src, byte [] dest, int offset)
    {
       dest[offset  ] = (byte)((src>>24) & 0x0ff);
       dest[offset+1] = (byte)((src>>16) & 0x0ff);
       dest[offset+2] = (byte)((src>> 8) & 0x0ff);
       dest[offset+3] = (byte)((src    ) & 0x0ff);
    }

   // ----------------------------------------------------------------
    private void packString(final String src, final int maxLen, 
                                         byte [] dest, int offset)
    {  
       // since this is getting byte-order translated, we have to 
       // tweak the ordering.  The string
       //     0123456789
       // needs to look like:
       //     32107654..98
       // (where . is null)

       // we want to only copy the first maxLen characters of the
       // string, or the entire string.. whichever one is shorter.
       int n = Math.min(maxLen, src.length()) ; 
       for(int i = 0 ; i < n ; i++) {
//          dest[offset + i] = (byte) src.charAt(i);
          dest[offset + 3-i%4 + 4* (i/4)] = (byte) src.charAt(i);
       }

       // can't just put values in from n to maxLen as is
       // demonstrated by the above..  there needs to be two
       // zeros inserted between the 54 and 89.  So, we have
       // to do the same translation as for the data itself
       for (int i=n; n < maxLen; n++) {
          dest[offset + 3-i%4 + 4* (i/4)] = (byte) 0;
       }
    }

   // ----------------------------------------------------------------
    private void packFloat(final float src, byte [] dest, int offset)
    {
       packInt(Float.floatToRawIntBits(src), dest, offset);
    }

   // ----------------------------------------------------------------
    private void packFloatArray(float[] src, byte [] dest, int offset)
    {
       for (int i=0; i < src.length; i++)
       {
          packFloat(src[i], dest, offset+i*4);
       }
    }

   // ----------------------------------------------------------------
    // put headers in data packet that won't change
    private void initialPack(byte [] array)
    {
       // Endian check
       packInt(1, array, 0);

       // put API version at the beginning
       packInt(m_iApiVersion, array, 4);

       // send identifier to identify this phone.  Could be 
       // stored in a database in VMD so that VMD automatically
       // knows a user for future invocations
       packString( mPrefs.getString("nickEntry","Anonymous User"), 16, array, 8);
       if (m_iApiVersion >= 9)
       {
          // tell the server what port we are listening on
          packInt( m_iCurrentListenPort , array, 24);
          // tell the server what scaling factors we are using
          packFloat( Float.parseFloat(mPrefs.getString("rotScale",
                       "" + DEFAULT_ROTSCALE_FACTOR)), array, 28);
          packFloat( Float.parseFloat(mPrefs.getString("zoomScale",
                       "" + DEFAULT_ZOOMSCALE_FACTOR)), array, 32);
          packFloat( Float.parseFloat(mPrefs.getString("tranScale",
                       "" + DEFAULT_TRANSCALE_FACTOR)), array, 36);
       }

    } // end of initialPack

   // ----------------------------------------------------------------
   // ----------------------------------------------------------------
   class SendPacket extends AsyncTask<String, Void, Object>
   {
      protected Object doInBackground(String... parms)
      {
         try {
            if (skt != null && !skt.isClosed()) skt.send(pkt);
         } catch (Exception e) {
            logConsole("SendPacket exception:" + e);
         }
         return null;
      }
      protected void doPostExecute(Object o) {
         m_bInSendPacket = false;
      }
   }
   // ----------------------------------------------------------------
   // ----------------------------------------------------------------
    private void packAndSend(final byte [] data, final int length) 
                   throws Exception
    {
       synchronized (this) {
          if (pkt != null) {
             pkt.setData(data);
             pkt.setLength(length);
             m_bInSendPacket = true;
             new SendPacket().execute();
//             if (skt != null && !skt.isClosed()) skt.send(pkt);
          }
       }
    }

   // ----------------------------------------------------------------
   // ----------------------------------------------------------------
   public class Tuple {
      private String key, value;
      public Tuple(final String k, final String v) {
         key = k; value = v;
      }
      public boolean equals(final Tuple t) {
         return (key.equals(t.getKey()) && value.equals(t.getValue()));
      }
      public String getKey() { return key; }
      public String getValue() { return value; }
      public String toString() { return key + "=" + value; }
   }
   // ----------------------------------------------------------------
   // ----------------------------------------------------------------
} // end of vmdMobile class



