/***************************************************************************
 *cr
 *cr            (C) Copyright 2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

package edu.uiuc.ks.vmdmobile;

import java.io.*;
import java.net.*;

import android.util.Log;

/** This class just sends a message every X milliseconds
 * @author <a href="http://www.ks.uiuc.edu/~kvandivo/">Kirby Vandivort</a>
 * @version $Revision: 1.5 $ $Date: 2013/03/14 20:29:35 $ $Author: kvandivo $
 */
public class UDPServer extends Thread
{

// --------------------------------------------------------------------------   
   /** constructor
    * @param mob callback for owner
    */
   public UDPServer(VMDMobile mob, final int port)
   {
      lastContact = System.currentTimeMillis();
      m_iPort = port;
      try {
         skt = new DatagramSocket(m_iPort);
//         Log.d("UDPServer", "constructor()");
      } catch (SocketException se) {
         Log.d("UDPServer", "socket exception:" + se);
      }
      m_mob = mob;
//      setDaemon(true);
   } // end of constructor

// --------------------------------------------------------------------------   
   public void setContinue(final boolean kb)
   {
//      Log.d("UDPServer", "setting continue to " + kb);
      m_bContinue = kb;
      if (!kb) {
         try {
            interrupt();
         } catch(SecurityException e) { /* don't care */ }
      }
   }

// --------------------------------------------------------------------------   
   public void checkTimeout()
   {
      if (isConnected && (System.currentTimeMillis() - lastContact > TIMEOUT)) {
//         Log.d("UDPServer", "I think we lost the server");
         isConnected = false;

         // tell the main program that we think the server is gone
         // (for now, at least)
         m_mob.runOnUiThread(new Runnable() { public void run() {
            m_mob.vmdServerGone();
         } });
      }

   }
// --------------------------------------------------------------------------   
   public void run()
   {
      // size corresponds to buffer specified in MobileInterface.h on
      // the VMD side
      byte [] data = new byte[1536];

      setContinue(true);
//      Log.d("UDPServer", "run(): starting. port: " + m_iPort);
      while (m_bContinue)  
      {
         checkTimeout();
         DatagramPacket packet = new DatagramPacket(data, data.length);
         try {
            if (skt == null) {
               sleep(3000);
               skt = new DatagramSocket(m_iPort);
            }
            if (skt != null)
            {
               boolean to = false;
               skt.setSoTimeout(1000);
               try {
                  skt.receive(packet);
               } catch (SocketTimeoutException e) {
                  to = true;
               }
//               Log.d("UDPServer", "run(): received packet");

              if (!to)
              {
//                 Log.d("UDPServer", "run(): msg from server");
                 isConnected = true;
                 lastContact = System.currentTimeMillis();
                 // the important part of the next lines:  the method call
                 // to packetReceived.  
                 // Only the UI threads are allowed to updated the UI, and this
                 // thread isn't a UI thread. The packetReceived method might 
                 // update the UI, so
                 // extra coding has to be done.  Only 'final' vars can
                 // be passed in, and you have to use this verbage to actually 
                 // have the UI thread do the work, and not this one.
                 final int pcktLength = packet.getLength();
                 final byte [] dataF = packet.getData();
                 final String addrF = packet.getAddress().getHostAddress();
                 m_mob.runOnUiThread(new Runnable() { public void run() {
                    m_mob.packetReceived(dataF, pcktLength, addrF);
                 } });
              }
            }

         } catch (Exception e) {
            Log.d("UDPServer", "run(): Exception" + e);
         }

      } // end of while on continuing

      if (skt != null && !skt.isClosed())
      {
         skt.disconnect();
         skt.close();
      }
//      Log.d("UDPServer", "run(): Terminating" );

   } // end of run

   protected void finalize() {
      if (skt != null && !skt.isClosed())
      {
         skt.disconnect();
         skt.close();
      }
   }

   private VMDMobile m_mob;

   private boolean m_bContinue;
   private int  m_iPort;

   private DatagramSocket skt;
   private long lastContact;

   private final long TIMEOUT = 10 * 1000;   // 10 seconds
   private boolean isConnected;

} // end of UDPServer class



