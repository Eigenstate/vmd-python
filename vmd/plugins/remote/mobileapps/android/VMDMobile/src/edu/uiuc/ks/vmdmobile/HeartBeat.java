/***************************************************************************
 *cr
 *cr            (C) Copyright 2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

package edu.uiuc.ks.vmdmobile;

/** This class just sends a message every X milliseconds
 * @author <a href="http://www.ks.uiuc.edu/~kvandivo/">Kirby Vandivort</a>
 * @version $Revision: 1.2 $ $Date: 2012/03/17 20:38:16 $ $Author: kvandivo $
 */
public class HeartBeat extends Thread
{

// --------------------------------------------------------------------------   
   /** constructor
    * @param kiMilliseconds time in milliseconds that we wait between sends
    * @param mn class that we are supposed to notify 
    */
   public HeartBeat(final int kiMilliseconds, VMDMobile mn)
   {
      m_iInterval = kiMilliseconds;
      m_mn = mn;
      setDaemon(true);

   } // end of constructor

// --------------------------------------------------------------------------   
   public void setContinue(final boolean kb)
   {
      m_bContinue = kb;
   }

// --------------------------------------------------------------------------   
   public void run()
   {
      setContinue(true);
      while (m_bContinue)  // since it is a daemon thread, it will automagically
      {                    // die when the app does
         try { 
            sleep(m_iInterval); 
            m_mn.sendHeartBeat();
            if (m_iCounter++%20 == 5) {     // send this out 1/20th as often
               m_mn.requestButtonConfig();
            }
         }
         catch (InterruptedException e) {
            setContinue(false);
         }

      } // end of while on continuing
   } // end of run()

   private int m_iCounter=0;
   private int m_iInterval;

   private VMDMobile m_mn;

   private boolean m_bContinue;

} // end of HeartBeat class



