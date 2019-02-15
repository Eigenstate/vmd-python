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
 *	$RCSfile: CommandQueue.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.42 $	$Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * This stores all the Commands to be run in a queue.  The idea is that
 * the various events add Commands to the command queue then they
 * are read off the queue and the UIs are notified.
 *
 * There is one global instance of this class in VMDApp, called commandQueue.
 * It is used by all the user interface objects (UIObject classes).
 * Each time a new action is requested by the user or some other part of
 * VMD, a new Command instance is created and added to the
 * CommandQueue.  Within the main event loop of VMD, after each
 * UIObject is checked for new events, the commands in the queue are
 * all executed until the queue is empty (since the execution of one command
 * may result in the queuing of a new command, this process continues until
 * the queue is empty).
 *
 * NOTES:
 *  1) To add new commands to queue, use routine 'append(Command *)',
 *     inherited since this is a ResizeArray<> object.
 *  2) To do something, use 'execute' routine. 
 *     This will execute the top command and inform all the UIs 
 *  3) 'execute_all' will do all the commands until the queue is empty.
 *
 ***************************************************************************/
#ifndef COMMANDQUEUE_H
#define COMMANDQUEUE_H

#include <stdio.h>
#include <string.h>
#include "ResizeArray.h"
#include "Command.h"
class UIObject;

/// Stores Commands to be run in a queue, notifies UIs when they are run
class CommandQueue {
  private:
    ResizeArray<Command *> cmdlist; ///< the command list itself
    ResizeArray<UIObject *> uilist; ///< the list of UIObjects

  public:
    CommandQueue(void);             ///< constructor
    ~CommandQueue(void);            ///< destructor
    void register_UI(UIObject *);   ///< add a new UIObject
    void unregister_UI(UIObject *); ///< remove a UIObject
    void append(Command *);         ///< enqueue a command, does not execute
    void runcommand(Command *);     ///< run a new command
    void execute_all(void);         ///< execute commands until queue is empty
    void check_events();            ///< Have registered UI's check for events
};

#endif

