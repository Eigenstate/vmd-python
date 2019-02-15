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
 *	$RCSfile: P_CmdTool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.49 $	$Date: 2019/01/17 21:21:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * Commands and the text interface.
 *
 ***************************************************************************/


#include "P_CmdTool.h"
#include <stdlib.h>
#include <string.h>

CmdToolCreate::CmdToolCreate(const char *thetype, int theargc, const char **theUSL)
    : Command(Command::TOOL_CREATE) {
  int i;
  argc=theargc;
  USL = new char *[argc];
  for(i=0;i<argc;i++)
    USL[i]=strdup(theUSL[i]);
  type = strdup(thetype);
}

CmdToolCreate::~CmdToolCreate() {
  int i;
  for(i=0;i<argc;i++) free(USL[i]);
  delete [] USL;
  free(type);
}

void CmdToolCreate::create_text() {
  int i;
  *cmdText << "tool create " << type;
  for(i=0;i<argc;i++) *cmdText << " " << USL[i];
  *cmdText << ends;
}

CmdToolChange::CmdToolChange(const char *thetype, int thenum)
    : Command(Command::TOOL_CHANGE) {
  num = thenum;
  type = strdup(thetype);
}

CmdToolChange::~CmdToolChange() {
  free(type);
}

void CmdToolChange::create_text() {
  *cmdText << "tool change " << type << " " << num << ends;
}

CmdToolDelete::CmdToolDelete(int thenum)
  : Command(Command::TOOL_DELETE) {
  num=thenum;
}

void CmdToolDelete::create_text() {
  *cmdText << "tool delete " << num;
}

CmdToolScale::CmdToolScale(float thescale, int thenum)
  : Command(Command::TOOL_SCALE) {
  num=thenum;
  scale=thescale;
}

void CmdToolScale::create_text() {
  *cmdText << "tool scale " << scale << " " << num;
}

CmdToolScaleForce::CmdToolScaleForce(float thescale, int thenum)
  : Command(Command::TOOL_SCALE_FORCE) {
  num=thenum;
  scale=thescale;
}

void CmdToolScaleForce::create_text() {
  *cmdText << "tool scaleforce " << scale << " " << num;
}

CmdToolScaleSpring::CmdToolScaleSpring(float thescale, int thenum)
  : Command(Command::TOOL_SCALE_SPRING) {
  num=thenum;
  scale=thescale;
}

void CmdToolScaleSpring::create_text() {
  *cmdText << "tool scalespring " << scale << " " << num;
}

CmdToolOffset::CmdToolOffset(float *theoffset, int thenum)
  : Command(Command::TOOL_OFFSET) {
  int i;
  num=thenum;
  for(i=0;i<3;i++)
    offset[i] = theoffset[i];
}

void CmdToolOffset::create_text() {
  *cmdText << "tool offset "
	   << offset[0] << " " << offset[1] << " " << offset[2] << " "
	   << num;
}

CmdToolAddDevice::CmdToolAddDevice(const char *thename, int thenum)
  : Command(Command::TOOL_ADD_DEVICE) {
  name = strdup(thename);
  num = thenum;
}

CmdToolAddDevice::~CmdToolAddDevice() {
  free(name);
}

void CmdToolAddDevice::create_text() {
  *cmdText << "tool adddevice " << name << " " << num;
}

CmdToolDeleteDevice::CmdToolDeleteDevice(const char *thename,
					 int thenum)
  : Command(Command::TOOL_DELETE_DEVICE) {
  name = strdup(thename);
  num = thenum;
}

CmdToolDeleteDevice::~CmdToolDeleteDevice() {
  free(name);
}

void CmdToolDeleteDevice::create_text() {
  *cmdText << "tool removedevice " << name << " " << num;
}

CmdToolRep::CmdToolRep(int tool, int mol, int rep) 
: Command(Command::TOOL_REP), toolnum(tool), molid(mol), repnum(rep) {}

void CmdToolRep::create_text() {
  *cmdText << "tool rep " << toolnum << " " << molid << " " << repnum;
}
   
CmdToolCallback::CmdToolCallback(int the_on)
  : Command(Command::TOOL_CALLBACK) {
  on = the_on;
}

void CmdToolCallback::create_text() {
  if(on) {
    *cmdText << "tool callback on";
  } else {
    *cmdText << "tool callback off";
  }
}
