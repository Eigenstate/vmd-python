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
 *      $RCSfile: CmdLabel.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.70 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Command objects used to create, list, delete, or graph labels for measuring
 * geometries.
 *
 ***************************************************************************/

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "CmdLabel.h"
#include "utilities.h"
#include "Inform.h"

// The following uses the Cmdtypes LABEL_ADD, LABEL_SHOW, LABEL_LIST,
// and LABEL_DELETE from the Command class

////////////////////// add a new spring
CmdLabelAddspring::CmdLabelAddspring(int themol, int theatom1, int theatom2,
			       float thek) :
	Command(LABEL_ADDSPRING) {
  /* initialize the member variables */
  molid = themol;
  atom1 = theatom1;
  atom2 = theatom2;
  k = thek;
}

void CmdLabelAddspring::create_text() {
  *cmdText << "label addspring " << molid << " " << atom1 << " "
	   << atom2 << " " << k << ends;
}

////////////////////// add a new label 
CmdLabelAdd::CmdLabelAdd(const char *geomcat, int n, int *middata, 
                         int *atmdata) 
: Command(LABEL_ADD) {

  sprintf(geomcatStr, "%s ", geomcat);
  num_geomitems = n;
  for (int j=0; j<n; j++) {
    char buf[50];
    sprintf(buf,"%d/%d",middata[j], atmdata[j]); 
    geomitems[j] = stringdup(buf);
  }
}

void CmdLabelAdd::create_text(void) {
  *cmdText << "label add " << geomcatStr;
  for (int i=0; i<num_geomitems; i++)
    *cmdText << geomitems[i] << " ";
  *cmdText << ends;
}

CmdLabelAdd::~CmdLabelAdd(void) {
  for (int i=0; i<num_geomitems; i++)  
    delete [] geomitems[i];
}


////////////////////// toggle a geometry category on/off
CmdLabelShow::CmdLabelShow(const char *geomcat, int n, int s) :
	Command(LABEL_SHOW), item(n), show(s) {
  sprintf(geomcatStr, "%s", geomcat);
}

void CmdLabelShow::create_text(void) {
  *cmdText << "label " << (show ? "show" : "hide") << " " << geomcatStr;
  if(item >= 0)
    *cmdText << " " << item;
  *cmdText << ends;
}

//////////////////////// delete a label
CmdLabelDelete::CmdLabelDelete(const char *geomcat, int n) :
	Command(LABEL_DELETE), item(n) {
  sprintf(geomcatStr, "%s", geomcat);
}

void CmdLabelDelete::create_text(void) {
  *cmdText << "label delete " << geomcatStr;
  if (item >= 0)
    *cmdText << " " << item;
  *cmdText << ends;
}

void CmdLabelTextSize::create_text() {
  *cmdText << "label textsize " << size << ends;
}

void CmdLabelTextThickness::create_text() {
  *cmdText << "label textthickness " << thickness << ends;
}

CmdLabelTextOffset::CmdLabelTextOffset(const char *name, int ind, float x, float y)
: Command(LABEL_TEXTSIZE), n(ind), m_x(x), m_y(y) {
  nm = stringdup(name);
}
CmdLabelTextOffset::~CmdLabelTextOffset() {
  delete [] nm;
}

void CmdLabelTextOffset::create_text() {
  *cmdText << "label textoffset " << (const char *)nm << " " << n << " { " << m_x << " " << m_y << " } " << ends;
}

CmdLabelTextFormat::CmdLabelTextFormat(const char *name, int ind, 
    const char *fmt)
: Command(LABEL_TEXTSIZE), n(ind) {
  nm = stringdup(name);
  format = stringdup(fmt);
}
CmdLabelTextFormat::~CmdLabelTextFormat() {
  delete [] nm;
  delete [] format;
}

void CmdLabelTextFormat::create_text() {
  *cmdText << "label textformat " << (const char *)nm << " " << n 
           << " { " << format << " " << " } " << ends;
}

