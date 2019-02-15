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
 *      $RCSfile: frame_selector.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Fltk dialogs for selecting/deleting ranges of frames
 ***************************************************************************/

#include "frame_selector.h"

#include <string.h>
#include <stdio.h>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Value_Input.H>
#include <FL/Fl_Box.H>

/// Fltk dialog for selecting/deleting ranges of frames
class FrameSelector : public Fl_Window {
public:
  FrameSelector(const char *, const char *, int);
  static void first_cb(Fl_Widget *w, void *v);
  static void last_cb(Fl_Widget *w, void *v);
  int do_dialog();
  
  Fl_Button *okbutton;
  Fl_Button *cancelbutton;
  Fl_Value_Input *firstvalue;
  Fl_Value_Input *lastvalue;
  Fl_Value_Input *stridevalue;
  Fl_Box *molnamelabel;
  Fl_Box *messagelabel;

  const int maxframe;
};

void FrameSelector::first_cb(Fl_Widget *w, void *v) {
  Fl_Value_Input *counter = (Fl_Value_Input *)w;
  FrameSelector *sel = (FrameSelector *)v;

  double val = counter->value();
  if (val < 0) {
    // wrap to max if we have a max, otherwise set back to 0
    if (sel->maxframe > 0) 
      counter->value(sel->maxframe);
    else
      counter->value(0);
  } else if (sel->maxframe >= 0 && val > sel->maxframe) {
    counter->value(0);
  }
}
    
void FrameSelector::last_cb(Fl_Widget *w, void *v) {
  Fl_Value_Input *counter = (Fl_Value_Input *)w;
  FrameSelector *sel = (FrameSelector *)v;

  double val = counter->value();

  if (val < 0) {
    // wrap to max if we have a max, otherwise set back to -1 
    if (sel->maxframe > 0) 
      counter->value(sel->maxframe);
    else
      counter->value(-1);
  } else if (sel->maxframe >= 0 && val > sel->maxframe) {
    counter->value(0);
  }
}


int FrameSelector::do_dialog() {
  int result = 0;
  hotspot(this);
  show();
  while(1) {
    Fl::wait();
    Fl_Widget *o = Fl::readqueue();
    if (o == okbutton) {result=1; break;}
    else if (o == cancelbutton || o == this) break;
  }
  hide();
  return result;
}

 
FrameSelector::FrameSelector(const char *message, const char *molname, int max) 
: Fl_Window(240, 275, "Frame Selector"), maxframe(max) {
  
  messagelabel = new Fl_Box(0, 10, 240, 20, message);
  messagelabel->align(FL_ALIGN_INSIDE|FL_ALIGN_CENTER);

  molnamelabel = new Fl_Box(0, 37, 240, 20, molname);
  molnamelabel->align(FL_ALIGN_INSIDE|FL_ALIGN_CENTER);
    
  { Fl_Box* o = new Fl_Box(10, 70, 220, 20, "In the range...");
    o->align(FL_ALIGN_INSIDE|FL_ALIGN_LEFT);
  }
  
  firstvalue = new Fl_Value_Input(75, 100, 100, 25, "First: ");
  firstvalue->minimum(0);
  firstvalue->maximum((double)max);
  firstvalue->step(1);
  firstvalue->value(0);
  firstvalue->align(FL_ALIGN_LEFT);

  lastvalue = new Fl_Value_Input(75, 130, 100, 25, "Last: ");
  lastvalue->minimum(0);
  lastvalue->maximum((double)max);
  lastvalue->step(1);
  lastvalue->value((double)max);
  lastvalue->align(FL_ALIGN_LEFT);

  { Fl_Box* o = new Fl_Box(10, 170, 220, 20, "And keeping frames every...");
    o->align(FL_ALIGN_INSIDE|FL_ALIGN_LEFT);
  }
  
  stridevalue = new Fl_Value_Input(75, 200, 100, 25, "Stride: ");
  stridevalue->minimum(1);
  stridevalue->maximum((double)max+1);  // since max is index 0 based
  stridevalue->step(1);
  stridevalue->value(1);
  stridevalue->align(FL_ALIGN_LEFT);

  okbutton = new Fl_Button(45, 240, 85, 25, "Select");
  cancelbutton = new Fl_Button(150, 240, 60, 25, "Cancel");
    
  set_modal();
  end();
}


int frame_selector(const char *message, const char *molname, int maxframe,
                          int *first, int *last, int *stride) {
  char *molstr=new char[strlen(molname)+3];
  sprintf(molstr, "\"%s\"", molname);  //XXX it'd be nice to add the molid here
  
  FrameSelector *f = new FrameSelector(message, molstr, maxframe);
  if (*first < 0) *first = maxframe;
  if (*last < 0) *last = maxframe;
  f->firstvalue->value((double)*first);
  f->lastvalue->value((double)*last);
  f->stridevalue->value((double)*stride);

  int ok = f->do_dialog();
 
  if (ok) {
    *first = (int)f->firstvalue->value();
    *last = (int)f->lastvalue->value();
    *stride = (int)f->stridevalue->value();
  }
  delete f;
  delete[] molstr;
  
  return ok;
}



int frame_delete_selector(const char *molname, int maxframe,
                          int *first, int *last, int *stride) { 
  char *molstr=new char[strlen(molname)+3];
  sprintf(molstr, "\"%s\"", molname);  //XXX it'd be nice to add the molid here
  
  FrameSelector *f = new FrameSelector("Select frames to delete for:", molstr, maxframe);
  f->firstvalue->value((double)*first);
  f->lastvalue->value((double)*last);
  f->stridevalue->minimum(0); 
  f->stridevalue->value((double)*stride);
  f->okbutton->label("Delete");
  
  int ok = f->do_dialog();

  if (ok) {
    *first = (int)f->firstvalue->value();
    *last = (int)f->lastvalue->value();
    *stride = (int)f->stridevalue->value();
  }
  delete f;
  delete[] molstr;
  
  return ok;
}



