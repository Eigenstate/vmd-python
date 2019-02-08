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
 *	$RCSfile: P_Tool.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.84 $	$Date: 2019/01/24 04:57:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 *
 ***************************************************************************/

#include "P_Tracker.h"
#include "P_Feedback.h"
#include "P_Buttons.h"
#include "P_Tool.h"
#include "DispCmds.h"
#include "Scene.h"
#include "VMDApp.h"
#include "AtomSel.h"
#include "MoleculeList.h"
#include "PickList.h"
#include "Displayable.h"

/// Displayable used to render a tool on the screen
class DrawTool : public Displayable {
public:
  DrawTool(Displayable *aParent) 
  : Displayable(aParent) {
     
    rot_off();  // don't listen to anyone else's transformations
    scale_on();
    set_scale(1);
    scale_off();
    glob_trans_off();
    cent_trans_off();

    // set up the display list once, since it never changes
    DispCmdCone cone;
    DispCmdColorIndex drawcolor;
    float base[3] = {0, 0.0, 0.0};
    float tip[3] = {-0.7f, 0.0, 0.0};
    append(DMATERIALON);
    drawcolor.putdata(REGTAN, cmdList);
    cone.putdata(tip, base, 0.07f, 0, 40, cmdList);
    tip[0] = -0.6f;
  
    base[0] = -0.6f;
    base[1] =  0.1f;
    base[2] =  0.0f;
    cone.putdata(base, tip, 0.04f, 0, 20, cmdList);
  }
};


Tool::Tool(int serialno, VMDApp *vmdapp, Displayable *aParent)
  : UIObject(vmdapp), my_id(serialno) {

  tracker = NULL;
  buttons = NULL;
  feedback = NULL;
  amalive = 1;
 
  lost_sensor=0;
  wasgrabbing=0;
  forcescale = 1;
  springscale = 1;

  targeted_atom = targeted_molecule = -1;
  // As long as a rep is targeted, the targeted molecule must be valid and
  // not change.
  targeted_rep = NULL;
  sel_total_mass = 0;

  dtool = new DrawTool(aParent);
}

Tool::~Tool() {
  delete dtool;
  delete [] targeted_rep;
  if (!lost_sensor) {
    delete tracker;
    delete buttons;
    delete feedback;
  } 
}

void Tool::clear_devices() {
  forceoff();
  delete tracker; tracker = NULL; 
  delete buttons; buttons = NULL; 
  delete feedback; feedback = NULL;
}

int Tool::add_tracker(VMDTracker *t, const SensorConfig *config) {
  delete tracker;
  tracker = t;
  if (tracker) {
    trackerDev = (char *)config->getdevice();
    return tracker->start(config);
  } 
  return TRUE;
}
int Tool::add_feedback(Feedback *f, const SensorConfig *config) {
  delete feedback;
  feedback = f;
  if (feedback) {
    feedbackDev = (char *)config->getdevice();
    return feedback->start(config);
  } 
  return TRUE;
}
int Tool::add_buttons(Buttons *b, const SensorConfig *config) {
  delete buttons;
  buttons = b;
  if (buttons) {
    buttonDev = (char *)config->getdevice();
    return buttons->start(config);
  } 
  return TRUE;
}

int Tool::remove_device(const char *device) {
  if (tracker && !strcmp(device, (const char *)trackerDev)) {
    delete tracker;
    tracker = NULL;
  } else if (feedback && !strcmp(device, (const char *)feedbackDev)) {
    delete feedback;
    feedback = NULL;
  } else if (buttons && !strcmp(device, (const char *)buttonDev)) {
    delete buttons;
    buttons = NULL;
  } else {
    return 0;
  }
  return 1;
}

int Tool::steal_sensor(Tool *from) {
  clear_devices();
  tracker = from->tracker;
  buttons = from->buttons;
  feedback = from->feedback;
  trackerDev = from->trackerDev;
  buttonDev = from->buttonDev;
  feedbackDev = from->feedbackDev;

  springscale = from->springscale;
  forcescale = from->forcescale;
  from->lost_sensor = 1;
  return TRUE;
}
void Tool::getdevices(char **ret) {
  int i=0;
  if (tracker) ret[i++] = (char *)(const char *)trackerDev;
  if (buttons) ret[i++] = (char *)(const char *)buttonDev;
  if (feedback) ret[i++] = (char *)(const char *)feedbackDev;
  ret[i] = NULL;
}

int Tool::isgrabbing() {
  if (buttons) return buttons->state(0);
  return 0;
}

const float *Tool::position() const {
  if (tracker) return pos;
  return NULL;
}

const Matrix4 *Tool::orientation() {
  if (tracker) return &orient;
  return NULL;
}

void Tool::update() {
  if (tracker) {
    tracker->update();
    if (!tracker->alive()) {
      msgWarn << "Tool: lost connection to tracker " << tracker->device_name() 
              << sendmsg;
      amalive = 0;
      delete tracker;
      tracker = NULL;
    } else {
      const float *tmp = tracker->position();
      for (int i=0; i<tracker->dimension(); i++) pos[i] = tmp[i];
      orient.loadmatrix(tracker->orientation());
    }
  }
  if (buttons) buttons->update();
  if (feedback) feedback->update();
}

int Tool::check_event() {
  update();

  // subclass-specific action
  do_event();

  // update the visual orientation of the tool if there is a tracker
  if (tracker) {
    dtool->rot_on();
    dtool->glob_trans_on();
    dtool->set_rot(*orientation());
    dtool->set_glob_trans(position()[0], position()[1], position()[2]);
    dtool->rot_off();
    dtool->glob_trans_off();
  }
  return 1;
}

void Tool::setscale(float sc) {
  if (tracker) tracker->set_scale(sc);
}
float Tool::getscale() {
  if (tracker) return tracker->get_scale();
  return 0;
}

void Tool::setoffset(float *offset) {
  if (tracker) tracker->set_offset(offset);
}
const float *Tool::getoffset() {
  if (tracker) return tracker->get_offset();
  return NULL;
}

int Tool::assign_rep(int mol, int rep) {
  Molecule *m = app->moleculeList->mol_from_id(mol);
  if (!m) return FALSE;
  DrawMolItem *item = m->component(rep);
  if (!item) return FALSE;
  clear_rep();
  targeted_rep = stringdup(item->name);
  targeted_molecule = mol;
  const AtomSel *sel = item->atomSel;

  // get the total mass
  sel_total_mass=0;
  const float *mass = m->mass();
  for (int i=sel->firstsel; i<=sel->lastsel; i++) {
    if (sel->on[i]) {
      sel_total_mass += mass[i];
    }
  }

  // kill the highlight
  if (app->pickList) {
    app->pickList->pick_callback_clear((char *)"uivr");
  }  
  return TRUE;
}
 
int Tool::get_targeted_atom(int *molret, int *atomret) const {
  if (targeted_rep != NULL) return 0; // must be an atom
  if (targeted_molecule == -1) return 0; // must be targeted
  *molret = targeted_molecule;
  *atomret = targeted_atom;
  return 1;
}

void Tool::tool_location_update() {
  int tag;

  // pick a nearby atom if necessary
  if(position() && make_callbacks && !targeted_rep) 
    app->pickList->pick_check(3, position(), tag, NULL,
		    0.2f,(char *)"uivr");
}

int Tool::target(int target_type, float *mpos, int just_checking) {
  int mol = -1;
  int atom = -1;
  int tag = -1;
  Molecule *m;
  Matrix4 globm;

  mpos[0]=mpos[1]=mpos[2]=0;

  // is the position NULL?
  if(!position()) return 0;

  // are we selecting something?
  if(targeted_rep && target_type == TARGET_TUG) {
    // Check that the targeted rep still exists
    int repid = app->molrep_get_by_name(targeted_molecule, targeted_rep);
    if (repid >= 0 && app->molecule_numframes(targeted_molecule) > 0) {
      Molecule *m = app->moleculeList->mol_from_id(targeted_molecule);
      Timestep *ts = m->current();
      const AtomSel *sel = m->component(repid)->atomSel;
      
      // Loop over each atom in the selection to get the COM.
      float com[3] = {0,0,0};
      const float *amass = m->mass();
      for (int i=sel->firstsel; i<=sel->lastsel; i++) {
        if (sel->on[i]) {
	  float mass = amass[i];
	  float *p = ts->pos + 3*i;
	  com[0] += p[0]*mass;
	  com[1] += p[1]*mass;
	  com[2] += p[2]*mass;
        }
      }
      vec_scale(com,1/sel_total_mass,com);
    
      // Now transform the coordinates for the tracker
      globm = m->tm; // use centering as well, so we find the exact atom
      globm.multpoint3d(com,mpos);
  
      return 1;
    } else {
      // Our targeted rep has flown the coop.  Fall back on regular target
      clear_rep();
    }
  }

  if(targeted_molecule == -1) {
    Pickable *p=NULL;
    if (app->display) {
      if(make_callbacks) {
	p = app->pickList->pick_check(3, position(), tag, NULL, 0.2f,
				    (char *)"uivr");
      } else {
	p = app->pickList->pick_check(3, position(), tag, NULL, 0.2f);
      }

      if (p) {
	Molecule *m = app->moleculeList->check_pickable(p);
	if (m) {
	  mol = m->id();
	  atom = tag;
	}
      }
    }
  
    if (atom == -1 || mol == -1) return 0;

    if(!just_checking) {
      targeted_atom = atom;
      targeted_molecule = mol;
    }
  }
  else {
    atom = targeted_atom;
    mol = targeted_molecule;
  }

  m = app->moleculeList->mol_from_id(mol);

  /* verify sanity */
  if(m==NULL) {
    msgErr << "Tool: Bad molecule ID found." << sendmsg;
    let_go();
    return 0;
  }

  /* Now return the correct position to the targeting tool */
  if(target_type == TARGET_GRAB) { // target a molecule
    globm.translate(m->globt);
    vec_copy(mpos, m->globt);
  } else { // target an atom
    globm = m->tm; // use centering as well, so we find the exact atom
    Timestep *ts = m->current();
    if(ts->pos == NULL) {
      msgErr << "Timestep has NULL position." << sendmsg;
    }

    float *p = ts->pos + 3*atom;

    globm.multpoint3d(p,mpos);
  }    

  return 1;
}

void Tool::tug(const float *theforce) {
  Molecule *m = app->moleculeList->mol_from_id(targeted_molecule);  
  if (!m) return;

  Matrix4 rot = m->rotm;
  // since this is an orthogonal matrix, a transpose is an inverse
  rot.transpose();
  float force[3];
  rot.multpoint3d(theforce, force);

  if (targeted_rep) {
    // if there is a selection, pull with a total force <force>
    const AtomSel *sel = m->component(
      app->molrep_get_by_name(targeted_molecule, targeted_rep))->atomSel;
    const float *amass = m->mass();
    for (int i=sel->firstsel; i<=sel->lastsel; i++) {
      if (sel->on[i]) {
	float mass = amass[i];
	float atomforce[3];
	vec_scale(atomforce,mass/sel_total_mass,force);
	m->addForce(i, atomforce);
      }
    }
  }
  else {
    // otherwise just pull with a simple force
    m->addForce(targeted_atom, force);
  }
}

void Tool::dograb() {
  Molecule *m;
  Matrix4 rot;
  float p[3], mpos[3], mdiff[3], mrotdiff[3], mchange[3], futurepos[3];

  m = app->moleculeList->mol_from_id(targeted_molecule);
  if(m==NULL) return;

  mpos[0] = m->globt[0];
  mpos[1] = m->globt[1];
  mpos[2] = m->globt[2];

  if (grabs) {
    last_rot.inverse();
    rot=*orientation();
    rot.multmatrix(last_rot);
    p[0] = position()[0] - last_pos[0];
    p[1] = position()[1] - last_pos[1];
    p[2] = position()[2] - last_pos[2];

    // apply the translation due to translation of the tracker
    app->scene_translate_by(p[0], p[1], p[2]);
    // where the molecule will be after that command
    vec_add(futurepos,p,mpos);

    // rotate the molecule about its center
    app->scene_rotate_by(rot.mat);

    // compute the vector between the molecule and the tracker
    vec_sub(mdiff,futurepos,position());
    // rotate that
    rot.multpoint3d(mdiff, mrotdiff);
    // and subtract
    vec_sub(mchange,mrotdiff,mdiff);
    
    // giving us the translation for the molecule due to rotation
    app->scene_translate_by(mchange[0],mchange[1], mchange[2]);
  } else {
    int i;
#ifdef VMDVRJUGGLER
    // VR Juggler: 
    // do this through VMDApp, so the cmd is sent to the slaves as well
    for(i=0; i<app->moleculeList->num(); i++) {
      int id = app->moleculeList->molecule(i)->id();
      app->molecule_fix(id, true);
    }
    int id = m->id();
    app->molecule_fix(id, false);
#else
    for(i=0; i<app->moleculeList->num(); i++)
      app->moleculeList->molecule(i)->fix();
    m->unfix();
#endif
  }

  last_rot = *orientation();
  last_pos[0] = position()[0];
  last_pos[1] = position()[1];
  last_pos[2] = position()[2];
}

void Tool::ungrab() {
  int i;
  if (!targeted_molecule)
    targeted_molecule = -1;
#ifdef VMDVRJUGGLER
  // VR Juggler: 
  // do this through VMDApp, so the cmd is sent to the slaves as well
  for(i=0; i<app->moleculeList->num(); i++) {
    int id = app->moleculeList->molecule(i)->id();
    app->molecule_fix(id, false);
  }
#else
  for(i=0;i<app->moleculeList->num();i++)
    app->moleculeList->molecule(i)->unfix();
#endif
}

float Tool::getTargetScale() {
  if (targeted_molecule != -1) {
    return app->moleculeList->mol_from_id(targeted_molecule)->scale;
  }
  return 0;
}

int Tool::dimension() {
  if (tracker) return tracker->dimension();
  return 0;
}

void Tool::setplaneconstraint(float k, const float *p, const float *n) {
  float scaledpoint[3];
  if (!feedback) return;
  if (!tracker) return; 
  const float *offset = tracker->get_offset();
  const float scale = tracker->get_scale();
  scaledpoint[0] = p[0]/scale-offset[0];
  scaledpoint[1] = p[1]/scale-offset[1];
  scaledpoint[2] = p[2]/scale-offset[2];
  feedback->zeroforce();
  feedback->addplaneconstraint(springscale*k,scaledpoint,n);
}

void Tool::addplaneconstraint(float k, const float *p, const float *n) {
  float scaledpoint[3];
  if (!feedback) return;
  if (!tracker) return; 
  const float *offset = tracker->get_offset();
  const float scale = tracker->get_scale();
  scaledpoint[0] = p[0]/scale-offset[0];
  scaledpoint[1] = p[1]/scale-offset[1];
  scaledpoint[2] = p[2]/scale-offset[2];
  feedback->addplaneconstraint(springscale*k,scaledpoint,n);
}

void Tool::setconstraint(float k, const float *p) {
  float scaledcenter[3];
  if (!feedback) return;
  if (!tracker) return; 
  const float *offset = tracker->get_offset();
  const float scale = tracker->get_scale();
  scaledcenter[0] = p[0]/scale-offset[0];
  scaledcenter[1] = p[1]/scale-offset[1];
  scaledcenter[2] = p[2]/scale-offset[2];
  feedback->zeroforce();
  feedback->addconstraint(springscale*k,scaledcenter);
}

void Tool::setforcefield(const float *o, const float *force, 
                         const float *jacobian) {

  float sforce[3];
  float scaledjacobian[9];
  int i;
  if (!tracker) return;
  if (!feedback) return;
  const float scale = tracker->get_scale();
  for (i=0; i<3; i++) sforce[i] = force[i]*springscale;
  for(i=0;i<9;i++) scaledjacobian[i] = jacobian[i]*springscale/scale;
  feedback->zeroforce();
  feedback->addforcefield(o, sforce, scaledjacobian);
}

void Tool::sendforce() {
  if (tracker && feedback) feedback->sendforce(position());
}
void Tool::forceoff() {
  if (feedback) feedback->forceoff();
}

