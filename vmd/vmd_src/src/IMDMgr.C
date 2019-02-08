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
 *      $RCSfile: IMDMgr.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.43 $       $Date: 2019/01/17 21:20:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  High level interactive MD simulation management and update routines.
 ***************************************************************************/

#include "IMDMgr.h"
#include "imd.h"
#include "IMDSimThread.h"
#include "IMDSimBlocking.h"
#include "utilities.h"
#include "TextEvent.h"
#include "Molecule.h"

IMDMgr::IMDMgr(VMDApp *vmdapp)
: UIObject(vmdapp) { 
  mol = NULL;  
  sim = NULL;
  host = NULL;
  port = 0;
  Trate = 1;
  keep_rate = 0;
  copy_unit_cell = 0;  // copy unit cell info from first (previous) frame
  frames_received = 0;
  energies = new IMDEnergies;
  want_command(Command::MOL_DEL);
}

IMDMgr::~IMDMgr() {
  if (mol)
    detach();
  delete [] host;
  delete energies;
}

int IMDMgr::connect(Molecule *m, const char *h, int p) {
  if (mol) {
    return 0;
  }

  delete [] host;
  host = stringdup(h);
  port = p;

#ifdef VMDTHREADS
  sim = new IMDSimThread(h, p);
#else
  sim = new IMDSimBlocking(h, p);
#endif
  if (!sim->isConnected()) {
    delete sim;
    sim = 0;
    return 0;
  }
  mol = m;
  frames_received = 0;
  return 1;
}

void IMDMgr::pause() {
  if (sim) sim->pause();
}

void IMDMgr::unpause() {
  if (sim) sim->unpause();
}

void IMDMgr::togglepause() {
  if (!sim) return;
  int state = sim->getSimState();
  if (state == IMDSim::IMDRUNNING)
    sim->pause();
  else if (state == IMDSim::IMDPAUSED)
    sim->unpause();
}

void IMDMgr::detach() {
  if (sim) sim->detach();
  delete sim;
  sim = NULL;
  mol = NULL;
}

void IMDMgr::kill() {
  if (sim) sim->kill();
  delete sim;
  sim = NULL;
  mol = NULL;
}

int IMDMgr::send_forces(int n, const int *ind, const float *force) {
  if (!sim) return 0;

  // make a temporary copy because sim may byte-swap ind and force.
  int *tmpind = new int[n];
  float *tmpforce = new float[3L*n];
  memcpy(tmpind, ind, n*sizeof(int));
  memcpy(tmpforce, force, 3L*n*sizeof(float));
  sim->send_forces(n, tmpind, tmpforce);
  delete [] tmpind;
  delete [] tmpforce;
  return 1;
}

void IMDMgr::set_trans_rate(int rate) {
  if (rate > 0) {
    Trate = rate;
    if (sim) sim->set_transrate(rate);
  }
}

void IMDMgr::set_keep_rate(int rate) {
  if (rate >= 0) {
    keep_rate = rate;
  }
}

void IMDMgr::set_copyunitcell(int onoff) {
  if (onoff) {
    copy_unit_cell = 1;
  } else {
    copy_unit_cell = 0;
  }
}

int IMDMgr::check_event() {
  if (sim && !sim->isConnected()) {
    detach();
    msgInfo << "IMD connection ended unexpectedly; connection terminated."
           << sendmsg;
  }
  if (!sim) return 0;
  
  sim->update();
  if (sim->next_ts_available()) {
    Timestep *newts = mol->get_last_frame();
    int do_save = (!newts || frames_received < 1 || 
        (keep_rate > 0 && !(frames_received % keep_rate)));
    if (do_save) {
      newts = new Timestep(mol->nAtoms);

      // XXX Hack to enable copying cell size and shape from previous frame
      // since the existing IMD protocol doesn't provide a means to 
      // transmit updates to the unit cell info.
      if (copy_unit_cell) {
        Timestep *oldts = mol->get_last_frame();
        if (oldts) {
          newts->a_length = oldts->a_length;
          newts->b_length = oldts->b_length;
          newts->c_length = oldts->c_length;
          newts->alpha = oldts->alpha;
          newts->beta = oldts->beta;
          newts->gamma = oldts->gamma;
        }
      }
    }

    float *pos = newts->pos;
    sim->get_next_ts(pos, energies);

    newts->timesteps = energies->tstep; 
    newts->energy[TSE_BOND] = energies->Ebond; 
    newts->energy[TSE_ANGLE] = energies->Eangle; 
    newts->energy[TSE_DIHE] = energies->Edihe;
    newts->energy[TSE_IMPR] = energies->Eimpr; 
    newts->energy[TSE_VDW] = energies->Evdw; 
    newts->energy[TSE_COUL] = energies->Eelec; 
    newts->energy[TSE_HBOND] = 0;    // not supported
    newts->energy[TSE_TEMP] = energies->T; 
    newts->energy[TSE_PE] = energies->Epot; 
    newts->energy[TSE_TOTAL] = energies->Etot; 
    newts->energy[TSE_KE] = energies->Etot - energies->Epot;
 
    if (do_save) {
      mol->append_frame(newts);
    } else {
      mol->force_recalc(DrawMolItem::MOL_REGEN);
    }
    frames_received++;
    runcommand(new TimestepEvent(mol->id(), mol->numframes()-1));
  }
  return 0;
}

int IMDMgr::act_on_command(int type, Command *cmd) {
  if (type == Command::MOL_DEL) {
    detach();
    mol = NULL;
  }
  return 0;
}

