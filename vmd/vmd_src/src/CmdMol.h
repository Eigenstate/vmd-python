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
 *      $RCSfile: CmdMol.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.75 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for affecting molecules.
 *
 ***************************************************************************/
#ifndef CMDMOL_H
#define CMDMOL_H

#include <string.h>
#include "Command.h"
#include "VMDApp.h"


// the following uses the Cmdtypes MOL_NEW, MOL_DEL, MOL_ACTIVE,
// MOL_FIX, MOL_ON, MOL_TOP, MOL_SELECT, MOL_REP, MOL_COLOR, MOL_ADDREP,
// MOL_MODREP, MOL_DELREP, MOL_MODREPITEM

/// Create a new molecule
class CmdMolNew : public Command {
public:
  CmdMolNew() : Command(MOL_NEW) {}
};

/// Notification that a molecule has been created 
class CmdMolLoad : public Command {
protected:
    virtual void create_text();

public:
  int molid;
  char *name;
  char *type;
  FileSpec spec;
  CmdMolLoad(int themolid, const char *thename, const char *thetype, 
      const FileSpec *thespec) 
  : Command(MOL_NEW), molid(themolid), spec(*thespec) {
    name = strdup(thename);
    type = strdup(thetype);
  }
  ~CmdMolLoad() {
    free(type);
    free(name);
  }
};
  

/// Cancel the loading/saving of a coordinate file
class CmdMolCancel : public Command {
protected:
  virtual void create_text(void);

public:
  int whichMol;
  CmdMolCancel(int molid)
  : Command(MOL_CANCEL), whichMol(molid) {}
};


/// delete the Nth molecule in the molecule List
class CmdMolDelete : public Command {

protected:
  virtual void create_text(void);

public:
  int whichMol;
  CmdMolDelete(int molid)
  : Command(MOL_DEL), whichMol(molid) {}
};


/// make the Nth molecule 'active' or 'inactive'
class CmdMolActive : public Command {

protected:
  virtual void create_text(void);

public:
  int whichMol;
  int yn;                ///< new state for characteristic

  /// constructor: which molecule, new setting
  CmdMolActive(int id, int on)
  : Command(MOL_ACTIVE), whichMol(id), yn(on) {}
};


/// make the Nth molecule 'fixed' or 'free'
class CmdMolFix : public Command {

protected:
  virtual void create_text(void);

public:
  int whichMol;
  int yn;                ///< new state for characteristic
  /// constructor: which molecule, new setting
  CmdMolFix(int id, int on) 
  : Command(MOL_FIX), whichMol(id), yn(on) {}
};



/// make the Nth molecule 'on' or 'off'
class CmdMolOn : public Command {
protected:
  virtual void create_text(void);

public:
  int whichMol;
  int yn;                ///< new state for characteristic
  /// constructor: which molecule, new setting
  CmdMolOn(int id, int on) 
  : Command(MOL_ON), whichMol(id), yn(on) {}
};



///  make the Nth molecule 'top'.  
class CmdMolTop : public Command {
protected:
  virtual void create_text(void);

public:
  int whichMol;
  /// constructor: which molecule
  CmdMolTop(int id)
  : Command(MOL_TOP), whichMol(id) {}
};


/// set the current atom selection in moleculeList
class CmdMolSelect : public Command {
public:
  /// new selection command (if NULL, print out current settings)
  char *sel;
  
protected:
  virtual void create_text(void);

public:
  CmdMolSelect(const char *newsel)
  : Command(MOL_SELECT) { sel = strdup(newsel); }
  virtual ~CmdMolSelect(void)      { free(sel); }
};


/// set the current atom representation in moleculeList
class CmdMolRep : public Command {
public:
  // new representation command (if NULL, print out current settings)
  char *sel;
  
protected:
  virtual void create_text(void) ;

public:
  CmdMolRep(const char *newsel)
  : Command(MOL_REP) {  sel = strdup(newsel); }
  virtual ~CmdMolRep(void)      { free(sel); }
};


/// set the current atom coloring method in moleculeList
class CmdMolColor : public Command {
public:
  /// new color command (if NULL, print out current settings)
  char *sel;
  
protected:
  virtual void create_text(void);

public:
  CmdMolColor(const char *newsel)
  : Command(MOL_COLOR) {  sel = strdup(newsel); }
  virtual ~CmdMolColor(void) { free(sel); }
};


/// set the current atom material method in moleculeList
class CmdMolMaterial : public Command {
public:
  /// new material method 
  char *mat;
  
protected:
  virtual void create_text(void);

public:
  CmdMolMaterial(const char *newmat)
  : Command(MOL_MATERIAL) { mat = strdup(newmat); }
  virtual ~CmdMolMaterial(void)      { free(mat); }
};


/// add a new representation to the specified molecule
class CmdMolAddRep : public Command {

protected:
  virtual void create_text(void);

public:
  int whichMol;
  /// constructor: which molecule
  CmdMolAddRep(int id) 
  : Command(MOL_ADDREP), whichMol(id) {}
};


/// change the representation to the current defaults
class CmdMolChangeRep : public Command {
protected:
  virtual void create_text(void);
 
public:
  int whichMol;
  int repn;
  CmdMolChangeRep(int molid, int repid)
  : Command(MOL_MODREP), whichMol(molid), repn(repid) {}
};


/// change 1 representation characteristic for the specified mol
class CmdMolChangeRepItem : public Command {

public:
  enum RepData { COLOR, REP, SEL, MAT }; ///< modifiable characteristics
  int whichMol;                          ///< which molecule
  int repn;                              ///< which rep to change
  RepData repData;                       ///< type of item to change
  char *str;                             ///< new value

protected:
  virtual void create_text(void);

public:
  /// constructor: which rep, which mol, rep data
  CmdMolChangeRepItem(int rpos, int id, RepData rd, const char *s)
  : Command(MOL_MODREPITEM), whichMol(id), repn(rpos), repData(rd) {
    str = strdup(s);
  }

  virtual ~CmdMolChangeRepItem(void) { free(str); }
};


/// Change the auto-update for the selection of the specified rep 
class CmdMolRepSelUpdate : public Command {
public:
  int whichMol;
  int repn;
  int onoroff;

protected:
  virtual void create_text();

public:
  CmdMolRepSelUpdate(int rpos, int id, int on)
  : Command(MOL_REPSELUPDATE), whichMol(id), repn(rpos), onoroff(on) {}
};


/// Change the auto-update for the color of the specified rep 
class CmdMolRepColorUpdate : public Command {
public:
  int whichMol;
  int repn;
  int onoroff;

protected:
  virtual void create_text();

public:
  CmdMolRepColorUpdate(int rpos, int id, int on)
  : Command(MOL_REPCOLORUPDATE), whichMol(id), repn(rpos), onoroff(on) {}
};


/// delete a representation for the specified molecule
class CmdMolDeleteRep : public Command {
public:
  int whichMol;
  int repn;       ///< which rep to delete

protected:
  virtual void create_text(void);

public:
  CmdMolDeleteRep(int rpos, int id) 
  : Command(MOL_DELREP), whichMol(id), repn(rpos) {}
};


/// re-analyze structure after atom names, bonds, etc have been modified
class CmdMolReanalyze : public Command {
protected:
  virtual void create_text(void);
 
public:
  int whichMol;
  CmdMolReanalyze(int id) 
  : Command(MOL_REANALYZE), whichMol(id) {}
};


/// re-analyze structure after atom names, bonds, etc have been modified
class CmdMolBondsRecalc : public Command {
protected:
  virtual void create_text(void);
 
public:
  int whichMol;
  CmdMolBondsRecalc(int id) 
  : Command(MOL_BONDRECALC), whichMol(id) {}
};

/// recalculate secondary structure based on current coordinates
class CmdMolSSRecalc : public Command {
protected:
  virtual void create_text(void);
 
public:
  int whichMol;
  CmdMolSSRecalc(int id) 
  : Command(MOL_SSRECALC), whichMol(id) {}
};

/// Add a new volumetric dataset to the selected molecule
class CmdMolVolume : public Command {
public:
  int whichMol;
  CmdMolVolume(int id)
  : Command(MOL_VOLUME), whichMol(id) {}
};


/// rename a molecule
class CmdMolRename : public Command {
protected:
  virtual void create_text();

public:
  int whichMol;
  char *newname;

  CmdMolRename(int id, const char *nm);
  ~CmdMolRename();
};


/// Set which periodic images are displayed for the selected representation
class CmdMolShowPeriodic : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const int pbc;
  CmdMolShowPeriodic(int mol, int rep, int thepbc)
    : Command(MOL_SHOWPERIODIC), whichMol(mol), repn(rep), pbc(thepbc) {}
};


/// Set the number of periodic images displayed for the selected representation
class CmdMolNumPeriodic: public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const int nimages;
  CmdMolNumPeriodic(int mol, int rep, int n)
    : Command(MOL_NUMPERIODIC), whichMol(mol), repn(rep), nimages(n) {}
};


/// Set which instance images are displayed for the selected representation
class CmdMolShowInstances : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const int instances;
  CmdMolShowInstances(int mol, int rep, int theinstances)
    : Command(MOL_SHOWPERIODIC), whichMol(mol), repn(rep), instances(theinstances) {}
};



/// Set the color scale min/max range for the selected representation
class CmdMolScaleMinmax : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const float scalemin, scalemax;
  const int reset;
  CmdMolScaleMinmax(int mol, int rep, float themin, float themax,int doreset=0)
  : Command(MOL_SCALEMINMAX), whichMol(mol), repn(rep), scalemin(themin),
    scalemax(themax), reset(doreset) {}
};


/// Set the range of frames to draw for this rep
class CmdMolDrawFrames : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const char * framespec;
  CmdMolDrawFrames(int mol, int rep, const char *frametxt)
  : Command(MOL_DRAWFRAMES), whichMol(mol), repn(rep), framespec(frametxt) {}
};


/// Set the trajectory smoothing window size for the selected representation
class CmdMolSmoothRep : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const int winsize;
  CmdMolSmoothRep(int mol, int rep, int win)
  : Command(MOL_SMOOTHREP), whichMol(mol), repn(rep), winsize(win) {}
};


/// Set the "shown" state for the selected representation
class CmdMolShowRep : public Command {
protected:
  virtual void create_text();
public:
  const int whichMol;
  const int repn;
  const int onoff;
  CmdMolShowRep(int mol, int rep, int on)
  : Command(MOL_SHOWREP), whichMol(mol), repn(rep), onoff(on) {}
};

#endif
