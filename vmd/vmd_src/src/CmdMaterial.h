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
 *      $RCSfile: CmdMaterial.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Commands for manipulating materials
 ***************************************************************************/

#ifndef CMD_MATERIAL_H__
#define CMD_MATERIAL_H__

#include "Command.h"

/// Add a new material
class CmdMaterialAdd : public Command {
private:
  char *name, *copy;
 
protected:
  virtual void create_text(void);

public:
  CmdMaterialAdd(const char *, const char *copyfrom);
  ~CmdMaterialAdd(void);
};


/// Rename an existing material
class CmdMaterialRename : public Command {
private:
  char *oldname, *newname;
 
protected:
  virtual void create_text(void);

public:
  CmdMaterialRename(const char *oldnm, const char *newnm);
  ~CmdMaterialRename(void);
};


/// Change a property of an existing material
class CmdMaterialChange : public Command {
private:
  char *name;
  int property;
  float val;

protected:
  virtual void create_text(void);

public:
  CmdMaterialChange(const char *, int, float);
  ~CmdMaterialChange(void);
};


/// Delete a material
class CmdMaterialDelete : public Command {
private:
  char *name;

protected:
  virtual void create_text();

public:
  CmdMaterialDelete(const char *);
  ~CmdMaterialDelete();
};


/// Reset a material to defaults 
class CmdMaterialDefault : public Command {
protected:
  int ind;
  virtual void create_text();
public:
  CmdMaterialDefault(int matind)
  : Command(MATERIAL_DEFAULT), ind(matind) {}
};

#endif
