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
 *      $RCSfile: CmdMaterial.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.32 $       $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Commands for manipulating materials
 ***************************************************************************/

#include "CmdMaterial.h"
#include "MaterialList.h" // for MAT_XXX definitions
#include "utilities.h"
#include "config.h"
#include "Inform.h"
#include <stdlib.h>
#include <ctype.h>

///// Add

CmdMaterialAdd::CmdMaterialAdd(const char *s, const char *copyfrom)
: Command(MATERIAL_ADD) {
  name = copy = NULL;
  if (s)
    name = stringdup(s);
  if (copyfrom)
    copy = stringdup(copyfrom);
}

void CmdMaterialAdd::create_text(void) {
  *cmdText << "material add";
  if (name)
    *cmdText << " " << name;
  if (copy)
    *cmdText << " copy " << copy;
  *cmdText << ends;
}

CmdMaterialAdd::~CmdMaterialAdd(void) {
  delete [] name;
  delete [] copy;
}

///// Rename

CmdMaterialRename::CmdMaterialRename(const char *oldn, const char *newn) 
: Command(MATERIAL_RENAME) {
  oldname = stringdup(oldn);
  newname = stringdup(newn);
}

void CmdMaterialRename::create_text(void) {
  *cmdText << "material rename " << oldname << " " << newname << ends;
}

CmdMaterialRename::~CmdMaterialRename(void) {
  delete [] oldname;
  delete [] newname;
}

///// Change

CmdMaterialChange::CmdMaterialChange(const char *s, int p, float v)
: Command(MATERIAL_CHANGE) {
  name = stringdup(s);
  property = p;
  val = v;
}

void CmdMaterialChange::create_text(void) {
  *cmdText << "material change ";
  switch (property) {
    case MAT_AMBIENT: *cmdText << "ambient "; break;
    case MAT_SPECULAR: *cmdText << "specular "; break;
    case MAT_DIFFUSE: *cmdText << "diffuse "; break;
    case MAT_SHININESS: *cmdText << "shininess "; break;
    case MAT_MIRROR: *cmdText << "mirror "; break;
    case MAT_OPACITY: *cmdText << "opacity "; break;
    case MAT_OUTLINE: *cmdText << "outline "; break;
    case MAT_OUTLINEWIDTH: *cmdText << "outlinewidth "; break;
    case MAT_TRANSMODE: *cmdText << "transmode "; break;
  }
  *cmdText << name << " " << val << ends;
}

CmdMaterialChange::~CmdMaterialChange(void) {
  delete [] name;
}

///// Delete
CmdMaterialDelete::CmdMaterialDelete(const char *s) 
: Command(MATERIAL_DELETE) {
  name = stringdup(s);
}

void CmdMaterialDelete::create_text() {
  *cmdText << "material delete " << name << ends;
}

CmdMaterialDelete::~CmdMaterialDelete() {
  delete [] name;
}

void CmdMaterialDefault::create_text() {
  *cmdText << "material default " << ind << ends;
}

