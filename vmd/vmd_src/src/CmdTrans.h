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
 *      $RCSfile: CmdTrans.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.33 $      $Date: 2019/01/17 21:20:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Command objects for transforming the current scene.
 *
 ***************************************************************************/
#ifndef CMDTRANS_H
#define CMDTRANS_H

#include "Command.h"
#include "Matrix4.h"

/// apply a matrix transformation to the current scene
class CmdRotMat : public Command {
 public:
   /// is the transformation applied to the current one or is it a replacement?
   enum { BY, TO };
   Matrix4 rotMat;
   int byOrTo;

 protected:
   virtual void create_text(void);

 public:
   CmdRotMat(const Matrix4& newrot, int by_or_to);
};


/// rotate the current scene
class CmdRotate : public Command {
public:
  enum { BY, TO }; ///< enum with how to rotate, 'by' or 'to'
  char axis;       ///< axis to rotate
  float deg;       ///< amount to rotate
  int byOrTo;      ///< set, or add to, rotation?
  int steps;       ///< steps to rotate through; by default this is 1

protected:
  virtual void create_text(void);

public:
  /// first constructor: a single rotation, no smooth transition
  CmdRotate(float a, char ax, int by_or_to);
  
  /// second constructor: a smooth rotation in given increments ...
  /// only useful for "by" rotations.  If "to" is given to this constructor,
  /// a single-step rotation is done 
  CmdRotate(float a, char ax, int by_or_to, float inc);
};


/// translate the current scene
class CmdTranslate : public Command {
public:
  enum { BY, TO }; ///< enum with how to translate, 'by' or 'to'
  float x, y, z;   ///< amount to translate
  int byOrTo;      ///< set, or add to, translation?
  
protected:
  virtual void create_text(void);

public:
  CmdTranslate(float nx, float ny, float nz, int by_or_to);
};


/// scale the current scene
class CmdScale : public Command {
public:
  enum { BY, TO }; ///< enum with how to scale, 'by' or 'to'
  float s;         ///< amount to scale
  int byOrTo;      ///< set, or multiply, scaling?
  
protected:
  virtual void create_text(void);

public:
  CmdScale(float ns, int by_or_to);
};


/// rock the current scene
class CmdRockOn : public Command {
public:
  char axis;       ///< axis to rock
  float deg;       ///< amount to rock
  int steps;       ///< steps to rock (if < 0, continuous)

protected:
  virtual void create_text(void);

public:
  CmdRockOn(float a, char ax, int nsteps);
};


/// stop rocking the current scene
class CmdRockOff : public Command {
protected:
  virtual void create_text(void);

public:
  CmdRockOff() ;
};

#endif

