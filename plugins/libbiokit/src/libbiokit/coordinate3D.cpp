/*
 * University of Illinois Open Source License
 * Copyright 2007 Luthey-Schulten Group, 
 * All rights reserved.
 * 
 * Developed by: Luthey-Schulten Group
 * 			     University of Illinois at Urbana-Champaign
 * 			     http://www.scs.illinois.edu/~schulten
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the Software), to deal with 
 * the Software without restriction, including without limitation the rights to 
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
 * of the Software, and to permit persons to whom the Software is furnished to 
 * do so, subject to the following conditions:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimers.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimers in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * - Neither the names of the Luthey-Schulten Group, University of Illinois at
 * Urbana-Champaign, nor the names of its contributors may be used to endorse or
 * promote products derived from this Software without specific prior written
 * permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL 
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS WITH THE SOFTWARE.
 *
 * Author(s): John Eargle, Elijah Roberts
 */

#include <stdlib.h>
#include "coordinate3D.h"


Coordinate3D::Coordinate3D()
{
    unset();
}

Coordinate3D::Coordinate3D(float x, float y, float z)
{
    set(x, y, z);
}

Coordinate3D::Coordinate3D(const Coordinate3D& copyFrom)
{
    valid = copyFrom.valid;
    coords[0] = copyFrom.coords[0];
    coords[1] = copyFrom.coords[1];
    coords[2] = copyFrom.coords[2];
}

bool Coordinate3D::operator==(const Coordinate3D& coord) const
{

  if (!valid || !coord.valid || coord.coords[0] != coords[0] || coord.coords[1] != coords[1] || coord.coords[2] != coords[2])
    return false;
    
  return true;
}


float Coordinate3D::getX() {

  return coords[0];
}


float Coordinate3D::getY() {
  
  return coords[1];
}


float Coordinate3D::getZ() {

  return coords[2];
}

void Coordinate3D::set(float x, float y, float z)
{
    valid=1;
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
}

void Coordinate3D::unset()
{
    valid=0;
    coords[0] = 0.0;
    coords[1] = 0.0;
    coords[2] = 0.0;
}



// getDistanceTo - returns the Euclidean distance between this and coord;
//   returns -1.0 on error;
float Coordinate3D::getDistanceTo(Coordinate3D& coord) {

  if (!valid || !coord.valid) {
    return -1.0;
  }

  float distance = sqrt( pow(coords[0] - coord.getX(),2) +
			 pow(coords[1] - coord.getY(),2) +
			 pow(coords[2] - coord.getZ(),2) );
  
  return distance;
}

