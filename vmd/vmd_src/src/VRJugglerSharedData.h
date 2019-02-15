/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr VRJuggler patches contributed by Martijn Kragtwijk: m.kragtwijk@rug.nl
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VRJugglerSharedData.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific data sharing code for VMD
 ***************************************************************************/
#ifndef VRJUGGLER_SHAREDDATA_H
#define VRJUGGLER_SHAREDDATA_H

#include <vpr/IO/SerializableObject.h>
#include <vpr/IO/ObjectReader.h>
#include <vpr/IO/ObjectWriter.h>
#include <plugins/ApplicationDataManager/UserData.h>

/** Class to wrap the navigation matrix to share across cluster.
 */
class VRJugglerSharedData : public vpr::SerializableObject
{
public:
   virtual vpr::ReturnStatus readObject(vpr::ObjectReader* reader);

   virtual vpr::ReturnStatus writeObject(vpr::ObjectWriter* writer);

   void appendCommand(char const *);		// append the command

public:
	std::vector<std::string> commandStrings;      /* this is the data that is shared */
													  /* could use a queue? */
};

/**
 * Class to control all navigation. 
 */
class OsgNavigator
{
  

public:
   OsgNavigator()
   {;}

   void init();

   void update(float delta);

private:

   /** Current postion (as userdate for sharing across cluster */
   cluster::UserData<VRJugglerSharedData>  mNavData;
};

#endif /* VRJUGGLER_SHAREDDATA_H */
