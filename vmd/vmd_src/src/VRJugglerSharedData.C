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
 *      $RCSfile: VRJugglerSharedData.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $       $Date: 2019/01/17 21:21:02 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * a VRJuggler specific data sharing code for VMD
 ***************************************************************************/
#include <Inform.h>
#include "VRJugglerSharedData.h"

vpr::ReturnStatus VRJugglerSharedData::readObject(vpr::ObjectReader* reader)
{
	unsigned numStrings = reader->readUint32();			// read nr of strings
    commandStrings.clear();								// clear old content (!)
	for (unsigned i=0;i<numStrings;i++ )
    {
		std::string command = reader->readString();// read the strings themselves
		//msgInfo << "read string:" << command.c_str() << sendmsg;
		commandStrings.push_back(command);
	}
    return vpr::ReturnStatus::Succeed;
}

vpr::ReturnStatus VRJugglerSharedData::writeObject(vpr::ObjectWriter* writer)
{
	unsigned numStrings = commandStrings.size();
	writer->writeUint32(numStrings);					// write nr of strings
    for (unsigned i=0;i<numStrings;i++ )
    {
        writer->writeString(commandStrings[i]);	// write the strings themselves
    }
	commandStrings.clear();								// remove all content so we get a fresh start next frame!
    return vpr::ReturnStatus::Succeed;
}


void OsgNavigator::init()
{
    vpr::GUID new_guid("44ab594b-1dfb-40c1-8bf3-9af7b7c0ac8a");
    mNavData.init(new_guid);

    // Could hardcode the hostname like the following, but it is better to rely 
    // on the ApplicationData configuration element to get the hostname.
    //std::string hostname = "crash";
    //mNavData.init(new_guid, hostname);
}

void OsgNavigator::update(float delta)
{
    if(!mNavData.isLocal())
    {
    // std::cout << "Data is NOT local, returning.\n";
        return;
    }
    else
    {
//std::cout << "Data IS local.\n";
    }

	// do something
}



