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
 *	$RCSfile: P_SensorConfig.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.41 $	$Date: 2019/01/17 21:21:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "P_SensorConfig.h"
#include "Inform.h"
#include "utilities.h"


// This function goes through all the vmdsensors files, opens them and calls
// a functio defined by "behavior" for each one, and then closes the files.
// It passes along the pointer "params" to the called function.
// This mechanism is a bit clumsy, but it best accomodated the re-use of
// the previous code.
enum Scanning_Behaviors { PARSE_FOR_NAMES, PARSE_FOR_DEVICE };

void SensorConfig::ScanSensorFiles (int behavior, SensorConfig *sensor, void* params) {
  JString sensorfile;
  FILE *file;
  int found_sensorfile=FALSE; //keeps track of whether we found a sensor file

#if !defined(_MSC_VER)
  // First try looking for a .vmdsensors file in the user's
  // UNIX home directory 
  sensorfile = getenv("HOME");
  sensorfile += "/.vmdsensors";
  if ( (file = fopen(sensorfile,"r")) ) {
    found_sensorfile=TRUE;
    switch (behavior) {
      case PARSE_FOR_DEVICE:
        sensor->parseconfigfordevice(file, params);
        break;
      case PARSE_FOR_NAMES:
        parseconfigfornames(file, params);
        break;
    }
    fclose(file);
  }
#endif
  
  // Then, try path pointed to by the VMDSENSORS env. variable  
  if (!found_sensorfile) {
    char *sensorfile_str = getenv("VMDSENSORS");
    if (sensorfile_str && (file = fopen(sensorfile_str,"r"))) {
      found_sensorfile=TRUE;
      switch (behavior) {
        case PARSE_FOR_DEVICE:
          sensor->parseconfigfordevice(file, params);
          break;
        case PARSE_FOR_NAMES:
          parseconfigfornames(file, params);
          break;
      }
      fclose(file);
    }
  }
  
  // If we couldn't find a file in the home dir or VMDSENSORS env. variable
  // then try finding one in the VMD installation area.  
  if (!found_sensorfile) {
    sensorfile = getenv("VMDDIR");
#if defined(_MSC_VER)
    sensorfile += "\\.vmdsensors";
#else
    sensorfile += "/.vmdsensors";
#endif
    if ((file = fopen(sensorfile,"r"))) {
      switch (behavior) {
        case PARSE_FOR_DEVICE:
          sensor->parseconfigfordevice(file, params);
          break;
        case PARSE_FOR_NAMES:
          parseconfigfornames(file, params);
          break;
      }
      fclose(file);
    }
  }
  
}


static int splitline(FILE *f, JString *argv, int maxarg) {
  int argc, pos;
  char buf[128], word[128];
  memset(buf, 0, sizeof(buf));
  memset(word, 0, sizeof(word));

  if(!fgets(buf, sizeof(buf), f)) return -1;

  argc = 0;
  pos = 0;
  while (argc<maxarg && pos<100 && buf[pos]!=0 && buf[pos]!='\n'
	&& buf[pos]!='\r') {

    if (buf[pos]!=' ' && buf[pos]!='\t') {
      sscanf(buf+pos, "%99s", word); 
      pos += strlen(word);
      argv[argc] = (JString)word;
      argc++;
    } else {
      pos++;
    }
  }
  
  return argc;
}


static int need_args(int argc,int need, int line) {
  if (need!=argc) {
    msgErr << "SensorConfig: Wrong number of arguments at line " << line << "." << sendmsg;
    msgErr << "Expected " << need << ", got " << argc << "." << sendmsg;
    return 1;
  }
  return 0;
}

float SensorConfig::getfloat(const char *from, float defalt) {
  int ret;
  float f;
  ret=sscanf(from,"%f",&f);
  if(!ret) {
    msgErr << "SensorConfig: Error parsing float at line " << line << sendmsg;
    return defalt;
  }
  return f;
}
 
int SensorConfig::needargs(int argc,int need) {
  return need_args(argc,need,line);
}


void SensorConfig::parseconfigfordevice(FILE *file, void *) {
  // now we have an open configuration file, search for device lines
  int found=FALSE, argc;
  line = 0;
  JString argv[20];
  while ((argc=splitline(file, argv, 20))>=0) {
    line++;
    
    if (!argc) continue;
    if (argv[0][0]=='#') continue; // this line is a comment

    if (!compare(argv[0],"device")) { // found a device
      if (found) break; // done reading info about our device
      if (needargs(argc,3)) return;
      if (!compare(argv[1],device)) { // found our device
        // Comment out chatty options for now, until we can reduce the number
        // of times we have to read the .vmdsensors file.
        // msgInfo << "Found device '" << argv[1] << "'" << " on line " << line << "." << sendmsg;
	      found = TRUE;
	      USL = argv[2];
	      continue; // we start parsing the options on the next line
      }
    }

    if (!found) continue;

    // Comment out chatty options for now, until we can reduce the number
    // of times we have to read the .vmdsensors file.
    // msgInfo << "Reading option '" <<  argv[0] << "'" << sendmsg;

    if (!compare(argv[0],"scale")) {
      if (needargs(argc,2)) return;
      scale = getfloat(argv[1],1);
    }
    else if (!compare(argv[0],"maxforce")) {
      // XXX Maxforce doesn't seem to work at the moment, causing dangerously
      // high forces to be sent to the haptic device.  Disabling for now until
      // the reason is discovered.
#if 0
      if (needargs(argc,2)) return;
      maxforce = getfloat(argv[1],1);
#else
      msgInfo << "Sorry, maxforce parameter not currently implemented." 
              << sendmsg;
#endif
    }
    else if (!compare(argv[0],"offset")) {
      int i;
      if (needargs(argc,4)) return;
      for (i=0;i<3;i++)
	      offset[i] = getfloat(argv[1+i],0);
    }
    else if (!compare(argv[0],"rot")) {
      int i;
      if (needargs(argc,11)) return;
      if (!compare(argv[1],"right"))
	      for (i=0;i<9;i++)
	        right_rot.mat[i+i/3] = getfloat(argv[2+i],right_rot.mat[i+i/3]);
      else
	      for (i=0;i<9;i++)
	        left_rot.mat[i+i/3] = getfloat(argv[2+i],right_rot.mat[i+i/3]);
    }
    else msgErr << "Error: Unrecognized tool option on line " << line << sendmsg;
  }

  if (USL=="") msgErr << "Device " << device << " not found." << sendmsg;
  else parseUSL();
}


SensorConfig::SensorConfig(const char *thedevice) {

  if (thedevice==NULL) return;

  USL = "";
  strcpy(nums,"");
  strcpy(name,"");
  strcpy(place,"");
  strcpy(type,"");
  strcpy(device,thedevice);
  offset[0] = offset[1] = offset[2] = 0;
  scale = 1;
  maxforce = -1; // default is to not enforce a maximum
  right_rot.identity();
  left_rot.identity();
  
  ScanSensorFiles(PARSE_FOR_DEVICE, this, NULL);
}


int SensorConfig::parseUSL() {
  int ret;
  strcpy(nums,"0"); // fill in default of zero

  ret = sscanf(USL,"%20[^:]://%100[^/]/%100[^:]:%101s",
	       		   type,     place,   name,    nums);
  
  if(ret<3) {
    msgErr << "USL on line " << line << " is not of the form: "
	   << "type://place/name[:num,num,...]" << sendmsg;
    return 0; // if we get anything less than blah://blah/blah
  }

  read_sensor_nums();
  
  return 1;
}

void SensorConfig::read_sensor_nums() {
  char *s=strdup(nums);
  char *r=s;

  for (int cur=0; cur<= 200; cur++) {
    if(s[cur]==',' || s[cur]==0) {
      int tmp = 0;
      sscanf(r,"%d",&tmp);
      sensors.append(tmp);
      r = s + cur + 1;
      if(s[cur]==0) break;
      s[cur]=0;
    }
  }

  free(s);
}


// Used as a callback in SensorConfig::getnames()
void SensorConfig::parseconfigfornames(FILE *f, void *ret_void) {
  int argc, line=0;
  JString argv[20];
  
  while ((argc=splitline(f, argv, 20))>=0) {
    line++;
    
    if(!argc) continue;
    if(argv[0][0]=='#') continue; // this line is a comment

    if(!compare(argv[0],"device")) { // found a device
      if(need_args(argc,3,line)) continue;
      JString *newname = new JString(argv[1]);
      ((ResizeArray<JString *>*) ret_void)->append(newname);
    }
  }
}


ResizeArray<JString *> *SensorConfig::getnames() {
  ResizeArray<JString *> *ret = new ResizeArray<JString *>;
  
  ScanSensorFiles(PARSE_FOR_NAMES, NULL, (void*)ret);
  return ret;
}


const char *SensorConfig::getUSL() const {
  return USL;
}

float SensorConfig::getscale() const {
  return scale;
}

float SensorConfig::getmaxforce() const {
  return maxforce;
}

const float *SensorConfig::getoffset() const {
  return offset;
}

const Matrix4 *SensorConfig::getright_rot() const {
  return &right_rot;
}

const Matrix4 *SensorConfig::getleft_rot() const {
  return &left_rot;
}

const char *SensorConfig::getdevice() const { return device; }
const char *SensorConfig::gettype() const { return type; }
const char *SensorConfig::getplace() const { return place; }
const char *SensorConfig::getname() const { return name; }
const char *SensorConfig::getnums() const { return nums; }
const ResizeArray<int> *SensorConfig::getsensors() const { return &sensors; }

SensorConfig::~SensorConfig() {
}

int SensorConfig::have_one_sensor() const {
  if (sensors.num() != 1) {
    msgErr << "Please specify exactly one sensor." << sendmsg;
    return 0;
  }
  return 1;
}
 
void SensorConfig::make_vrpn_address(char *buf) const {
  // As of VRPN 7.15 enforcing TCP-only connections is supported
  // by appending @tcp to the device name and using an USL in 
  // format Device@tcp://machine:port instead of Device@machine:port.
  // Create a new style USL when the name contains @tcp. 
  if (strstr(name,"@tcp"))
    sprintf(buf, "%s://%s", name, place);
  else
    sprintf(buf, "%s@%s", name, place);
}
   
int SensorConfig::require_local() const {
  if(strcmp(place,"local")) {
    msgErr << "Sorry, this local device requires place name \"local\"."
            << sendmsg;
    return 0;
  }
  return 1;
}

int SensorConfig::require_cave_name() const {
  if(!require_local()) return 0;
  if(strcmp(name,"cave")) {
    msgErr << "Sorry, the device name for the CAVE is \"cave\"." << sendmsg;
    return 0;
  }
  return 1;
}

int SensorConfig::require_freevr_name() const {
  if(!require_local()) return 0;
  if(strcmp(name,"freevr")) {
    msgErr << "Sorry, the device name for FreeVR is \"freevr\"." << sendmsg;
    return 0;
  }
  return 1;
}

