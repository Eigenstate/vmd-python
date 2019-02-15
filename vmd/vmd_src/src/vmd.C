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
 *	$RCSfile: vmd.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.104 $	$Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main program entry points.
 *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#if defined(_MSC_VER)
#include "win32vmdstart.h"
#endif

#if !defined(VMDNOMACBUNDLE) && defined(__APPLE__)
#include "macosxvmdstart.h"
#endif

#include "vmd.h"
#include "VMDApp.h"
#include "utilities.h"  // for TRUE, and for string processing utilities
#include "config.h"     // for compiled-in defaults
#include "WKFThreads.h"
#include "Inform.h"
#include "CommandQueue.h"
#include "TextEvent.h"
#include "MaterialList.h" // for MAT_XXX definitions
#include "SymbolTable.h"  // for processing atomselection macros

#include "ProfileHooks.h" // NVTX profiling

#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#ifdef VMDFLTK
#include <FL/Fl.H>
#endif

#ifdef VMDOPENGL        // OpenGL-specific files
#ifdef VMDCAVE          // CAVE-specific files
#include "cave_ogl.h"
#include "CaveRoutines.h"
#endif
#ifdef VMDFREEVR        // FreeVR-specific files
#include "freevr.h"
#include "FreeVRRoutines.h"
#endif
#endif

#ifdef VMDMPI
#include "VMDMPI.h"
#endif

#ifdef VMDTCL
#include <tcl.h>
#include <signal.h>

//
// set up signal handlers
//
static Tcl_AsyncHandler tclhandler;

extern "C" {
  typedef void (*sighandler_t)(int);

  void VMDTclSigHandler(int) {
    Tcl_AsyncMark(tclhandler);
  }

  int VMDTclAsyncProc(ClientData, Tcl_Interp *, int) {
    signal(SIGINT, (sighandler_t) VMDTclSigHandler);
    return TCL_ERROR;
  }

}
#endif // VMDTCL


/// Application-level routine for initializing Tcl.  This should be called
/// before creating any instances of VMDApp.  Returns pointer to argv0.
static const char *vmd_initialize_tcl(const char *argv0) {
#ifdef VMDTCL

#if defined(_MSC_VER)
  static char buffer[MAX_PATH +1];
  char *p;

  // get full pathname to VMD executable
  GetModuleFileName(NULL, buffer, sizeof(buffer));

  // convert filename to Tcl-format
  for (p = buffer; *p != '\0'; p++) {
    if (*p == '\\') {
      *p = '/';
    }
  }

  Tcl_FindExecutable(buffer); 
  return buffer;
#else
  if (argv0) {
    Tcl_FindExecutable(argv0);
  }
  return argv0;
#endif

#else  // no Tcl
  return "";
#endif

}

/// Call this after deleting last VMDApp instance, to ensure that all event
/// handlers (like Tk) shut down cleanly.
static void vmd_finalize_tcl() {
#ifdef VMDTCL
  Tcl_Finalize();
#endif
}


extern "C" {

// function pointer to shared memory allocator/deallocator
void * (*vmd_alloc)(size_t);
void (*vmd_dealloc)(void *);
void * (*vmd_realloc)(void *, size_t);

// function to resize allocations depending on whether or not the allocator
// provides a realloc() function or not.
void * vmd_resize_alloc(void *ptr, size_t oldsize, size_t newsize) {
  void *newptr=NULL;

  if (ptr == NULL) { 
    newptr = vmd_alloc(newsize);
    return newptr; 
  }

  if (vmd_realloc != NULL) {
    newptr = vmd_realloc(ptr, newsize);
  }

  if (newptr == NULL) {
    newptr = vmd_alloc(newsize);
    if (newptr != NULL) {
      memcpy(newptr, ptr, oldsize);
      vmd_dealloc(ptr);
    }
  }

  return newptr;
}

} // end of extern "C"

extern void VMDupdateFltk() {
#ifdef VMDFLTK
#if (defined(__APPLE__)) && defined(VMDTCL)
  // don't call wait(0) since this causes Tcl/Tk to mishandle events
  Fl::flush();
#else
  Fl::wait(0);
#endif
#endif
}

/***************************************************************************
 enumerates for different initial status variables, such as the type of
 display to use at startup
 ***************************************************************************/
  
// display types at startup
// For a complete case we should have:
//   FLTK on? (or Tk? or MFC?)
//   GL or OpenGL? (or other??)
//   use the CAVE?
// and be smart about choosing the GL/OpenGL CAVE options, etc.
enum DisplayTypes { 
  DISPLAY_WIN,         // standard display in a window.
  DISPLAY_WINOGL,      // use OpenGL, if a valid option
  DISPLAY_OGLPBUFFER,  // use OpenGL off-screen rendering, if a valid option
  DISPLAY_CAVE,        // Display in the CAVE, no FLTK
  DISPLAY_TEXT,        // Don't use a graphics display
  DISPLAY_CAVEFORMS,   // Use the CAVE _and_ FLTK
  DISPLAY_FREEVR,      // Use the FREEVR, no FLTK
  DISPLAY_FREEVRFORMS, // Use the CAVE _and_ FLTK
  NUM_DISPLAY_TYPES
};

static const char *displayTypeNames[NUM_DISPLAY_TYPES] = {
  "WIN",  "OPENGL", "OPENGLPBUFFER", 
  "CAVE", "TEXT", "CAVEFORMS", "FREEVR", "FREEVRFORMS"
};

#define DISPLAY_USES_WINDOW(d) ((d) == DISPLAY_WIN || (d) == DISPLAY_WINOGL)
#define DISPLAY_USES_CAVE(d) ((d) == DISPLAY_CAVE || (d) == DISPLAY_CAVEFORMS)
#define DISPLAY_USES_FREEVR(d) ((d) == DISPLAY_FREEVR || (d) == DISPLAY_FREEVRFORMS)
#define DISPLAY_USES_GUI(d) (DISPLAY_USES_WINDOW(d) || (d) == DISPLAY_CAVEFORMS || (d) == DISPLAY_FREEVRFORMS)

// how to show the title
enum TitleTypes { 
  TITLE_OFF, TITLE_ON, NUM_TITLE_TYPES 
};

static const char *titleTypeNames[NUM_TITLE_TYPES] = {
  "OFF", "ON"
};

// display options set at startup time
static int   showTitle      = INIT_DEFTITLE;   
static int   which_display  = DISPLAY_OGLPBUFFER;
static float displayHeight  = INIT_DEFHEIGHT;
static float displayDist    = INIT_DEFDIST;
static int   displaySize[2] = { -1, -1 };
static int   displayLoc[2]  = { -1, -1 };

// filenames for init and startup files
static const char *startupFileStr;
static const char *beginCmdFile;

// Change the text interpreter to Python before processing commands. 
// (affects the "-e" file, not the .vmdrc or other files.
static int cmdFileUsesPython;

// Filename parsing on the command line works as follows.  The loadAsMolecules
// flag is either on or off.  When on, each filename parsed will be loaded as
// a separate molecule.  When off, all subsequent files will be loaded into
// the same molecule.  The flag is turned on with "-m" and off with
// "-f".  The default state is off.  "-m" and "-f" can be
// specified multiple times on the command line.
static int loadAsMolecules = 0;
static int startNewMolecule = 1;
static ResizeArray<int> startNewMoleculeFlags;
static ResizeArray<const char *>initFilenames;
static ResizeArray<const char *>initFiletypes;

// Miscellaneous stuff
static int eofexit  = 0;       
static int just_print_help = 0;
static ResizeArray<char *>customArgv;

// forward declaration of startup processing routines
static void VMDtitle();
static void VMDGetOptions(int, char **, int mpienabled);


int VMDinitialize(int *argc, char ***argv, int mpienabled) {
  int i;

  PROFILE_PUSH_RANGE("VMDinitialize()", 1);

#if defined(VMDMPI)
  if (mpienabled) {
    // hack to fix up env vars if necessary
    for (i=0; i<(*argc); i++) {
      if(!strupcmp((*argv)[i], "-vmddir")) {
        if((*argc) > (i + 1)) {
          setenv("VMDDIR", (*argv)[++i], 1);
        } else {
          msgErr << "-vmddir must specify a fully qualified path." << sendmsg;
        }
      }
    }

    vmd_mpi_init(argc, argv);  // initialize MPI, fix up env vars, etc.
  }
#endif

#if defined(_MSC_VER) && !defined(VMDSEPARATESTARTUP)
  win32vmdstart(); // get registry info etc
#endif

#if !defined(VMDNOMACBUNDLE) && defined(__APPLE__)
  macosxvmdstart(*argc, *argv); // get env variables etc
#endif

  // Tell Tcl where the executable is located
  const char *argv0 = vmd_initialize_tcl((*argv)[0]);

#ifdef VMDTCL
  // register signal handler
  tclhandler = Tcl_AsyncCreate(VMDTclAsyncProc, (ClientData)NULL); 
  signal(SIGINT, (sighandler_t) VMDTclSigHandler);  
#endif

  // Let people know who we are.
  VMDtitle();

  // Tell the user what we think about the hardware we're running on.
  // If VMD is compiled for MPI, then we don't print any of the normal
  // standalone startup messages and instead we use the special MPI-specific
  // node scan startup messages only.
  if (!mpienabled) {
#if defined(VMDTHREADS) 
    int vmdnumcpus = wkf_thread_numprocessors();
    msgInfo << "Multithreading available, " << vmdnumcpus 
            << ((vmdnumcpus > 1) ? " CPUs" : " CPU") <<  " detected." 
            << sendmsg;

    wkf_cpu_caps_t cpucaps;
    if (!wkf_cpu_capability_flags(&cpucaps)) {
      msgInfo << "  CPU features: ";
      if (cpucaps.flags & CPU_SSE2)
        msgInfo << "SSE2 ";
      if (cpucaps.flags & CPU_AVX)
        msgInfo << "AVX ";
      if (cpucaps.flags & CPU_AVX2)
        msgInfo << "AVX2 ";
      if (cpucaps.flags & CPU_FMA)
        msgInfo << "FMA ";

      if ((cpucaps.flags & CPU_KNL) == CPU_KNL) {
        msgInfo << "KNL:AVX-512F+CD+ER+PF ";
      } else {
        if (cpucaps.flags & CPU_AVX512F)
          msgInfo << "AVX512F ";
        if (cpucaps.flags & CPU_AVX512CD)
          msgInfo << "AVX512CD ";
        if (cpucaps.flags & CPU_AVX512ER)
          msgInfo << "AVX512ER ";
        if (cpucaps.flags & CPU_AVX512PF)
          msgInfo << "AVX512PF ";
      }
      msgInfo << sendmsg;
    }
#endif

    long vmdcorefree = vmd_get_avail_physmem_mb();
    if (vmdcorefree >= 0) {
      long vmdcorepcnt = vmd_get_avail_physmem_percent();

      // on systems with large physical memory (tens of GB) we
      // report gigabytes of memory rather than megabytes
      if (vmdcorefree > 8000) {
        vmdcorefree += 512; // round up to nearest GB 
        vmdcorefree /= 1024; 
        msgInfo << "Free system memory: " << vmdcorefree 
                << "GB (" << vmdcorepcnt << "%)" << sendmsg;
      } else {
        msgInfo << "Free system memory: " << vmdcorefree 
                << "MB (" << vmdcorepcnt << "%)" << sendmsg;
      }
    }
  }

  // Read environment variables and command line options.
  // Initialize customArgv with just argv0 to avoid problems with
  // Tcl extension.
  customArgv.append((char *)argv0);
  VMDGetOptions(*argc, *argv, mpienabled); 

#if (!defined(__APPLE__) && !defined(_MSC_VER)) && (defined(VMDOPENGL) || defined(VMDFLTK))
  // If we're using X-windows, we autodetect if the DISPLAY environment
  // variable is unset, and automatically switch back to text mode without
  // requiring the user to pass the "-dispdev text" command line parameters
  if ((which_display == DISPLAY_WIN) && (getenv("DISPLAY") == NULL)) {
    which_display = DISPLAY_TEXT;
  }
#endif

#if defined(VMDTKCON)
  vmdcon_init();
  msgInfo << "Using VMD Console redirection interface." << sendmsg;
  // we default to a widget mode console, unless text mode is requested.
  // we don't have an tcl interpreter registered yet, so it is set to NULL.
  // flushing pending messages to the screen, is only in text mode possible.
  if ((which_display == DISPLAY_TEXT) || (which_display == DISPLAY_OGLPBUFFER) 
       || just_print_help) {
    vmdcon_use_text(NULL);
    vmdcon_purge();
  } else {
    vmdcon_use_widget(NULL);
  }
#endif

#ifdef VMDFLTK
  // Do various special FLTK initialization stuff here
  if ((which_display != DISPLAY_TEXT) && (which_display != DISPLAY_OGLPBUFFER)) {
    // Cause FLTK to to use 24-bit color for all windows if possible
    // This must be done before any FLTK windows are shown for the first time.
    if (!Fl::visual(FL_DOUBLE | FL_RGB8)) {
      if (!Fl::visual(FL_RGB8)) {
        Fl::visual(FL_RGB); 
      }
    }

    // Disable the use of the arrow keys for navigating buttons and other
    // non-text widgets, we'll try it out and see how it pans out
    Fl::visible_focus(0);

    // Disable Drag 'n Drop since the only text field in VMD is the
    // atomselection input and DND severely gets in the way there.
    Fl::dnd_text_ops(0);
  }
#endif

  // Quit now if the user just wanted a list of command line options.
  if (just_print_help) {
    vmd_sleep(10);  // This is here so that the user can see the message 
                    // before the terminal/shell exits...

    PROFILE_POP_RANGE();
    return 0;
  }

  // Set up default allocators; these may be overridden by cave or freevr. 
  vmd_alloc   = malloc;  // system malloc() in the default case
  vmd_dealloc = free;    // system free() in the default case
  vmd_realloc = realloc; // system realloc(), set to NULL when not available 

  // check for a CAVE display
  if (DISPLAY_USES_CAVE(which_display)) {
#ifdef VMDCAVE
    // allocate shared memory pool used to communicate with child renderers
    int megs = 2048;
    if (getenv("VMDCAVEMEM") != NULL) {
      megs = atoi(getenv("VMDCAVEMEM"));
    } 
    msgInfo << "Attempting to get " << megs << 
            "MB of CAVE Shared Memory" << sendmsg;
    grab_CAVE_memory(megs);

    CAVEConfigure(argc, *argv, NULL); // configure cave walls and memory use

    // point VMD shared memory allocators to CAVE routines
    vmd_alloc = malloc_from_CAVE_memory;
    vmd_dealloc = free_to_CAVE_memory;
    vmd_realloc = NULL; // no realloc() functionality is available presently
#else
    msgErr << "Not compiled with the CAVE options set." << sendmsg;
    which_display = DISPLAY_WIN;    
#endif
  }

  // check for a FreeVR display
  if (DISPLAY_USES_FREEVR(which_display)) {
#ifdef VMDFREEVR
    int megs = 2048;
    if (getenv("VMDFREEVRMEM") != NULL) {
      megs = atoi(getenv("VMDFREEVRMEM"));
    } 
    msgInfo << "Attempting to get " << megs << 
            "MB of FreeVR Shared Memory" << sendmsg;
    grab_FreeVR_memory(megs); // have to do this *before* vrConfigure() if
                              // we want more than the default shared mem.
    vrConfigure(NULL, NULL, NULL); // configure FreeVR walls

    // point shared memory allocators to FreeVR routines
    vmd_alloc = malloc_from_FreeVR_memory;
    vmd_dealloc = free_to_FreeVR_memory;
    vmd_realloc = NULL; // no realloc() functionality is available presently
#else
    msgErr << "Not compiled with the FREEVR options set." << sendmsg;
    which_display = DISPLAY_WIN;    
#endif
  }

  // return custom argc/argv
  *argc = customArgv.num();
  for (i=0; i<customArgv.num(); i++) {
    (*argv)[i] = customArgv[i];
  }

  PROFILE_POP_RANGE();

  return 1; // successful startup
}

const char *VMDgetDisplayTypeName() {
  return displayTypeNames[which_display];
}

void VMDgetDisplayFrame(int *loc, int *size) {
  for (int i=0; i<2; i++) {
    loc[i] = displayLoc[i];
    size[i] = displaySize[i];
  }
}

void VMDshutdown(int mpienabled) {
  vmd_finalize_tcl();  // after all VMDApp instances are deleted

#ifdef VMDCAVE
  if (DISPLAY_USES_CAVE(which_display)) {  // call the CAVE specific exit
    CAVEExit();
  }
#endif
#ifdef VMDFREEVR
  if (DISPLAY_USES_FREEVR(which_display)) {  // call the FreeVR specific exit
    vrExit();
  }
#endif
#ifdef VMDMPI
  if (mpienabled) {
    vmd_mpi_fini();
  }
#endif
}

static void VMDtitle() {
  msgInfo << VERSION_MSG << "\n";
  msgInfo << "http://www.ks.uiuc.edu/Research/vmd/                         \n";
  msgInfo << "Email questions and bug reports to vmd@ks.uiuc.edu           \n";
  msgInfo << "Please include this reference in published work using VMD:   \n";
  msgInfo << "   Humphrey, W., Dalke, A. and Schulten, K., `VMD - Visual   \n";
  msgInfo << "   Molecular Dynamics', J. Molec. Graphics 1996, 14.1, 33-38.\n";
  msgInfo << "-------------------------------------------------------------\n";
  msgInfo << sendmsg;
}

/////////////////////////  routines  ///////////////////////////////

// look for all environment variables VMD can use, and initialize the
// proper variables.  If an env variable is not found, use a default value.
// ENVIRONMENT VARIABLES USED BY VMD (default values set in config.h):
//	VMDDIR		directory with VMD data files and utility programs
//	VMDTMPDIR	directory in which to put temporary files (def: /tmp)
static void VMDGetOptions(int argc, char **argv, int mpienabled) {
  char *envtxt;

  //
  // VMDDISPLAYDEVICE: which display device to use by default
  // 
  if((envtxt = getenv("VMDDISPLAYDEVICE"))) {
    for(int i=0; i < NUM_DISPLAY_TYPES; i++) {
      if(!strupcmp(envtxt, displayTypeNames[i])) {
        which_display = i;
        break;
      }
    }
  }

  // 
  // VMDTITLE: whether to enable the title screen
  //  
  if((envtxt = getenv("VMDTITLE"))) {
    for(int i=0; i < NUM_TITLE_TYPES; i++) {
      if(!strupcmp(envtxt, titleTypeNames[i])) {
        showTitle = i;
        break;
      }
    }
  }

  //
  // VMDSCRHEIGHT: height of the screen
  //
  if((envtxt = getenv("VMDSCRHEIGHT")))
    displayHeight = (float) atof(envtxt);

  //
  // VMDSCRDIST: distance to the screen
  //
  if((envtxt = getenv("VMDSCRDIST")))
    displayDist = (float) atof(envtxt); 

  // 
  // VMDSCRPOS: graphics window location
  //
  if((envtxt = getenv("VMDSCRPOS"))) {
    char * dispStr = NULL;
    char * dispArgv[64];
    int dispArgc;

    if((dispStr = str_tokenize(envtxt, &dispArgc, dispArgv)) != NULL
                && dispArgc == 2) {
      displayLoc[0] = atoi(dispArgv[0]);
      displayLoc[1] = atoi(dispArgv[1]);
    } else {
      msgErr << "Illegal VMDSCRPOS environment variable setting '" 
             << envtxt << "'." << sendmsg;
    }
    if(dispStr)  delete [] dispStr;
  }

  // 
  // VMDSCRSIZE: graphics window size
  //
  if((envtxt = getenv("VMDSCRSIZE"))) {
    char * dispStr = NULL;
    char * dispArgv[64];
    int dispArgc;
    if((dispStr = str_tokenize(envtxt, &dispArgc, dispArgv)) != NULL
                && dispArgc == 2) {
      displaySize[0] = atoi(dispArgv[0]);
      displaySize[1] = atoi(dispArgv[1]);
 
      // force users to do something that makes sense
      if (displaySize[0] < 100) 
        displaySize[0] = 100; // minimum sane width
      if (displaySize[1] < 100) 
        displaySize[1] = 100; // minimum sane height

    } else {
      msgErr << "Illegal VMDSCRSIZE environment variable setting '" 
             << envtxt << "'." << sendmsg;
    }
    if(dispStr)  delete [] dispStr;
  }

  // initialize variables which indicate how VMD starts up, and
  // parse the command-line options

  // go through the arguments
  int ev = 1;
  while(ev < argc) {
    if(!strupcmp(argv[ev], "-dist")) {
      if(argc > (ev + 1)) {
        displayDist = (float) atof(argv[++ev]);
      } else
        msgErr << "-dist must also specify a distance." << sendmsg;

    } else if(!strupcmp(argv[ev], "-e")) {
      if(argc > (ev + 1)) {
        beginCmdFile = argv[++ev];
      } else
        msgErr << "-e must also specify a filename." << sendmsg;

    } else if(!strupcmp(argv[ev], "-height")) {
      if(argc > (ev + 1)) {
        displayHeight = (float) atof(argv[++ev]);
      } else
        msgErr << "-height must also specify a distance." << sendmsg;

    } else if(!strupcmp(argv[ev], "-pos")) {
      if(argc > (ev + 2) && *(argv[ev+1]) != '-' && *(argv[ev+2]) != '-') {
        displayLoc[0] = atoi(argv[++ev]);
        displayLoc[1] = atoi(argv[++ev]);
      } else
        msgErr << "-pos must also specify an X Y pair." << sendmsg;

    } else if(!strupcmp(argv[ev], "-size")) {
      if(argc > (ev + 2) && *(argv[ev+1]) != '-' && *(argv[ev+2]) != '-') {
        displaySize[0] = atoi(argv[++ev]);
        displaySize[1] = atoi(argv[++ev]);
      } else
        msgErr << "-size must also specify an X Y pair." << sendmsg;

    } else if(!strupcmp(argv[ev], "-startup")) {
      // use next argument as startup config file name
      if(argc > (ev + 1))
        startupFileStr = argv[++ev];
      else
        msgErr << "-startup must also have a new file name specified."
	       << sendmsg;

    } else if(!strupcmp(argv[ev], "-nt")) {
      // do not print out the program title
      showTitle = TITLE_OFF;

    } else if (!strupcmp(argv[ev], "-dispdev")) {  // startup Display
      ev++;
      if (argc > ev) {
        if (!strupcmp(argv[ev], "cave")) {  
          which_display = DISPLAY_CAVE;        // use the CAVE
        } else if (!strupcmp(argv[ev], "win")) {       
          which_display = DISPLAY_WIN;         // use OpenGL, the default
        } else if (!strupcmp(argv[ev], "opengl")) {  
          which_display = DISPLAY_WINOGL;      // use OpenGL if available
        } else if (!strupcmp(argv[ev], "openglpbuffer")) {  
          which_display = DISPLAY_OGLPBUFFER;  // use OpenGLPbuffer if available
        } else if (!strupcmp(argv[ev], "text")) {
          which_display = DISPLAY_TEXT;        // use text console only 
        } else if (!strupcmp(argv[ev], "caveforms")) {
          which_display = DISPLAY_CAVEFORMS;   // use CAVE+Forms
        } else if (!strupcmp(argv[ev], "freevr")) {
          which_display = DISPLAY_FREEVR;      // use FreeVR
        } else if (!strupcmp(argv[ev], "freevrforms")) {
          which_display = DISPLAY_FREEVRFORMS; // use FreeVR+Forms
        } else if (!strupcmp(argv[ev], "none")) {      
          which_display = DISPLAY_TEXT;        // use text console only
        } else {
          msgErr << "-dispdev options are 'win' 'opengl' (default), 'openglpbuffer', 'cave', 'caveforms', 'freevr', 'freevrforms', or 'text | none'" << sendmsg;
        }
      } else {
        msgErr << "-dispdev options are 'win' 'opengl' (default), 'openglpbuffer', 'cave', 'caveforms', 'freevr', 'freevrforms', or 'text | none'" << sendmsg;
      }
    } else if (!strupcmp(argv[ev], "-h") || !strupcmp(argv[ev], "--help")) {
      // print out command-line option summary
      msgInfo << "Available command-line options:" << sendmsg;
      msgInfo << "\t-dispdev <win | cave | text | none> Specify display device";
      msgInfo << sendmsg;
      msgInfo << "\t-dist <d>           Distance from origin to screen";
      msgInfo << sendmsg;
      msgInfo << "\t-e <filename>       Execute commands in <filename>\n";
      msgInfo << "\t-python             Use Python for -e file and subsequent text input\n";
      msgInfo << "\t-eofexit            Exit when end-of-file occurs on input\n";
      msgInfo << "\t-h | --help         Display this command-line summary\n";
      msgInfo << "\t-height <h>         Height of display screen";
      msgInfo << sendmsg;
      msgInfo << "\t-pos <X> <Y>        Lower-left corner position of display";
      msgInfo << sendmsg;
      msgInfo << "\t-nt                 No title display at start" << sendmsg;
      msgInfo << "\t-size <X> <Y>       Size of display" << sendmsg;
      msgInfo << "\t-startup <filename> Specify startup script file" << sendmsg;
      msgInfo << "\t-m                  Load subsequent files as separate molecules\n";
      msgInfo << "\t-f                  Load subsequent files into the same molecule\n";
      msgInfo << "\t<filename>          Load file using best-guess file type\n";
      msgInfo << "\t-<type> <filename>  Load file using specified file type\n";
      msgInfo << "\t-args               Pass subsequent arguments to text interpreter\n";
      msgInfo << sendmsg;
      just_print_help = 1;
    } else if (!strupcmp(argv[ev], "-eofexit")) {  // exit on EOF
      eofexit = 1;
    } else if (!strupcmp(argv[ev], "-node")) { 
      // start VMD process on a cluster node, next parm is node ID..
      ev++; // skip node ID parm
    } else if (!strupcmp(argv[ev], "-webhelper")) { 
      // Unix startup script doesn't run VMD in the background, so that
      // web browsers won't delete files out from under us until it really
      // exits.  We don't do anything special inside VMD itself presently
      // however.
    } else if (!strupcmp(argv[ev], "-python")) {
      cmdFileUsesPython = 1;
    } else if (!strupcmp(argv[ev], "-args")) {
      // pass the rest of the command line arguments, and only those, 
      // to the embedded text interpreters.
      while (++ev < argc)
        customArgv.append(argv[ev]);

    } else if (!strupcmp(argv[ev], "-m")) {
      loadAsMolecules = 1;
      startNewMolecule = 1;
    } else if (!strupcmp(argv[ev], "-f")) {
      loadAsMolecules = 0;
      startNewMolecule = 1;
#ifdef VMDMPI
    } else if (mpienabled && !strupcmp(argv[ev], "-vmddir")) {
      ev++; // skip VMDDIR directory parm, since we already handled this
            // in MPI startup before we got to this loop...
#endif
    } else {
      // any other argument is treated either as a filename or as a 
      // filetype/filename pair of the form -filetype filename.
      const char *filename, *filetype;
      if (argv[ev][0] == '-') {
        // must be filetype/filename pair
        if (argc > ev+1) {
          filetype = argv[ev]+1;
          filename = argv[ev+1];
          ev++;
        } else {
          msgErr << "filetype argument '" << argv[ev] << "' needs a filename."
            << sendmsg;
          ev++;  // because we skip past the ev++ at the bottom of the loop.
          continue; 
        }
      } else {
        // Given just a filename.  The filetype will have to be guessed.
        filename = argv[ev];
        filetype = NULL;
      }
      initFilenames.append(filename);
      initFiletypes.append(filetype);
      startNewMoleculeFlags.append(startNewMolecule);
      if (!loadAsMolecules) startNewMolecule = 0;
    }
    ev++;
  }

  // command-line options have been parsed ... any init status variables that
  // have been given initial values will have flags saying so, and their
  // values will not be changed when the init file(s) is parsed.
}

static int parseColorDefs(const char *path, VMDApp *app) {
  FILE *fd = fopen(path, "rt");
  char buf[256];
  memset(buf, 0, sizeof(buf));

  int success = TRUE;

  if (!fd) {
    msgErr << "Color definitions file '" << path << "' does not exist." << sendmsg;
    return FALSE;
  }
  while (fgets(buf, sizeof(buf), fd)) {
    if (buf[0] == '\0' || buf[0] == '#') continue;
    char first[128], second[128], third[128], fourth[128];
    memset(first, 0, sizeof(first));
    memset(second, 0, sizeof(second));
    memset(third, 0, sizeof(third));
    memset(fourth, 0, sizeof(fourth));

    // handle cases like "Structure {Alpha Helix} purple
    int rc = sscanf(buf, "%s { %s %s %s", first, second, third, fourth);
    if (rc == 4) {
      char *right = strchr(third, '}');
      if (right) *right = '\0';
      strcat(second, " ");
      strcat(second, third);
      if (!app->color_add_item(first, second, fourth)) {
        msgErr << "Failed to add color definition: '" << buf << "'" << sendmsg;
        success = FALSE;
      }
    } else if (sscanf(buf, "%s %s %s", first, second, third) == 3) {
      if (!app->color_add_item(first, second, third)) {
        msgErr << "Failed to add color definition: '" << buf << "'" << sendmsg;
        success = FALSE;
      }
    }
  }
  fclose(fd);
  return success;
}

static int parseMaterialDefs(const char *path, VMDApp *app) {
  FILE *fd = fopen(path, "rt");
  char buf[256];
  int success = TRUE;

  if (!fd) {
    msgErr << "Material definitions file '" << path << "' does not exist." << sendmsg;
    return FALSE;
  }
  while (fgets(buf, sizeof(buf), fd)) {
    if (buf[0] == '\0' || buf[0] == '#') continue;
    char name[100];
    float vals[10];
    int readcount;

    memset(vals, 0, sizeof(vals));
    readcount=sscanf(buf, "%s %f %f %f %f %f %f %f %f %f", 
                     name, vals, vals+1, vals+2, vals+3, vals+4, 
                     vals+5, vals+6, vals+7, vals+8);
    if ((readcount < 7) || (readcount > 10))
      continue; // skip bad material

    if (!app->material_add(name, NULL)) {
      msgErr << "Failed to add material '" << name << "'" << sendmsg;
      success = FALSE;
      continue;
    }
    app->material_change(name, MAT_AMBIENT, vals[0]);
    app->material_change(name, MAT_DIFFUSE, vals[1]);
    app->material_change(name, MAT_SPECULAR, vals[2]);
    app->material_change(name, MAT_SHININESS, vals[3]);
    app->material_change(name, MAT_MIRROR, vals[4]);
    app->material_change(name, MAT_OPACITY, vals[5]);
    app->material_change(name, MAT_OUTLINE, vals[6]);
    app->material_change(name, MAT_OUTLINEWIDTH, vals[7]);
    app->material_change(name, MAT_TRANSMODE, vals[8]);
  }
  fclose(fd);
  return success;
}

static int parseRestypes(const char *path, VMDApp *app) {
  FILE *fd = fopen(path, "rt");
  char buf[256];
  memset(buf, 0, sizeof(buf));
  int success = TRUE;

  if (!fd) {
    msgErr << "Residue types file '" << path << "' does not exist." << sendmsg;
    return FALSE;
  }
  while (fgets(buf, sizeof(buf), fd)) {
    if (buf[0] == '\0' || buf[0] == '#') continue;
    char name[64], type[64];
    memset(name, 0, sizeof(name));
    memset(type, 0, sizeof(type));

    if (sscanf(buf, "%s %s", name, type) != 2) continue;

    if (!app->color_set_restype(name, type)) {
      msgErr << "Failed to add residue type '" << buf << "'" << sendmsg;
      success = FALSE;
    }
  }
  fclose(fd);
  return success;
}

static int parseAtomselMacros(const char *path, VMDApp *app) {
  char buf[256];
  memset(buf, 0, sizeof(buf));

  FILE *fd = fopen(path, "rt");
  if (!fd) {
    msgErr << "Atomselection macro file '" << path << "' does not exist." << sendmsg;
    return FALSE;
  }
  int success= TRUE;
  while (fgets(buf, sizeof(buf), fd)) {
    if (buf[0] == '\0' || buf[0] == '#' || isspace(buf[0])) continue;
    char *macro = strchr(buf, ' ');
    if (!macro) continue;
    *macro = '\0';
    macro++;
    // Remove trailing newline characters
    macro[strcspn(macro, "\r\n")] = 0;
    if (!app->atomSelParser->add_custom_singleword(buf, macro)) {
      msgErr << "Failed to add macro '" << buf << "'" << sendmsg;
      success = FALSE;
    }
  }
  fclose(fd);
  return success;
}

// Read scripts in scripts/vmd
void VMDreadInit(VMDApp *app) {
  char path[4096];
  
  const char *vmddir = getenv("VMDDIR"); 
  if (vmddir == NULL) {
    msgErr << "VMDDIR undefined, startup failure likely." << sendmsg;
#if defined(_MSC_VER)
    vmddir = "c:/program files/university of illinois/vmd";
#else
    vmddir = "/usr/local/lib/vmd";
#endif
  } 
  sprintf(path, "%s/scripts/vmd/colordefs.dat", vmddir);
  if (!parseColorDefs(path, app)) {
    msgErr << "Parsing color definitions failed." << sendmsg;
  }
  sprintf(path, "%s/scripts/vmd/materials.dat", vmddir);
  if (!parseMaterialDefs(path, app)) {
    msgErr << "Parsing material definitions failed." << sendmsg;
  }
  sprintf(path, "%s/scripts/vmd/restypes.dat", vmddir);
  if (!parseRestypes(path, app)) {
    msgErr << "Parsing residue types failed." << sendmsg;
  }
  sprintf(path, "%s/scripts/vmd/atomselmacros.dat", vmddir);
  if (!parseAtomselMacros(path, app)) {
    msgErr << "Parsing atomselection macros failed." << sendmsg;
  }
}

// read in the startup script, execute it, and then execute any other commands
// which might be necessary (i.e. to load any molecules at start)
// This searches for the startup file in the following
// places (and in this order), reading only the FIRST one found:
//		1. Current directory
//		2. Home directory
//		3. 'Default' directory (here, /usr/local/vmd)
// If a name was given in the -startup switch, that file is checked for ONLY.
void VMDreadStartup(VMDApp *app) {
  char namebuf[512], *envtxt;
  int found = FALSE;
  FILE * tfp;
  char *DataPath; // path of last resort to find a .vmdrc file

  // These options were set by environment variables or command line options
  app->display_set_screen_height(displayHeight);
  app->display_set_screen_distance(displayDist);
  app->set_eofexit(eofexit);
  if (showTitle == TITLE_ON && (which_display != DISPLAY_TEXT) && 
      (which_display != DISPLAY_OGLPBUFFER)) {
    app->display_titlescreen();
  }

  if ((envtxt = getenv("VMDDIR")) != NULL)
    DataPath = stringdup(envtxt);
  else
    DataPath = stringdup(DEF_VMDENVVAR);
  stripslashes(DataPath); // strip out ending '/' chars.

  // check if the file is available
  if (startupFileStr) {	// name specified by -startup
    if ((tfp = fopen(startupFileStr, "rb")) != NULL) {
      found = TRUE;
      fclose(tfp);
      strcpy(namebuf, startupFileStr);
    }
  } else {	// search in different directories, for default file
    const char *def_startup = VMD_STARTUP;
    // first, look in current dir
    strcpy(namebuf, def_startup);
    if ((tfp = fopen(namebuf, "rb")) != NULL) {
      found = TRUE;
      fclose(tfp);
    } else {
      // not found in current dir; look in home dir
      if ((envtxt = getenv("HOME")) != NULL)
        strcpy(namebuf, envtxt);
      else
        strcpy(namebuf, ".");
      strcat(namebuf, "/");
      strcat(namebuf, def_startup);
      if ((tfp = fopen(namebuf, "rb")) != NULL) {
        found = TRUE;
        fclose(tfp);
      } else {
        // not found in home dir; look in default dir
	strcpy(namebuf, DataPath);
	strcat(namebuf, "/");
	strcat(namebuf, def_startup);
        if ((tfp = fopen(namebuf, "rb")) != NULL) {
          found = TRUE;
          fclose(tfp);
	}
      }
    }
  }
  delete [] DataPath; DataPath = NULL;

  //
  // execute any commands needed at start
  //
  
  PROFILE_PUSH_RANGE("VMDreadStartup(): process cmd args", 4);

  // read in molecules requested via command-line switches
  FileSpec spec;
  spec.waitfor = -1; // wait for all files to load before proceeding
  int molid = -1;    // set sentinel value to determine if files were loaded

  if (startNewMoleculeFlags.num() > 0) {
    msgInfo << "File loading in progress, please wait." << sendmsg;
  }

  for (int i=0; i<startNewMoleculeFlags.num(); i++) {
    const char *filename = initFilenames[i];
    const char *filetype = initFiletypes[i];
    if (!filetype) {
      filetype = app->guess_filetype(filename);
      if (!filetype) {
        // assume pdb 
        msgErr << "Unable to determine file type for file '"
          << filename << "'.  Assuming pdb." << sendmsg;
        filetype = "pdb";
      }
    }
    if (startNewMoleculeFlags[i]) {
      molid = app->molecule_load(-1, filename, filetype, &spec);
    } else {
      molid = app->molecule_load(molid, filename, filetype, &spec);
    }
    if (molid < 0) {
      msgErr  << "Loading of startup molecule files aborted." << sendmsg;
      break;
    }
  }

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMDreadStartup(): process " VMD_STARTUP, 3);

  // if the startup file was found, read in the text commands there
  if (found) {
    app->logfile_read(namebuf);
  }

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMDreadStartup(): load plugins", 5);

  // Load the extension packages here, _after_ reading the .vmdrc file,
  // so that the search path for extensions can be customized.
  app->commandQueue->runcommand(
    new TclEvalEvent("vmd_load_extension_packages"));   
  
  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMDreadStartup(): start Python", 1);

  // Switch to Python if requested, before reading beginCmdFile
  if (cmdFileUsesPython) {
    if (!app->textinterp_change("python")) {
      // bail out since Python scripts won't be readable by Tcl.
      msgErr << "Skipping startup script because Python could not be started." 
             << sendmsg;
      return;
    }
  }

  PROFILE_POP_RANGE();
  PROFILE_PUSH_RANGE("VMDreadStartup(): process cmd scripts", 1);

  // after reading in startup file and loading any molecule, the file
  // specified by the -e option is set up to be executed.  
  if (beginCmdFile) {
    app->logfile_read(beginCmdFile);
  } 

  PROFILE_POP_RANGE();
}

