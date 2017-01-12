#==============================================================================
# VMD Lite
#
# Authors:
#   Conner Herndon
#   Georgia Institute of Technology
#   http://simbac.gatech.edu/
#
#   James C. Gumbart
#   Georgia Institute of Technology
#   gumbart_physics.gatech.edu
#   http://simbac.gatech.edu/
#
# Citation:
#   If used, please reference the website at http://simbac.gatech.edu/VMDLite/
#
# Usage:
#   VMDLite is a tool for educational instruction using VMD.  Upon 
#   replacement of the vmd.rc file with the one provided, VMD Lite will 
#   becomes the default window that appears upon startup.  It offers the 
#   option to load a pre-designed lesson (downloaded separately) or the 
#   "sandbox".  In the sandbox, the student can easily explore one of a 
#   handful of common biomolecules through visualization and rudimentary 
#   analysis.
#
#
#==============================================================================



# ############################################ #
# VMD lite
#
# Conner Herndon
# Version 1.0 May 12, 2014
# Version 1.1 June 4, 2014
#   >> switch to notebook style and situate GUI
#   to be more user friendly -- ruthlessly
#   cannibalizing FFTK code (thanks Chris Mayne)
# Version 1.11 June 11, 2014
#   >> Split into multiple files and use
#   namespaces to more intelligently run.
# Version 1.12 September 10, 2014
#   >> Expanded functionality allows users to
#   select from arbitrary lessons.
# ############################################ #

package provide vmdlite 1.1

namespace eval ::VMDlite:: {
    display projection Orthographic
    display depthcue off
    color Display Background black
}
namespace eval ::VMDlite::main {
    set ::VMDlite::main::VMDLITEDIR $env(VMDLITEDIR);
}
namespace eval ::VMDlite::lesson {
}

# call for help
package require Tk 8.5

# avengers, assemble!
source [file join $env(VMDLITEDIR) vmdlite_main.tcl]
source [file join $env(VMDLITEDIR) vmdlite_guiProcs.tcl]
source [file join $env(VMDLITEDIR) vmdlite_system.tcl]
source [file join $env(VMDLITEDIR) vmdlite_graphics.tcl]
source [file join $env(VMDLITEDIR) vmdlite_analysis.tcl]
source [file join $env(VMDLITEDIR) vmdlite_sandbox.tcl]


# source [file join ./ vmdlite_main.tcl]
# source [file join ./ vmdlite_guiProcs.tcl]
# source [file join ./ vmdlite_system.tcl]
# source [file join ./ vmdlite_graphics.tcl]
# source [file join ./ vmdlite_analysis.tcl]
# source [file join ./ vmdlite_sandbox.tcl]

