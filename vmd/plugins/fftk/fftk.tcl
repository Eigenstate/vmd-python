#
# $Id: fftk.tcl,v 1.13 2017/12/13 18:34:16 gumbart Exp $
#
#==============================================================================
# Force Field ToolKit (ffTk) and GUI
#
# Authors:
#   Christopher G. Mayne
#   Beckman Institute for Advanced Science and Technology
#   University of Illinois, Urbana-Champaign
#   mayne@ks.uiuc.edu
#   http://www.ks.uiuc.edu/~mayne
#
#   James C. Gumbart
#   Georgia Institute of Technology
#   gumbart_physics.gatech.edu
#   http://simbac.gatech.edu/
#
# Citation:
#   C.G. Mayne, J. Saam, K. Schulten, E. Tajkhorshid, and J.C. Gumbart.
#   J. Comput. Chem. 2013, 34, 2757-2770.
#
# Usage:
#   The ffTK was designed to be used through that accompanying GUI,
#   launched from the "Extensions->Modeling" menu.  Certain procedures
#   and optimizations can be run in text-mode.  The "build run script"
#   option, where available, generates a tcl script than can be run
#   directly from the VMD console.
#
#   Also see http://www.ks.uiuc.edu/Research/vmd/plugins/fftk for the
#   accompanying documentation.
#
#==============================================================================


# package provide statement
package provide forcefieldtoolkit 1.2

# package requirements
package require exectool
package require qmtool
package require topotools
package require readcharmmpar
package require optimization
package require namdenergy
package require psfgen
package require utilities

#======================================================
namespace eval ::ForceFieldToolKit:: {
    
}
#======================================================

# source code base
source [file join $env(FFTKDIR) fftk_BuildPar.tcl]
source [file join $env(FFTKDIR) fftk_Configuration.tcl]
source [file join $env(FFTKDIR) fftk_GeomOpt.tcl]
source [file join $env(FFTKDIR) fftk_GenZMatrix.tcl]
source [file join $env(FFTKDIR) fftk_ChargeOpt.tcl]
source [file join $env(FFTKDIR) fftk_ChargeOpt_ESP.tcl]
source [file join $env(FFTKDIR) fftk_GenBonded.tcl]
source [file join $env(FFTKDIR) fftk_BondAngleOpt.tcl]
source [file join $env(FFTKDIR) fftk_GenDihScan.tcl]
source [file join $env(FFTKDIR) fftk_DihOpt.tcl]
source [file join $env(FFTKDIR) fftk_SharedFcns.tcl]
source [file join $env(FFTKDIR) fftk_distort.tcl]

# only load gui-driven code when running from VMD GUI
# (i.e., not VMD text mode)
if { [info exists tk_version] } {
    package require Tk 8.5
    package require multiplot
    source [file join $env(FFTKDIR) fftk_guiInterface.tcl]
    source [file join $env(FFTKDIR) fftk_guiProcs.tcl]
}
