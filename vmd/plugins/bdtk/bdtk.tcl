
# Tell Tcl that we are a package
package provide bdtoolkit 1.0

namespace eval ::BDTK:: {
	namespace export bdtoolkit
}

# Source procs that are common between GUI and Command Line modes
if {[catch {source [file join $env(BDTKDIR) bdtk_procs.tcl]}]} {
	puts "Fatal error! Could not load bdtk_procs.tcl"
	return 1;
}


# If there is Tk and Tile -> start GUI
if { [info exists tk_version] } {
	package require Tk 8.5
	package require tile

	# Start GUI
	if {[catch {source [file join $env(BDTKDIR) bdtk_gui.tcl]}]} {
		puts "Fatal error! Could not load bdtk_gui.tcl"
		return 1;
	}

}

# Proc that gets called by VMD
proc bdtk { } {
	return [eval ::BDTK::gui::bdtk_gui]
}


