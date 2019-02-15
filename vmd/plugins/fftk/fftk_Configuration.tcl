#
# $Id: fftk_Configuration.tcl,v 1.2 2017/03/17 19:11:10 mayne Exp $
#
#==============================================================================
# Configurations -- variables and proc for setting and querying ffTK properties
#                   and accessing multi-use variables
#==============================================================================
namespace eval ::ForceFieldToolKit::Configuration {
	
	# General Settings / Defaults
	variable namdBin
	variable tmpDir

	# Multi-use files
	variable geomOptPDB
	variable chargeOptPSF
}
#==============================================================================
proc ::ForceFieldToolKit::Configuration::init {} {
	# Sets some basic defaults for ffTK
	# Passed: nothing
	# Returns: nothing

	# General Settings
	variable namdBin "namd2"
	variable tmpDir [file normalize .]

	# Multi-use files
	variable geomOptPDB ""
	variable chargeOptPSF ""

	return
}
#==============================================================================
proc ::ForceFieldToolKit::Configuration::write {fname} {
	# Writes a file that contains the ffTK settings
	# Passed: filename for the output file
	# Returns: nothing

	# open the file for writing
	set outfile [open $fname w]

	# write a file header
	puts $outfile "# Force Field Toolkit Configuration File\n"
	flush $outfile

	# write all of the variable data to file
	foreach vName [info vars ::ForceFieldToolKit::Configuration::*] {
		upvar 0 $vName vValue
		if { $vValue ne "" } {
			puts $outfile "set $vName $vValue"
		} else {
			puts $outfile "set $vName \"\""
		}
	}

	# clean up
	close $outfile
	return
}
#==============================================================================
proc ::ForceFieldToolKit::Configuration::read {fname} {
	# "Reads" the configuration file
	# Passed: configuration filename
	# Returns: nothing
	source $fname
}
#==============================================================================
# INITIALIZE SELF UPON STARTUP
# init script has no other dependencies than namespace declarations above
::ForceFieldToolKit::Configuration::init
#==============================================================================
