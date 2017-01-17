
#read a pdb file and parse the residue numbers and create one directory for each resid


namespace eval ::alascan:: {
 	variable w
 	variable w1
	variable w2
	variable w1exists 0
	variable w2exists 0
 	variable version 1
 	variable pdbfile
	variable pdbfile_nat
 	variable psffile
	variable psffile_nat 
	variable resname_nat
	variable resid_nat
	variable segname_nat
 	variable resname 
 	variable resid 
 	variable segname 
	variable parseResname
	variable parseResid
	variable parseSegname
	variable parseResname_uni
	variable parseResid_uni
	variable parseSegname_uni
 	variable selavl 
 	variable selnonala 
 	variable seldformutation 
 	variable notseldformut 
 	variable totalres 
 	variable feprunpath
 	variable dirpath 
 	variable root
 	variable nonala
 	variable nonalasegname 
 	variable seldformutsegname
 	variable seldformutresname
 	variable listparm
 	variable hybtopfile
 	variable xscfile
 	variable selectrange
 	variable alaresid
 	variable confminpath
 	variable conffeppath
	variable fepdir
	variable unidir
	variable unidirDGlst
	variable unidirDGerror 
 	variable wextend
 	variable sstruct
	variable sstruct_nat
	variable running 0
	variable done 0
 	variable minsteps		10000
 	variable eqsteps		100000
 	variable fepeqsteps		50000
 	variable fepwindows		20
 	variable totalfepsteps		500000
 	variable temperature 		300
	variable totalFenergy
	variable totalError
	variable tempParse		300
	variable k 0.001987200
	variable kt  [expr $k * $tempParse ]
	variable hyst
}

package provide alascan $::alascan::version
package require psfgen
package require readcharmmtop 
package require exectool 1.2
package require autoionize 1.3
package require parsefep 1.9


#------------------------------------------------------------
#process the input file to extract the residue names and ids
#-----------------------------------------------------------
proc ::alascan::inputproc {} {
 	variable pdbfile
 	variable psffile
 	variable resname
 	variable resid
 	variable segname
 	variable w
 	variable feprunpath
 	variable xscfile
	variable sstruct {}
 	variable nonala {}
 	variable nonalasegname {}
 	variable nonalaresname {}
 	global env

	 if { $psffile == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "No PSF file loaded"
		return 0
		}

 	if { $pdbfile == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "No PDB file loaded!"
		return 0
		}

 	if { $xscfile == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "No XSC file loaded!"
		return 0
		}

	#load the psf and pdb file into vmd and select the resname and resid and segment names
	mol delete top
	mol load psf $psffile pdb $pdbfile
	set vmdpro [atomselect top "protein"]
	#check whether there is a protein loaded
	if { [$vmdpro num] == 0 } {
		puts "Protein not found"
		}
	set vmdproca [atomselect top "name CA"]
	set ::alascan::sstruct [$vmdproca get structure]		
	set vmdnonpro [atomselect top "not protein"]
	set resid [$vmdproca get {resid}]
	set resname [$vmdproca get {resname}]
	set segname [$vmdproca get {segname}]
	set ::alascan::root [file rootname [file tail $pdbfile]]
	set feprunpath [file dirname $pdbfile]
	mol delete top

	display_resnames
	return 0
}

#--------------------------
#edit simulation parameters
#--------------------------
proc ::alascan::editparameters {} {
	
	variable w2
 	variable minsteps
 	variable eqsteps
 	variable fepeqsteps
 	variable fepwindows
 	variable fepstepsperwindow
	variable temperature
	variable w2exists
		
	if { [winfo exists .editsimparameters] } {
		wm deiconify .editsimparameters
		return 
		}

	set w2 [toplevel ".editsimparameters"]
	wm title $w2 "Simulation parameters"
	wm resizable $w2 1 1

	#equilibration parameters
	labelframe $w2.eqparm -bd 2 -relief ridge -text "Minimization and Equilibration Parameters"

	grid [label $w2.eqparm.temp -text "Temperature:"] -row 1 -column 0 -sticky e
	grid [entry $w2.eqparm.tempentry -textvariable ::alascan::temperature] -row 1 -column 1 -sticky ew
	
	grid [label  $w2.eqparm.minst -text "Number of minimization steps:" ] -row 2 -column 0 -sticky e
	grid [entry $w2.eqparm.minentry -textvariable ::alascan::minsteps] -row 2 -column 1 -sticky ew

	grid [label $w2.eqparm.eqstep -text "Number of equilibration steps:" ] -row 3 -column 0 -sticky e
	grid [entry $w2.eqparm.eqstepentry -textvariable ::alascan::fepeqsteps] -row 3 -column 1 -sticky ew

	grid columnconfigure $w2.eqparm 1 -weight 1	
	pack $w2.eqparm -side top -padx 2 -pady 6 -expand 1 -fill x

	#FEP parameters
	labelframe $w2.fep -bd 2 -relief ridge -text "FEP Transformation Parameters"

	grid [label $w2.fep.windows -text "Number of FEP Windows:" ] -row 1 -column 0 -sticky e
	grid [entry $w2.fep.windowsentry -textvariable ::alascan::fepwindows] -row 1 -column 1 -sticky ew

	grid [label $w2.fep.stepsperwindow -text "Total FEP steps:" ] -row 2 -column 0 -sticky e
	grid [entry $w2.fep.stepsperwindowentry -textvariable ::alascan::totalfepsteps] -row 2 -column 1 -sticky ew

	grid [label $w2.fep.fepeqsteps -text "Equilibration FEP steps:"] -row 3 -column 0 -sticky e
	grid [entry $w2.fep.fepeqstepsentry -textvariable ::alascan::eqsteps] -row 3 -column 1 -sticky ew
		  	
	grid columnconfigure $w2.fep 1 -weight 1
	pack $w2.fep -side top -padx 2 -pady 6 -expand 1 -fill x

	set w2exists 1
}

#-------------------------
#reset parameters
#-------------------------
proc ::alascan::resetparameters {} {
 variable temperature 		300
 variable minsteps		100
 variable eqsteps		100
 variable fepeqsteps		50000
 variable fepwindows		20
 variable totalfepsteps		500
}

#------------------------------------------------
#main window gui
#------------------------------------------------
proc ::alascan::alascan_gui {} {
	variable w
	variable w1
	variable w2
	variable resname
	variable resid
 	variable segname

 	if { [winfo exists .alascangui] } {
		wm deiconify .alascangui
		return
		}

	set w [toplevel ".alascangui"]
	wm title $w "Alanine scanning - Calculation Setup - Analysis"
	wm resizable $w 0 1

	#Add a menubar
	frame $w.menubar -relief raised -bd 2
	pack $w.menubar -expand 1 -fill x
	menubutton $w.menubar.help -text "Help" -underline 0 -width 4 -menu $w.menubar.help.menu 
	$w.menubar.help config -width 5
	pack $w.menubar.help -side right


	menu $w.menubar.help.menu -tearoff 0
	$w.menubar.help.menu add command -label "About" \
		-command {tk_messageBox -type ok -title "About Alanine Scan" \
		-message "Setup input files to perform systematic Alanine scanning calculation using FEP method\
			 \n\t-Choose residues for Alanine scan\n\t-Upload force field parameters file\
			 \n\t-Select hybrid topology file\n\t-Choose a path to write setup files\n\t-Edit simulation parameters\
			 \n\t-Write NAMD config files\n\nAnalyze the fepout files produced from the systematic Alanine scanning calculations\
			 \n\t-Systematic analysis of fepout files for each\n\t mutated residues\
			 \n\t-Analysis output:\n\t\t->\u0394G for each transformation\
			 \n\t\t->Hysteresis of the transformation"}
	$w.menubar.help.menu add command -label "Help"  
	
	frame $w.intro  
	label $w.intro.label -text "Alanine scanning based on FEP method" -relief ridge -padx 50m -pady 1m
	pack $w.intro -fill x -padx 2 -pady 6 -expand 1
	pack $w.intro.label -fill x
	frame $w.host_guest
	checkbutton $w.host_guest.sel -text "  Host-Guest System" -variable ::alascan::hg_sel -anchor w
	pack $w.host_guest -fill x -expand 1 
	pack $w.host_guest.sel -fill x
	
	frame $w.setup_analysis
	grid [button $w.setup_analysis.setup -text "Setup FEP input files" -command ::alascan::inputSetup ] -row 2 -column 0 -sticky news 
	grid [button $w.setup_analysis.analysisFEP -text "Analyze FEP output" -command ::alascan::analyzeFEP ] -row 2 -column 1 -sticky news
	grid columnconfigure $w.setup_analysis 0 -weight 1
	grid columnconfigure $w.setup_analysis 1 -weight 1
	$w.setup_analysis.setup configure -foreground red
	pack $w.setup_analysis -fill x -padx 2 -pady 5 -expand 1 
	
} ;# "alascan::alascan_gui"


proc ::alascan::inputSetup {} {

	variable w
 	variable wextend 0

	destroy $w.input $w.intro1 $w.selected $w.selected.fr $w.lstparm $w.lstparm.fr $w.tlist $w.feppath $w.parameters $w.fep
	pack forget $w.intro2 $w.input_ana $w.status $w.l $w.frame $w.tit

	#take the psf and pdb file from vmd if there is one
	if {[molinfo num] != 0} {
    	foreach fname [lindex [molinfo top get filename] 0] ftype [lindex [molinfo top get filetype] 0] {
      	if { [string equal $ftype "pdb"] } {
        	set ::alascan::pdbfile $fname
      		} elseif { [string equal $ftype "psf"] } {
			set ::alascan::psffile $fname
			}
    		}
 	}

	frame $w.intro1  
	label $w.intro1.label -text "Setup FEP input files" -relief ridge -padx 50m
	pack $w.intro1 -fill x -padx 2 -pady 6 -expand 1
	pack $w.intro1.label -fill x

	labelframe $w.input -bd 2 -relief ridge -text "Input" 
	grid [label $w.input.psffile -text "PSF file:"] -row 1 -column 0 -sticky w
	grid [entry $w.input.psfpath -width 40 -textvariable ::alascan::psffile] -row 1 -column 1 -sticky ew
	grid [button $w.input.psfbutton -text "Browse" -command {
		set tempfile [tk_getOpenFile]
		if { ![string equal $tempfile ""] } {set ::alascan::psffile $tempfile} }] -row 1 -column 2 -sticky w

	grid [label $w.input.pdbfile -text "PDB file:"] -row 2 -column 0 -sticky w
	grid [entry $w.input.pdbpath -width 40 -textvariable ::alascan::pdbfile] -row 2 -column 1 -sticky ew
	grid [button $w.input.pdbbutton -text "Browse" -command {
		set tempfile [tk_getOpenFile]
		if {![string equal $tempfile ""]} { set ::alascan::pdbfile $tempfile} }] -row 2 -column 2 -sticky w

	grid [label $w.input.xscfile -text "XSC file:"] -row 3 -column 0 -sticky w
	grid [entry $w.input.xscpath -width 40 -textvariable ::alascan::xscfile] -row 3 -column 1 -sticky ew
	grid [button $w.input.xscbutton -text "Browse" -command {
		set tempfile [tk_getOpenFile]
		if {![string equal $tempfile ""]} { set ::alascan::xscfile $tempfile} }] -row 3 -column 2 -sticky w
	grid [button $w.input.next -text "Load input files" -command ::alascan::inputproc] 
	grid columnconfigure $w.input 1 -weight 1
	pack $w.input -side top -padx 4 -pady 5 -expand 1 -fill x

} ;#"alascan::inputSetup"



#get back the selected residues list here
	proc ::alascan::seledres {} {
 	variable w
	variable seldformutation
	variable seldformutsegname
	variable selfformutresname
	variable notseldformut
	variable totalres
	variable feprunpath
	variable listparm
	variable hybtopfile
	variable dirpath
	variable wextend 1
	variable host_wextend 0
	global env
		
	set totseldres [llength $seldformutation]
	set totnonalares [llength $::alascan::nonala]

	labelframe $w.selected -bd 2 -relief ridge -text "Summary of residue selection"
	frame $w.selected.fr
	if {$::alascan::hg_sel == 1} {
		set host_wextend 1
		grid [label $w.selected.fr.hs -text "Mutations will be applied on the host system, segname [lsort -unique $seldformutsegname]"]
		} elseif {$::alascan::hg_sel ==0 } {
			grid [label $w.selected.fr.hs -text ""]	
			}
	grid [label $w.selected.fr.l1 -text "Total number of amino acids: $totalres\tTotal number of non-Ala residues: $totnonalares"]
	grid [label $w.selected.fr.l3 -text "Number of amino acids selected for Alanine scanning: $totseldres"] 
	pack $w.selected.fr $w.selected -fill x -padx 4 -pady 5 -expand 1 
	
	#list selected parameter files
	set listparm {}
	labelframe $w.lstparm -bd 2 -relief ridge -text "Upload force field parameter files:" 
	frame $w.lstparm.fr
	scrollbar $w.lstparm.fr.scrolly -orient vertical -command "$w.lstparm.fr.lbox yview"
	scrollbar $w.lstparm.fr.scrollx -orient horizontal -command "$w.lstparm.fr.lbox xview"
	listbox $w.lstparm.fr.lbox -activestyle dotbox -xscroll "$w.lstparm.fr.scrollx set" -yscroll "$w.lstparm.fr.scrolly set" \
		-width 55 -height 2 -setgrid 1 -selectmod multiple -listvariable ::alascan::listparm
	frame $w.lstparm.buttons
	button $w.lstparm.buttons.add -text "Add" -command [namespace code {
		set tempfile [tk_getOpenFile]
		if {$tempfile != ""} {lappend ::alascan::listparm $tempfile}	

		}]
	button $w.lstparm.buttons.delete -text "Delete" -command [namespace code {
		foreach i [$w.lstparm.fr.lbox curselection] {
			$w.lstparm.fr.lbox delete $i
			}
		}]
	pack $w.lstparm.buttons.add $w.lstparm.buttons.delete -expand 1 -fill x -side top
	pack $w.lstparm.buttons -side right
	pack $w.lstparm.fr.scrollx $w.lstparm.fr.lbox -side bottom -fill x -expand 1
	pack $w.lstparm.fr.scrolly $w.lstparm.fr.lbox -side left -fill y -expand 1
	pack $w.lstparm.fr 
	pack $w.lstparm -fill x -padx 4 -pady 5 -expand 1

	#list topology files from readcharmmtop1.1 for selection
	set ::alascan::hybtopfile [glob $env(CHARMMTOPDIR)/*hybrid*.inp]
	labelframe $w.tlist -bd 2 -relief ridge -text "Select a hybrid topology file:"
	frame $w.tlist.list
	scrollbar $w.tlist.list.scroll -orient vertical -command "$w.tlist.list.list yview" 
	scrollbar $w.tlist.list.scrollx -orient horizontal -command "$w.tlist.list.list xview"
	listbox $w.tlist.list.list -activestyle dotbox -xscroll "$w.tlist.list.scrollx set" -yscroll "$w.tlist.list.scroll set"  \
		-width 55 -height 2 -setgrid 1 -selectmod browse -listvariable ::alascan::hybtopfile
	frame $w.tlist.buttons
	button $w.tlist.buttons.add -text "Add" -command [namespace code {
		set toptypes {
			{{CHARMM Topology Files} {.top .inp .rtf}}
			{{All Files} {*}}
			}
		set temfile [tk_getOpenFile -filetypes $toptypes]
		if {$temfile !=""} {lappend ::alascan::hybtopfile $temfile}
	}]
	button $w.tlist.buttons.delete -text "Delete" -command [namespace code {
		foreach i [$w.tlist.list.list curselection] {
			$w.tlist.list.list delete $i
		}
	}]
	pack $w.tlist.buttons.add $w.tlist.buttons.delete -expand 1 -fill x -side top
	pack $w.tlist.buttons -side right
	pack $w.tlist.list.scrollx $w.tlist.list.list -side bottom -fill x -expand 1 
	pack $w.tlist.list.scroll $w.tlist.list.list -side left -fill y -expand 1 
	grid columnconfigure $w.tlist.list 0 -weight 1
	pack $w.tlist.list
	pack $w.tlist -fill x -padx 4 -pady 5 -expand 1	
	
	frame $w.feppath
	grid [label $w.feppath.l1 -text "FEP run path: " -width 20] -row 1 -column 0 -sticky w
	grid [entry $w.feppath.p -width 25 -textvariable ::alascan::feprunpath] \
		-row 1 -column 1 -sticky ew
	grid [button $w.feppath.bt -text "Browse" \
		-command {
			set tempfile [tk_chooseDirectory]
			if {![string equal $tempfile ""]} { set ::alascan::feprunpath $tempfile }
			}] -row 1 -column 2 -sticky w
	grid columnconfigure $w.feppath 1 -weight 1
	pack $w.feppath -side top -padx 2 -pady 6 -expand 1 -fill x
	
	frame $w.parameters
	grid [label $w.parameters.l1 -text "Simulation Parameters:"] -row 0 -column 0
	grid [button $w.parameters.edit -text "Edit" -command ::alascan::editparameters] -row 0 -column 1 
	grid [button $w.parameters.reset -text "Reset" -command ::alascan::resetparameters] -row 0 -column 2 
	pack $w.parameters -padx 8 -expand 1 -fill x

	frame $w.fep
	grid [button $w.fep.write -text "Write NAMD config files" -command ::alascan::mutator] 
	pack $w.fep.write -fill x -padx 2 -pady 10 -expand 1
	pack $w.fep -fill x -expand 1

	} ;# proc seledres

global env
source [file join $env(ALASCANFEPDIR) alascan_analysis.tcl]

#-----------
#reset variable in summary of residue selection
#------------
proc ::alascan::seledres_two {} {
	variable seldformutation
	variable seldformutsegname
	variable host_wextend
	variable pdbfile
	variable w
	set feprunpath [file dirname $pdbfile]	
	set totseldres [llength $seldformutation]
	.alascangui.selected.fr.l3 configure -text "Number of amino acids selected for Alanine scanning: $totseldres"
	if {$::alascan::hg_sel == 1} {
		.alascangui.selected.fr.hs configure -text "Mutations will be applied on the host system, segname [lsort -unique $seldformutsegname]"	
		} elseif {$::alascan::hg_sel == 0} {.alascangui.selected.fr.hs configure -text ""} 
}

#---------------------
#	mutator step 1	
#---------------------
proc ::alascan::mutator {} {
	variable w
	variable seldformutation
	variable seldformutsegname
	variable psffile
	variable pdbfile
	variable segname
	variable listparm
	variable feptcl
	variable hybtopfile
	variable dirpath
	variable owrite
	variable topstatus 0
	
	global env
	resetpsf
	psfcontext reset
	#check force field parameter file
	if {$listparm == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "Upload force field parameter file"
		return 0
		}

	#check topology file
	set feptopfilesel [$w.tlist.list.list curselection]
	set feptopselected [lindex $hybtopfile $feptopfilesel]
	if { $feptopfilesel == {} } {
		tk_messageBox -icon error -type ok -title Message -parent $w \
		-message "Select a hybrid topology file"
		return 0
		}

	array set aa [list ALA A CYS C ASP D GLU E PHE F GLY G HSD H HSE X HSP Z ILE I LYS K LEU L MET M \
	    		   ASN N PRO P GLN Q ARG R SER S THR T VAL V TRP W TYR Y]
	
	#create a dir to write setup files		
	set dirpath "$::alascan::feprunpath/$::alascan::root-FEP"
	if {[file exists $dirpath] == 1} { 
		puts "Alanine scan: Directory name \"$dirpath\" already exists, overwriting"
		file delete -force $dirpath
		file mkdir $dirpath	
		} else {file mkdir $dirpath}

	if {$::alascan::hg_sel == 1} {set hg 1}
	if {$::alascan::hg_sel == 0} {set hg 0}	

	#construct the hybdrid structure and topolog files by passing variables to the mutator plugin
	foreach eachres $seldformutation proseg $seldformutsegname {
	set patchlist {}; set patchlist_host {};

	mol load psf $psffile pdb $pdbfile
	set prot [atomselect top "protein"]
	set notprot [atomselect top "not protein"]
	set getresname [atomselect top "resid $eachres and name CA"]
	set resnamedir [$getresname get resname]
	#create directory for the current residue
	set tempala 2ALA
	file mkdir $dirpath/$eachres-$resnamedir$tempala
	#host segname
	if {$hg == 1} {set host_segname [lsort -unique $seldformutsegname]}
	#load top files
	#set topfile [file join $env(CHARMMTOPDIR) top_all27_prot_lipid_na.inp]
	#set topfile [file join $env(CHARMMTOPDIR) top_all36_hybrid.inp]
	set topfile $feptopselected
	puts "Alanine scan: Reading $psffile  $pdbfile"
		
 	#write pdbs for each protein segment
	foreach i [lsort -unique [$prot get segname]] {
		set c [atomselect top "segname $i"]
		puts "Writing temporary pdb file for protein segment $i"
		$c writepdb $dirpath/$eachres-$resnamedir$tempala/mtemp-$i.pdb
		$c delete 
	}

	#make list of pathces, topology files here
	set npatches 0; set npatches_host 0
	set ntopos 0
	set infile [open $psffile r]
	   while { [gets $infile line] >= 0 } {
	if {[string first "NATOM" $line] >= 0 } {break}
        #topologies
        if {[string first "REMARKS topology" $line] >= 0} {
           set topoind [string first "topology" $line]
           lappend topofilelist [string trim [string range $line [string wordend $line $topoind] end]]
           incr ntopos
        }
	#patches - host_guest
	    if { ([string first "REMARKS patch" $line] >= 0) || ([string first "REMARKS defaultpatch" $line] >= 0) } {
           # new patch, does it involve protein?
           foreach i [lsort -unique [$prot get segname]] {
              if {[string first " $i:" $line] > 0} {
                 lappend patchlist [string range $line [string first "patch" $line] end]
                 incr npatches

		#patches - host
		if {$hg == 1} {
		if {$i == $host_segname && [string first " $host_segname:" $line] > 0} {
			lappend patchlist_host [string range $line [string first "patch" $line] end]
			incr npatches_host	
			}
                 break
		}
              }
            }
          }
	}
	close $infile
	if {$npatches == 0 } {
		puts "Alanine scan: no patches involving protein were found in the PSF file!"
		}

	# Write psf/pdb for non-protein stuff (water, membrane, ions, etc.)
	set readtop 0
    	if { $ntopos > 0 } {
       		foreach topo $topofilelist {
          	if {[file exists $topo] > 0} {
             		topology $topo 
			set readtop 1 
          		}
       		   }
		if {$readtop == 0} {
			puts "Alanine scan: Reading default topolog..."
			if ($topstatus==0) {topology $topfile; set topstatus 1}
			}
    		} else {
       			puts "Alanine scan: Reading default topology..."
       			if ($topstatus==0) {topology $topfile; set topstatus 1}
    			}
	#write the non-prot segments in a separate temp files
    	if { [$notprot num] != 0} {
		resetpsf
		readpsf $psffile
		coordpdb $pdbfile
		foreach segres [lsort -unique [$prot get {segname resid}]] {
		    delatom [lindex $segres 0] [lindex $segres 1]
		}
		guesscoord
	
		puts "\n Alanine scan: Writing temporary files for non-protein components"
		writepsf $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.psf
		writepdb $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.pdb
    		}		
	
	
	resetpsf
	#topology $feptopselected

	foreach i [lsort -unique [$prot get segname]] {
	    segment $i {
		pdb $dirpath/$eachres-$resnamedir$tempala/mtemp-$i.pdb
               	if {$npatches > 0} { first none }
		if {$i == $proseg} {
			set alpha [atomselect top "segname $i and resid $eachres and alpha"]
			set oldres [$alpha get resname]
  			set hyb [format "%s2%s" $aa($oldres) $aa(ALA)]
			puts "Alanine scan: Residue $oldres $eachres of segment $proseg is mutated to $hyb"
			mutate $eachres $hyb
			$alpha delete
			}
                if {$npatches > 0} { last none }
	    }
	}


	#Reading non protein stuff
	if { [$notprot num] != 0} {
	    readpsf $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.psf
	}
					
        # add patches
        if {$npatches > 0} {
            puts "Alanine scan: Adding patches"
            foreach patch $patchlist {
               eval $patch
            }
        }

        # only read coordinates AFTER adding patches
        foreach i [lsort -unique [$prot get segname]] {
		puts "segname: $i\npdb file: $dirpath/$eachres-$resnamedir$tempala/mtemp-$i.pdb"
           coordpdb $dirpath/$eachres-$resnamedir$tempala/mtemp-$i.pdb $i
        }

	# Reading non protein stuff
	if { [$notprot num] != 0} {
	    coordpdb $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.pdb
	}

	# Saving all fep
        regenerate angles dihedrals
	guesscoord
	if {$hg == 0} {writepdb $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.pdb}
	if {$hg == 1} {
		file mkdir $dirpath/$eachres-$resnamedir$tempala/Host-Guest
		writepdb $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.pdb
		}
	if {$hg == 0} {writepsf $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.fep.psf}
	if {$hg == 1} {writepsf $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.tmp.fep.psf}
	
	#FEPfile 	
	#Loading pdb, 
	if {$hg==0} {mol load psf $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.fep.psf pdb $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.pdb}
	if {$hg==1} {mol new $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.pdb type pdb}
	set prot [atomselect top "protein"]
	set modified {}
	set initial {}
	set final {}
	foreach i [lsort -unique [$prot get segname]] {
	    if { [info exists $proseg] } {
		if { $i == $proseg } {
		    set modified [atomselect top "segname $proseg and resid $eachres and not name N HN CA HA C O"]
		} else { 
                    set modified [atomselect top "none"]
                }
	    } else {
		set modified [atomselect top "segname $i and resid $eachres and not name N HN CA HA C O"]
	    }
	
	    foreach j [$modified list] {
	       set temp [atomselect top "index $j"]
	       set letter [string index [$temp get name] end]
	    
	       if { $letter == "A" } {
		  lappend initial $j
	       } elseif { $letter == "B" } {
	 	  lappend final $j
	       } else {
	   	  puts "Alanine scan: WARNING - unexpected atom name [$temp get name]: belongs to a patch?"
	       }
               $temp delete
	    }
        }

	set init [atomselect top "index $initial"]
	set fin  [atomselect top "index $final"]
	
	$init set beta -1.0
	$fin  set beta 1.0
	
	if {$hg==0} {animate write pdb $dirpath/$eachres-$resnamedir$tempala/$eachres.fep waitfor all}
	if {$hg==1} {animate write pdb $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep waitfor all}


	#run alchemify
 	if {$hg==0} {set exit_code [alchemify $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.fep.psf $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.psf $dirpath/$eachres-$resnamedir$tempala/$eachres.fep] }
	if {$hg==1} {set exit_code [alchemify $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.tmp.fep.psf $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.psf $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep] }
	
	if { $exit_code } {
	    puts "Alanine scan: ERROR --- Alchemify returned code $exit_code"
	} else {
	    puts "Alanine scan: Alchemify completed successfully"
	}
	if {$hg==0} {file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.fep.psf}
	if {$hg==1} {file delete -force $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.tmp.fep.psf}
	if {$hg==0} {file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.pdb}
	if {$hg==1} {file delete -force $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.pdb}
    	
   	# Deleting temporary items
	foreach i [lsort -unique [$prot get segname]] {file delete -force $dirpath/$eachres-$resnamedir$tempala/mtemp-$i.pdb}
    		
    	if { [$notprot num] != 0} {
		file delete -force $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.psf
		file delete -force $dirpath/$eachres-$resnamedir$tempala/mtemp-nprot.pdb
    		}
    
    	mol delete top
    	mol delete top
	$notprot delete
	
    	# Loading mutated/hybrid system, 
	if {$hg==0} {mol load psf $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.psf pdb $dirpath/$eachres-$resnamedir$tempala/$eachres.fep}
	if {$hg==1} {mol load psf $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.psf pdb $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep}
	mol delrep 0 top
	mol representation Lines
	mol color Beta
	mol selection {all}
	mol material Opaque
	mol addrep top
	
     	# re-compute net charge of the system
    	set sel [atomselect top all]
   	set netCharge [eval "vecadd [$sel get charge]"]
    	$sel delete
    	puts "Alanine scan: System net charge after mutation: ${netCharge}e"
	puts "Alanine scan: Hybrid structure and topology files are generated for residue id $eachres"

	#write equilibration file
	if {$hg==0} {set curdirpath $dirpath/$eachres-$resnamedir$tempala/}
	if {$hg==1} {set curdirpath $dirpath/$eachres-$resnamedir$tempala/Host-Guest/}
	if {$hg==0} {set feppsf $dirpath/$eachres-$resnamedir$tempala/$eachres.fep.psf}
	if {$hg==1} {set feppsf $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep.psf}
	if {$hg==0} {set feppdb $dirpath/$eachres-$resnamedir$tempala/$eachres.fep}
	if {$hg==1} {set feppdb $dirpath/$eachres-$resnamedir$tempala/Host-Guest/$eachres.fep}
	set host_write 0
	::alascan::write_fep_eq $curdirpath $feppsf $feppdb $host_write
	::alascan::write_fep_forward $curdirpath $feppdb $feppsf $host_write
	::alascan::write_fep_backward $curdirpath $feppdb $feppsf $host_write

	mol delete top
	$prot delete

	#for the host system alone
	if {$::alascan::hg_sel == 1} {
		resetpsf
		file mkdir $dirpath/$eachres-$resnamedir$tempala/Host
		#get the host pdb psf from host-guest
		mol load psf $psffile pdb $pdbfile
		set prot [atomselect top "protein"]
		readpsf $psffile
		coordpdb $pdbfile
		foreach segres [lsort -unique [[atomselect top "all"] get {segname resid}]] {
			if {[lindex $segres 0] != $host_segname} {delatom [lindex $segres 0] [lindex $segres 1]}
		}
		guesscoord
		puts "\n Alanine scan: Writing temporary files for host-system"
		writepsf $dirpath/$eachres-$resnamedir$tempala/mtemp-host.psf
		writepdb $dirpath/$eachres-$resnamedir$tempala/mtemp-host.pdb
		mol delete top
		resetpsf

		mol load psf $dirpath/$eachres-$resnamedir$tempala/mtemp-host.psf pdb $dirpath/$eachres-$resnamedir$tempala/mtemp-host.pdb		
	    	segment $host_segname {
			pdb $dirpath/$eachres-$resnamedir$tempala/mtemp-host.pdb
               		if {$npatches_host > 0} { first none }

			set alpha [atomselect top "segname $host_segname and resid $eachres and alpha"]
			set oldres [$alpha get resname]
  			set hyb [format "%s2%s" $aa($oldres) $aa(ALA)]
			mutate $eachres $hyb
			$alpha delete


                	if {$npatches_host > 0} { last none }
	    	      }			

		#add patches
        	if {$npatches_host > 0} {
            		puts "Alanine scan: Adding patches for host system"
            		foreach patch $patchlist_host {eval $patch}
        		}
        	#read coordinates 
           	coordpdb $dirpath/$eachres-$resnamedir$tempala/mtemp-host.pdb $host_segname
        	regenerate angles dihedrals
		guesscoord	
		writepdb $dirpath/$eachres-$resnamedir$tempala/$eachres.pdb
		writepsf $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.psf
		mol delete top
		file delete -force $dirpath/$eachres-$resnamedir$tempala/mtemp-host.psf
		file delete -force $dirpath/$eachres-$resnamedir$tempala/mtemp-host.pdb
		
		#Host solvation
		solvate $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.psf $dirpath/$eachres-$resnamedir$tempala/$eachres.pdb -t 12 -o $dirpath/$eachres-$resnamedir$tempala/$eachres.sol	
		file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.sol.log
		
		#add ions
		cd $dirpath/$eachres-$resnamedir$tempala/
		autoionize -psf $eachres.sol.psf -pdb $eachres.sol.pdb -neutralize
		file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.sol.pdb
		file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.sol.psf
		mol delete top; mol delete top
		resetpsf; $prot delete
		
		#Load fep files
		mol new $dirpath/$eachres-$resnamedir$tempala/ionized.pdb 
		set prot [atomselect top "protein"]
		set modified {}; set initial {}; set final {}
	
		#appearing and disappearing atoms	
		foreach i [lsort -unique [$prot get segname]] {
	    		if { [info exists $host_segname] } {
				if { $i == $host_segname } {
		    			set modified [atomselect top "segname $host_segname and resid $eachres and not name N HN CA HA C O"]
					} else { 
                    				set modified [atomselect top "none"]
                				}
	    		} else {
				set modified [atomselect top "segname $i and resid $eachres and not name N HN CA HA C O"]
	    			}
	
	    		foreach j [$modified list] {
	       			set temp [atomselect top "index $j"]
	       			set letter [string index [$temp get name] end]
	       			if { $letter == "A" } {
		  				lappend initial $j
	       					} elseif { $letter == "B" } {
	 	  					lappend final $j
	       						} else {
	   	  					puts "Alanine scan: WARNING - unexpected atom name [$temp get name]: belongs to a patch?"
	       						}
               			$temp delete
	    			}
        		}
		set init [atomselect top "index $initial"]
		set fin  [atomselect top "index $final"]
		$init set beta -1.0
		$fin  set beta 1.0
		animate write pdb $dirpath/$eachres-$resnamedir$tempala/Host/$eachres.fep waitfor all

		#Alchemify
		set exit_code [alchemify $dirpath/$eachres-$resnamedir$tempala/ionized.psf $dirpath/$eachres-$resnamedir$tempala/Host/$eachres.fep.psf $dirpath/$eachres-$resnamedir$tempala/Host/$eachres.fep]
		if { $exit_code } {
	    		puts "Alanine scan: ERROR --- Alchemify returned code $exit_code"
			} else {puts "Alanine scan: Alchemify completed successfully"}

		#write equilibration files
		set curdirpath $dirpath/$eachres-$resnamedir$tempala/Host
		set feppsf $dirpath/$eachres-$resnamedir$tempala/Host/$eachres.fep.psf
		set feppdb $dirpath/$eachres-$resnamedir$tempala/Host/$eachres.fep
		set host_write 1
		::alascan::write_fep_eq $curdirpath $feppsf $feppdb $host_write
		::alascan::write_fep_forward $curdirpath $feppdb $feppsf $host_write
		::alascan::write_fep_backward $curdirpath $feppdb $feppsf $host_write

		file delete -force $dirpath/$eachres-$resnamedir$tempala/ionized.pdb
		file delete -force $dirpath/$eachres-$resnamedir$tempala/ionized.psf
		file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.tmp.psf
		file delete -force $dirpath/$eachres-$resnamedir$tempala/$eachres.pdb
	
	mol delete top
	} ;#host system alone
	
	}
	mol load psf $psffile pdb $pdbfile
}; #::alascan::mutator 

#-------------------------------------
#write inputfile for fep equilibration
#-------------------------------------
proc ::alascan::write_fep_eq {path feppsf feppdb host_write} {
	variable w
	variable listparm
	variable minsteps
	variable fepeqsteps
	variable temperature
	variable xscfile 
	global env
	
	set feptclsource [file join $env(ALASCANFEPDIR) fep.tcl]
	if {[file exists $path/fep.tcl] == 0} {file copy -force $feptclsource $path}
	if {$host_write== 0} {if {[file exists $path/[file tail $xscfile]] == 0} {file copy -force $xscfile $path}}
	if {$host_write == 1} {
		mol load psf $feppsf pdb $feppdb
		set all [atomselect top all]
		set minmax [measure minmax $all]
		set vec [vecsub [lindex $minmax 1] [lindex $minmax 0]]
		set center [measure center $all]
		mol delete top	
		}
	
	set writeeq [open $path/fep_equilibration.namd w]
	puts $writeeq "#NAMD Config file - autogenerated by Alanine scanning plugin"
	puts $writeeq "#EQUILIBRATION\n"	
	puts $writeeq "#INPUT\n"
	puts $writeeq "set temp	\t\t$temperature"
	foreach parm $listparm {
		if {[file exists $path/[file tail $parm]] == 0} {file copy -force $parm $path}
		puts $writeeq "parameters	\t\t[file tail $parm]"
		}
	puts $writeeq "paraTypeCharmm	\t\ton\nexclude		\t\tscaled1-4\n1-4scaling	\t\t1.0\n"
	puts $writeeq "\n#TOPOLOGY\nstructure	\t\t[file tail $feppsf]\n\n#INITIAL CONDITIONS\n"
	puts $writeeq "coordinates	\t\t[file tail $feppdb]\ntemperature	\t\t$temperature"
	if {$host_write==0} {puts $writeeq "extendedsystem	\t\t[file tail $xscfile]"}
	if {$host_write==1} {
		puts $writeeq "\n#Periodic Boundary Conditions\ncellBasisVector1	\t[format %.2f [lindex $vec 0]] 0 0"
		puts $writeeq "cellBasisVector2	\t0 [format %.2f [lindex $vec 1]] 0\ncellBasisVector3	\t0 0 [format %.2f [lindex $vec 2]]"
		puts $writeeq "cellOrigin	\t\t[format %.2f [lindex $center 0]] [format %.2f [lindex $center 1]] [format %.2f [lindex $center 2]]"
		}
	puts $writeeq "\n\n#OUTPUT FREQUENCIES\n"
	puts $writeeq "outputenergies	\t\t100\noutputtiming	\t\t100\noutputpressure	\t\t100"
	puts $writeeq "restartfreq	\t\t100\nXSTFreq		\t\t100\ndcdfreq	\t\t\t2500\n\n\n#OUTPUT AND RESTART\n"
	puts $writeeq "outputname	\t\tfep_eq\nrestartname	\t\tfep_eq\nbinaryoutput	\t\tyes"
	puts $writeeq "binaryrestart	\t\tyes\n\n#CONSTANT-T\nlangevin	\t\ton"
	puts $writeeq "langevinTemp	\t\t$temperature\nlangevinDamping	\t\t1.0\n\n#PME\n"
	puts $writeeq "PME		\t\tyes\nPMETolerance	\t\t10e-6\nPMEInterpOrder	\t\t4"
	puts $writeeq "PMEGridSpacing	\t\t1.0\n\n"
	puts $writeeq "#WRAP WATER FOR OUTPUT\nwrapAll	\t\t\ton\n\n#CONSTANT-P\n"
	puts $writeeq "LangevinPiston	\t\ton\nLangevinPistonTarget	\t1\nLangevinPistonPeriod	\t100"
	puts $writeeq "LangevinPistonDecay	\t100\nLangevinPistonTemp	\t$temperature\nStrainRate	\t\t0.0 0.0 0.0"
	puts $writeeq "useGrouppressure	\tyes\nuseFlexibleCell	\t\tno\n\n#SPACE PARTITIONING\n"
	puts $writeeq "stepspercycle	\t\t20\nmargin	\t\t\t1.0\n\n#CUT-OFFS\nswitching	\t\ton\nswitchdist	\t\t10.0"
	puts $writeeq "cutoff	\t\t\t11.0\npairlistdist	\t\t12.0\n\n#RESPA PROPAGATOR\ntimestep	\t\t2.0\nfullElectFrequency	\t2"
	puts $writeeq "nonbondedFreq	\t\t1\n\n#SHAKE\nrigidbonds	\t\tall\nrigidtolerance	\t\t0.000001\nrigiditerations	\t\t400\n\n"
	puts $writeeq "#COM\nComMotion	\t\tno\n\n#FEP PARAMETERS\nsource	\t\t\tfep.tcl\nalch	\t\t\ton"
	puts $writeeq "alchType	\t\tFEP\nalchFile	\t\t[file tail $feppdb]\nalchCol	\t\t\tB\nalchOutFile	\t\teq.fepout"
	puts $writeeq "alchOutFreq	\t\t1000\nalchVdwLambdaEnd	\t1.0"
	puts $writeeq "alchElecLambdaStart	\t0.5"
	puts $writeeq "alchVdWShiftCoeff	\t4.0"
	puts $writeeq "alchDecouple	\t\toff"
	puts $writeeq "alchEquilSteps	\t\t0\nset numSteps	\t\t$fepeqsteps\nset numMinSteps	\t\t$minsteps\n"
	puts $writeeq "runFEPmin 0.0 0.0 0.0 \$numSteps \$numMinSteps $temperature"

	close $writeeq
	return 0
}

#--------------------------------------
#write forward.namd file
#--------------------------------------
proc ::alascan::write_fep_forward {path feppdb feppsf host_write} {
	variable listparm
	variable minsteps
	variable fepwindows
	variable totalfepsteps
	variable eqsteps
	variable temperature 
	global env
	
	set windowsfep [format %.5f [expr 1.0/$fepwindows]]
	set writeeq [open $path/forward.namd w]
	puts $writeeq "#NAMD Config file - autogenerated by Alanine scanning plugin"
	puts $writeeq "#FORWARD DECOUPLE OFF\n"	
	puts $writeeq "#INPUT\n"
	puts $writeeq "set temp	\t\t$temperature"
	foreach parm $listparm {
		puts $writeeq "parameters	\t\t[file tail $parm]"
		}
	puts $writeeq "paraTypeCharmm	\t\ton\nexclude		\t\tscaled1-4\n1-4scaling	\t\t1.0\n"
	puts $writeeq "\n#TOPOLOGY\nstructure	\t\t[file tail $feppsf]\n\n#INITIAL CONDITIONS\n"
	puts $writeeq "coordinates	\t\t[file tail $feppdb]\nbincoordinates	\t\tfep_eq.coor\nbinvelocities	\t\tfep_eq.vel"
	puts $writeeq "extendedsystem	\t\tfep_eq.xsc\n#OUTPUT FREQUENCIES\n"
	puts $writeeq "outputenergies	\t\t100\noutputtiming	\t\t100\noutputpressure	\t\t100"
	puts $writeeq "restartfreq	\t\t100\nXSTFreq		\t\t100\ndcdfreq	\t\t\t2500\n\n#OUTPUT AND RESTART\n"
	puts $writeeq "outputname	\t\tforward\nrestartname	\t\tforward\nbinaryoutput	\t\tyes"
	puts $writeeq "binaryrestart	\t\tyes\n\n#CONSTANT-T\nlangevin	\t\ton"
	puts $writeeq "langevinTemp	\t\t$temperature\nlangevinDamping	\t\t1.0\n\n#PME\n"
	puts $writeeq "PME		\t\tyes\nPMETolerance	\t\t10e-6\nPMEInterpOrder	\t\t4"
	puts $writeeq "PMEGridSpacing	\t\t1.0\n\n"
	puts $writeeq "#WRAP WATER FOR OUTPUT\nwrapAll	\t\t\ton\n\n#CONSTANT-P\n"
	puts $writeeq "LangevinPiston	\t\ton\nLangevinPistonTarget	\t1\nLangevinPistonPeriod	\t100"
	puts $writeeq "LangevinPistonDecay	\t100\nLangevinPistonTemp	\t$temperature\nStrainRate	\t\t0.0 0.0 0.0"
	puts $writeeq "useGrouppressure	\tyes\nuseFlexibleCell	\t\tno\n\n#SPACE PARTITIONING\n"
	puts $writeeq "stepspercycle	\t\t20\nmargin	\t\t\t1.0\n\n#CUT-OFFS\nswitching	\t\ton\nswitchdist	\t\t10.0"
	puts $writeeq "cutoff	\t\t\t11.0\npairlistdist	\t\t12.0\n\n#RESPA PROPAGATOR\ntimestep	\t\t2.0\nfullElectFrequency	\t2"
	puts $writeeq "nonbondedFreq	\t\t1\n\n#SHAKE\nrigidbonds	\t\tall\nrigidtolerance	\t\t0.000001\nrigiditerations	\t\t400\n\n"
	puts $writeeq "#COM\nComMotion	\t\tno\n\n#FEP PARAMETERS\nsource	\t\t\tfep.tcl\nalch	\t\t\ton"
	puts $writeeq "alchType	\t\tFEP\nalchFile	\t\t[file tail $feppdb]\nalchCol	\t\t\tB\nalchOutFile	\t\tforward.fepout"
	puts $writeeq "alchOutFreq	\t\t10\nalchVdwLambdaEnd	\t1.0"
	puts $writeeq "alchElecLambdaStart	\t0.5"
	puts $writeeq "alchVdWShiftCoeff	\t4.0"
	puts $writeeq "alchDecouple	\t\toff"
	puts $writeeq "alchEquilSteps	\t\t$eqsteps\nset numSteps	\t\t$totalfepsteps\n"
	puts $writeeq "runFEP 0.0 1.0 $windowsfep \$numSteps"

	close $writeeq
	return 0
}

#--------------------------------------
#write backward.namd
#--------------------------------------
proc ::alascan::write_fep_backward {path feppdb feppsf host_write} {
	variable w
	variable listparm
	variable minsteps
	variable fepwindows
	variable totalfepsteps
	variable eqsteps
	variable temperature 
	global env
	
	set windowsfep [format %.5f [expr 1.0/$fepwindows]]
	set writeeq [open $path/backward.namd w]
	puts $writeeq "#NAMD Config file - autogenerated by Alanine scanning plugin"
	puts $writeeq "#BACKWARD DECOUPLE OFF\n"	
	puts $writeeq "#INPUT\n"
	puts $writeeq "set temp	\t\t$temperature"
	foreach parm $listparm {
		puts $writeeq "parameters	\t\t[file tail $parm]"
		}
	puts $writeeq "paraTypeCharmm	\t\ton\nexclude		\t\tscaled1-4\n1-4scaling	\t\t1.0\n"
	puts $writeeq "\n#TOPOLOGY\nstructure	\t\t[file tail $feppsf]\n\n#INITIAL CONDITIONS\n"
	puts $writeeq "coordinates	\t\t[file tail $feppdb]\nbincoordinates	\t\tforward.coor\nbinvelocities	\t\tforward.vel"
	puts $writeeq "extendedsystem	\t\tforward.xsc\n#OUTPUT FREQUENCIES\n"
	puts $writeeq "outputenergies	\t\t100\noutputtiming	\t\t100\noutputpressure	\t\t100"
	puts $writeeq "restartfreq	\t\t100\nXSTFreq		\t\t100\ndcdfreq	\t\t\t2500\n\n#OUTPUT AND RESTART\n"
	puts $writeeq "outputname	\t\tbackward\nrestartname	\t\tbackward\nbinaryoutput	\t\tyes"
	puts $writeeq "binaryrestart	\t\tyes\n\n#CONSTANT-T\nlangevin	\t\ton"
	puts $writeeq "langevinTemp	\t\t$temperature\nlangevinDamping	\t\t1.0\n\n#PME\n"
	puts $writeeq "PME		\t\tyes\nPMETolerance	\t\t10e-6\nPMEInterpOrder	\t\t4"
	puts $writeeq "PMEGridSpacing	\t\t1.0"
	puts $writeeq "#WRAP WATER FOR OUTPUT\nwrapAll	\t\t\ton\n\n#CONSTANT-P\n"
	puts $writeeq "LangevinPiston	\t\ton\nLangevinPistonTarget	\t1\nLangevinPistonPeriod	\t100"
	puts $writeeq "LangevinPistonDecay	\t100\nLangevinPistonTemp	\t$temperature\nStrainRate	\t\t0.0 0.0 0.0"
	puts $writeeq "useGrouppressure	\tyes\nuseFlexibleCell	\t\tno\n\n#SPACE PARTITIONING\n"
	puts $writeeq "stepspercycle	\t\t20\nmargin	\t\t\t1.0\n\n#CUT-OFFS\nswitching	\t\ton\nswitchdist	\t\t10.0"
	puts $writeeq "cutoff	\t\t\t11.0\npairlistdist	\t\t12.0\n\n#RESPA PROPAGATOR\ntimestep	\t\t2.0\nfullElectFrequency	\t2"
	puts $writeeq "nonbondedFreq	\t\t1\n\n#SHAKE\nrigidbonds	\t\tall\nrigidtolerance	\t\t0.000001\nrigiditerations	\t\t400\n\n"
	puts $writeeq "#COM\nComMotion	\t\tno\n\n#FEP PARAMETERS\nsource	\t\t\tfep.tcl\n\nalch	\t\t\ton"
	puts $writeeq "alchType	\t\tFEP\nalchFile	\t\t[file tail $feppdb]\nalchCol	\t\t\tB\nalchOutFile	\t\tbackward.fepout"
	puts $writeeq "alchOutFreq	\t\t10\n\nalchVdwLambdaEnd	\t1.0"
	puts $writeeq "alchElecLambdaStart	\t0.5"
	puts $writeeq "alchVdWShiftCoeff	\t4.0"
	puts $writeeq "alchDecouple	\t\toff\n"
	puts $writeeq "alchEquilSteps	\t\t$eqsteps\nset numSteps	\t\t$totalfepsteps\n"
	puts $writeeq "runFEP 1.0 0.0 -$windowsfep \$numSteps"

	close $writeeq
	return 0
}

#------------------------------------------------
#display residue names for selection
#------------------------------------------------
proc ::alascan::display_resnames {} {

 variable w
 variable w1
 variable w1exists
 variable resname
 variable resid
 variable segname
 variable totalres
 variable nonala
 variable sstruct 
 variable nonalasegname {}
 variable seldformutation {}
 variable seldformutsegname {}
 variable seldformutresname {}
 variable notseldformut {}
 variable wextend
set alaresid {}
set nonalasegname {}
set seldformutation {}
set seldformutsegname {}
set seldformutresname {}
set wextend 

set totalres [llength $resid]
set totalheight [expr $totalres * 22]

 if { [winfo exists .disp] } {
	wm deiconify .disp
	return 0
	}

set w1 [toplevel .disp]
wm title $w1 "Select residues"
wm resizable $w1 0 0
set w1exists 1

#selectall, deselectall and submit buttons
proc ::alascan::selall {} {
	variable w1
	foreach selallvar $::alascan::nonala {
	$w1.frame.canvas.id$selallvar select
	}
}

proc ::alascan::deselall {} {
	variable w1
	foreach deselallvar $::alascan::nonala {
	$w1.frame.canvas.id$deselallvar deselect
	}
}

proc ::alascan::submit {} {
	variable w1
 	variable seldformutation {}
	variable notseldformut {} 
	variable seldformutsegname {} 
	variable seldformutresname {}
	variable wextend

	foreach submitvar $::alascan::nonala seg $::alascan::nonalasegname resname $::alascan::nonalaresname {
	if {$::alascan::selval($submitvar) == 1} {
		lappend seldformutation $submitvar
		lappend seldformutsegname $seg
		lappend seldformutresname $resname 
		} elseif {$::alascan::selval($submitvar) == 0} {
			lappend notseldformut $submitvar
		} else {
			puts "Error in residue selection"
			exit
			}		
	}
	
	#host guest system
	if { $::alascan::hg_sel == 1 } {
		set seg_unique_len [llength [lsort -unique $seldformutsegname]]
		if { $seg_unique_len > 1 } {
			tk_messageBox -icon error -type ok -title Message -parent $w1 \
			-message "Host-Guest System!!!\nPlease select residues from either \"Host\" or \"Guest\" system."
			return 0
			}
		}

if {$wextend == 0} {
	::alascan::seledres; destroy $w1 
	} else {
		::alascan::seledres_two; destroy $w1
		}
}

proc ::alascan::rangselect {} {
	variable selectrange
	variable w1
	variable alaresid	

	set selectrangesplit [split $selectrange ","]
	foreach i $selectrangesplit {
		set splitagain [split $i "-"]
		set splitlength [llength $splitagain]
		if {$splitlength == 1} {
			if {[lsearch -inline $alaresid $splitagain] != {} }  { 
				tk_messageBox -icon error -type ok -title Message -parent $w1 \
					-message "Selection includes Ala residue"
				return 0
				} else {
					$w1.frame.canvas.id$splitagain select
					}
			} elseif {$splitlength == 2} {
				set a [lindex $splitagain 0]; set b [lindex $splitagain 1]
				if {$a <= $b} {
						for {set i $a} {$i <=$b} {incr i} {
							$w1.frame.canvas.id$i select
						}	
					} else {
						tk_messageBox -icon error -type ok -title Message -parent $w1 \
						-message "wrong range selection!!\n\"$a-$b\""
						return 0
						}
				}
		}
}

frame $w1.labelsel
grid [label $w1.labelsel.l -text "Choose residues for Alanine scanning" -relief ridge -pady 8]
pack $w1.labelsel -side top -fill x -padx 4 -pady 6 -expand 1
pack $w1.labelsel.l -fill x 

frame $w1.buttons
grid [button $w1.buttons.selectall -text "Select all" -command ::alascan::selall] -row 0 -column 0 -sticky w
grid [button $w1.buttons.deselect -text "Deselect all" -command ::alascan::deselall] -row 0 -column 1 -sticky w
grid [button $w1.buttons.submit -text "Submit" -command ::alascan::submit] -row 0 -column 2 -sticky w
pack $w1.buttons  -padx 2 -pady 2 -expand 1 

#residue selection by range
frame $w1.range
grid [label $w1.range.l1 -text "Select by residue id (eg. 1,3-12,15,20-40):"] -row 0 -column 0 -sticky w
grid [entry $w1.range.path -textvariable ::alascan::selectrange] -row 1 -column 0 -sticky ew
grid [button $w1.range.apply -text "Apply" -command ::alascan::rangselect] -row 1 -column 1 -sticky w
pack $w1.range -side top -padx 2 -pady 0 -expand 1 -fill x 


canvas $w1.col -width 260 -height 25
$w1.col create text 42 14 -text "Secondary\nstructure" -font tkFixed
$w1.col create text 171 7 -text "C  E  T  B  H  G  I" -font tkFixed 
$w1.col create rect 99 12 119 30 -outline snow -fill snow
$w1.col create rect 120 12 139 30 -outline yellow -fill yellow
$w1.col create rect 140 12 159 30 -outline cyan -fill cyan
$w1.col create rect 160 12 179 30 -outline sienna -fill sienna
$w1.col create rect 180 12 199 30 -outline violet -fill violet
$w1.col create rect 200 12 222 30 -outline blue -fill blue
$w1.col create rect 222 12 242 30 -outline red -fill red
$w1.col create rect 99 12 99 30 -outline black -fill black
$w1.col create rect 120 12 120 30 -outline black -fill black
$w1.col create rect 140 12 140 30 -outline black -fill black
$w1.col create rect 160 12 160 30 -outline black -fill black
$w1.col create rect 180 12 180 30 -outline black -fill black
$w1.col create rect 200 12 200 30 -outline black -fill black
$w1.col create rect 222 12 222 30 -outline black -fill black
$w1.col create rect 242 12 242 30 -outline black -fill black
pack $w1.col

frame $w1.frame
canvas $w1.frame.canvas -width 260 -height 500 -yscrollcommand "$w1.frame.right set" \
	-scrollregion [list 0 0 260 $totalheight]

scrollbar $w1.frame.right -orient vertical -command "$w1.frame.canvas yview"
pack $w1.frame -fill x -side top -expand 1
grid $w1.frame.canvas $w1.frame.right -sticky news

set var1 1
	foreach i $resname j $segname k $resid l $sstruct {

		if {$l eq "C"} {set col "snow"
		} elseif {$l eq "E"} {set col "yellow"
		} elseif {$l eq "T"} {set col "cyan"
		} elseif {$l eq "B"} {set col "sienna"
		} elseif {$l eq "H"} {set col "violet"
		} elseif {$l eq "G"} {set col "blue"
		} elseif {$l eq "I"} {set col "red"
		}

		if {$i eq "ALA" || $i eq "Ala" || $i eq "ala"} {
			set resDis [format "%5s %3s %1s" $k $i $j]
			checkbutton $w1.frame.canvas.id$k -text "$resDis" -state disabled -anchor w -font tkFixed
			set var [format %.2f [expr $var1 * 21.6]]
			incr var1
			lappend ::alascan::alaresid $k
			$w1.frame.canvas create window 10 $var -window $w1.frame.canvas.id$k -anchor w
			$w1.frame.canvas create rectangle 120 [expr $var-10.5] 240 [expr $var+10.5] \
				-outline $col -fill $col
		
			} else {
				lappend ::alascan::nonala $k
				lappend ::alascan::nonalasegname $j
				lappend ::alascan::nonalaresname $i
				set resDis [format "%5s %3s %1s" $k $i $j]
				checkbutton $w1.frame.canvas.id$k -text "$resDis" -anchor w -variable ::alascan::selval($k) \
					-font tkFixed
				set var [format %.2f [expr $var1 * 21.6]]
				$w1.frame.canvas create window 10 $var -window $w1.frame.canvas.id$k -anchor w
				$w1.frame.canvas create rectangle 120 [expr $var-10.5] 240 [expr $var+10.5] \
					-outline $col -fill $col
				incr var1
				}
		}
return 0
}

proc alascan_tk {} {
	variable w
	::alascan::alascan_gui
	return $::alascan::w
}





