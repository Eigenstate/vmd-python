#
# Molefacture -- structure building plugin
#
# $Id: molefacture.tcl,v 1.104 2011/03/29 19:56:37 johns Exp $
#

package require psfgen
package require idatm
package require runante
package require runsqm
package require readcharmmtop
package require utilities
package require exectool
package provide molefacture 1.3

namespace eval ::Molefacture:: {

   proc initialize {} {
      variable selectcolor lightsteelblue
      global env
      variable toplist
      set deftop [file join $env(CHARMMTOPDIR) top_all27_prot_lipid_na.inp]
      set toplist [::Toporead::read_charmm_topology $deftop]

      variable molidorig
      variable slavemode      0;  # Molefacture can be run as a slave for another application
      variable exportfilename "molefacture_save.xbgf"
      variable mastercallback {}; # When we are done editing in slavemode this callback of the master will be called

      variable origsel     {}
      variable origmolid "" ;# molid of parent molecule
      variable origseltext "" ;# Selection text used to start molefacture
#      variable cursel      {}
#      set cursel [atomselect top none]
      variable bondlist    {}
      variable anglelist   {}
      variable tmpmolid -1
      variable fepmolid -1; # molecule for fep graphics
      variable atommarktags {}
      variable ovalmarktags {}
      variable labelradius  0.2
      variable picklist     {}
      variable pickmode     "atomedit"
      variable totalcharge  0
      variable atomlistformat {}
      variable bondlistformat {}
      variable anglelistformat {}
      variable bondlength {}
      variable dihedral   0
      variable dihedmarktags {}
      variable projectsaved 1

      variable showvale     1
      variable showellp     1

      variable angle 0
      variable anglemarktags {}

#      variable lpmol
#      set lpmol [mol new]

      # list of atoms needing markings in fep mode
      variable atomarktags {}

      variable atomlist    {}
      variable oxidation {}
#      variable atomaddlist {}
      variable valencelist {}
      variable openvalencelist {}
      variable lonepairs {}
      variable vmdindexes {}
      variable chargelist {}
      variable molid      -1
      variable fragmentlist
      variable taglist
      variable templist
      variable atomlistformat
      variable bondtaglist
      variable bondlistformat
      variable aapath [file join $::env(MOLEFACTUREDIR) lib amino_acids]
      variable nucpath [file join $::env(MOLEFACTUREDIR) lib nucleic_acids]
      variable protlasthyd -1
      variable nuclasthyd -1
      variable phi 0;# phi value for protein builder
      variable psi 0 ;# psi value for protein builder
      variable ralpha 180 ;# alpha angle for rna
      variable rbeta  180 ;# beta angle for rna
      variable rgamma 180 ;# gamma angle for rna
      variable rdelta 180 ;# delta angle for rna
      variable repsilon 180 ;# epsilon angle for rna
      variable rchi 180 ;# chi angle for rna
      variable rzeta 180 ;# zeta angle for rna
      variable nuctype "RNA" ;# RNA, DNA, or dsDNA
      variable dihedmoveatom "Atom1"

      variable addfrags ;# Array containing paths to fragments for H replacement, indexed by fragment name
      variable basefrags ;# Array containing paths to fragments used for making new molecules, indexed by basefrag name

      variable periodic {{} H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO  NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB  TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS  IR PT AU HG TL PB BI PO AT RN}
      variable valence
      array set valence {{} {0} H {1} HE {0}  LI {1} BE {2}  B {3}  C {4}  N {3}  O {2}  F {1} NE {0}  NA {1} MG {2} AL {3} SI {4}  P {3 5}  S {2 4 6} CL {1} AR {0}  K {1} CA {2} SC {0} TI {0}  V {0} CR {0} MN {0} FE {2 3 4 6} CO {0} NI {0} CU {0} ZN {0}  GA {3} GE {4} AS {3} SE {2} BR {1} KR {0}  RB {1} SR {2}  Y {0} ZR {0} NB {0} MO {0} TC {0} RU {0} RH {0} PD {0} AG {0} CD {0}  IN {3} SN {4} SB {3} TE {2}  I {1} XE {0}  CS {1} BA {2}  LA {0} CE {0} PR {0} ND {0} PM {0} SM {0} EU {0} GD {0} TB {0} DY {0} HO {0} ER {0} TM {0} YB {0}  LU {0} HF {0} TA {0}  W {0} RE {0} OS {0} IR {0} PT {0} AU {0} HG {0}  TL {0} PB {0} BI {0} PO {0} AT {0} RN {0}}
      variable octet
      array set octet {{} 0 H 2 HE 2  LI 8 BE 8 B 8 C 8 N 8 O 8 F 8 NE 8 NA 8 MG 8 AL 8 SI 8 P 10 S 8 CL 8 AR 8  K 18 CA 18 SC 18 TI 18 V 18 CR 18 MN 18 FE 18 CO 18 NI 18 CU 18 ZN 18  GA 18 GE 18 AS 18 SE 18 BR 18 KR 18 RB 18 SR 18 Y 18 ZR 18  NB 18 MO 18 TC 18 RU 18 RH 18 PD 18 AG 18 CD 18 IN 18 SN 18 SB 18  TE 18  I 18 XE 18 CS 32 BA 0 LA 0 CE 0 PR 0 ND 0 PM 0 SM 0 EU 0 GD 0 TB 0 DY 0 HO 0 ER 0 TM 0 YB 0 LU 0 HF 0 TA 0 W 0 RE 0 OS 0  IR 0 PT 0 AU 0 HG 0 TL 0 PB 0 BI 0 PO 0 AT 0 RN 0 }
   }

   variable mass_by_element
   array set mass_by_element { H 1.00794 He 4.002602 Li 6.941 Be 9.012182 B 10.811 C 12.011 N 14.00674 O 15.9994 F 18.9984032 Ne 20.1797 Na 22.989768 Mg 24.3050 Al 26.981539 Si 28.0855 P 30.973762 S 32.066 Cl 35.4527 K 39.0983 Ar 39.948 Ca 40.078 Sc 44.955910 Ti 47.88 V 50.9415 Cr 51.9961 Mn 54.93805 Fe 55.847 Ni 58.6934 Co 58.93320 Cu 63.546 Zn 65.39 Ga 69.723 Ge 72.61 As 74.92159 Se 78.96 Br 79.904 Kr 83.80 Rb 85.4678 Sr 87.62 Y 88.90585 Zr 91.224 Nb 92.90638 Mo 95.94 Tc 98 Ru 101.07 Rh 102.90550 Pd 106.42 Ag 107.8682 Cd 112.411 In 114.82 Sn 118.710 Sb 121.757 I 126.90447 Te 127.60 Xe 131.29 Cs 132.90543 Ba 137.327 La 138.9055 Ce 140.115 Pr 140.90765 Nd 144.24 Pm 145 Sm 150.36 Eu 151.965 Gd 157.25 Tb 158.92534 Dy 162.50 Ho 164.93032 Er 167.26 Tm 168.93421 Yb 173.04 Lu 174.967 Hf 178.49 Ta 180.9479 W 183.85 Re 186.207 Os 190.2 Ir 192.22 Pt 195.08 Au 196.96654 Hg 200.59 Tl 204.3833 Pb 207.2 Bi 208.98037 Po 209 At 210 Rn 222 Fr 223 Ra 226.025 Ac 227.028 Pa 231.0359 Th 232.0381 Np 237.048 U 238.0289 Am 243 Pu 244 Bk 247 Cm 247 Cf 251 Es 252 Fm 257 Md 258 No 259 Rf 261 Bh 262 Db 262 Lr 262 Sg 263 Hs 265 Mt 266}

   variable availablehyd 0

   variable maxaasize 26 ;# size of the largest residue that could be added
   variable dummyhydincr 300 ;# number of dummy hydrogens added at each step

   variable feptyping "None" ;# typing scheme to use in the fep module
   variable fepstartcharge 0
   variable fependcharge 0
   variable filename_FEP

   # atom types that require impropers for OPLS ATOM TYPES only!!
   # ALL 3 bonded C & N types: 
   variable improper_centeratoms [list C2 C1 N1 N2 N3 CA1 CA3 CA4 CA2 CM1 CM2 CM4 CM3 CT2 CT7 CT3 CT6 CT4 CT1 CT8  C* CN CB2 CV CW1 CX CR1 N21 NA CS1 CS2]
   variable improper_atomorders [list [list C O] [list N] [list CA] [list CM HC CM HC] [list CM HC CM CT] [list CM CT CM HC] [list CM CT CM CT]]

   initialize

}


proc ::Molefacture::set_slavemode { callback filename } {
   variable slavemode 1
   variable exportfilename $filename
   variable mastercallback $callback
}

## Temporary gui for getting molefacture started
proc ::Molefacture::molefacture_start {} {

  if { [winfo exists .molefacstart] } {
    wm deiconify .molefacstart
    focus .molefacstart
    raise .molefacstart
    return
  }



   set w [toplevel ".molefacstart"]
   wm title $w "Molefacture - Molecule Builder"
   wm resizable $w 0 0
   variable atomsel

   set atomsel ""

   label $w.warning -text "Enter a selection below and click \"Start\" to start molefacture \nand edit the atoms of this selection. Please check the documentation \n(accessible through the Help menu) to learn how to use it." -width 55
   frame $w.entry
   label $w.entry.sel -text "Selection: "
   entry $w.entry.entry -textvar [namespace current]::atomsel 
   button $w.entry.go -text "Start Molefacture" -command "[namespace current]::molefacture_gui_aux $[namespace current]::atomsel; wm state $w withdrawn"

   pack $w.entry.sel $w.entry.entry $w.entry.go -side left -fill x
   pack $w.warning $w.entry
}

proc ::Molefacture::molefacture_gui_aux {seltext} {
#  puts "|$seltext|"
  if {$seltext == ""} {
    set mysel ""
  } elseif {$seltext == "index"} {
      tk_messageBox -icon error -type ok -title Error \
      -message "You entered a selection containing no atoms. If you want to create a new molecule, invoke molefacture with no selection. Otherwise, please make a selection containing at least one atom."
      return
  } else {
    set mysel [atomselect top "$seltext"]
    if {[$mysel num] > 200} {
      tk_messageBox -icon error -type ok -title Warning \
	       -message "The current version of molefacture is best used on structures of 200 atoms or smaller. Future versions will be able to handle larger structures. You may continue, but some features may work slowly. See the molefacture documentation for more details."
    }
  }

  variable origseltext
  set origseltext $seltext

  if {$seltext != ""} {
    variable origmolid
    set origmolid [molinfo top]
  }

  ::Molefacture::molefacture_gui $mysel
  if {$mysel != ""} {$mysel delete}
}

###################################################
# Clean up and quit the program.                  #
###################################################

proc ::Molefacture::done { {force 0} } {
   fix_changes
   variable projectsaved 
   variable slavemode
   #variable exportfilename

   if {!$projectsaved && !$force && !$slavemode} {
      set reply [tk_dialog .foo "Quit - save file?" "Quit Molefacture - Save molecule?" \
        questhead  0 "Save" "Don't save" "Cancel"]
      switch $reply {
	 0 { fix_changes; export_molecule_gui}
	 1 { }
	 2 { return 0 }
      }
   }

   # Make the master molecule visible again
   variable molidorig
   if {$molidorig != -1} {molinfo $molidorig set drawn 1}

   # Remove_traces
   foreach t [trace info variable ::vmd_pick_event] {
      trace remove variable ::vmd_pick_event write ::Molefacture::atom_picked_fctn
   }

   # Set mouse to rotation mode
   mouse mode 0
   mouse callback off; 

   if { [winfo exists .molefac] }    { wm withdraw .molefac }

   if {$slavemode} {
      variable exportfilename
      variable mastercallback
      variable atomlist
      variable bondlist
      fix_changes
      export_molecule $exportfilename
      $mastercallback $exportfilename
   }

   variable tmpmolid
   mol delete $tmpmolid

   # Cleanup
   if {[file exists Molefacture_tmpmol.xbgf]} { file delete Molefacture_tmpmol.xbgf }

   # Close molefacture
}

proc molefacture_tk {} {
  # If already initialized, just turn on
  if { [winfo exists .molefacstart] } {
    wm deiconify .molefacstart
  } else {
    ::Molefacture::molefacture_start
  }
  return .molefacstart
}

#Load all procs from other molefacture files
foreach lib { molefacture_builder.tcl \
              molefacture_state.tcl \
              molefacture_geometry.tcl \
              molefacture_gui.tcl \
              molefacture_edit.tcl \
              molefacture_internals.tcl } {
  if { [catch { source [file join $env(MOLEFACTUREDIR) $lib] } errmsg ] } {
    error "ERROR: Could not load molefacture library file $lib: $errmsg"
  }
}  

