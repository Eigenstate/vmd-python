## 
## JMV saved session export script for VMD
## Script version 1.0
##
## $Id: jmvexport.tcl,v 1.4 2013/04/15 16:02:11 johns Exp $
##
 
## Tell Tcl that we're a package and any dependencies we may have
package provide jmvexport 1.0

proc jmvexport {} {
  ::jmvexport::save_session
}

namespace eval ::jmvexport:: {
  namespace export jmvexport

  variable representations 
  variable viewpoints
}


proc ::jmvexport::save_viewpoint {} {
   variable viewpoints
   if [info exists viewpoints] {unset viewpoints}
   # get the current matricies
   foreach mol [molinfo list] {
      set viewpoints($mol) [molinfo $mol get {
        center_matrix rotate_matrix scale_matrix global_matrix}]
   }
}

proc ::jmvexport::save_reps {} {
  variable representations
  foreach mol [molinfo list] {
    set representations($mol) ""
    for {set i 0} {$i < [molinfo $mol get numreps]} {incr i} {
      lappend representations($mol) [molinfo $mol get "{rep $i} {selection $i} {color $i} {material $i}"]
    }
  }
}

proc ::jmvexport::save_session {{file EMPTYFILE} {version 2.0}} {
  variable representations
  variable viewpoints

  # calculate viewpoints and reps for later use
  ::jmvexport::save_viewpoint
  ::jmvexport::save_reps

  # If no file was given, get a filename.  Use the Tk file dialog if 
  # available, otherwise get it from stdin. 
  if {![string compare $file EMPTYFILE]} {
    set title "Enter filename to save current VMD state:"
    set filetypes [list {{VMD files} {.jmv}} {{All files} {*}}]
    if { [info commands tk_getSaveFile] != "" } {
      set file [tk_getSaveFile -defaultextension ".jmv" \
        -title $title -filetypes $filetypes]
    } else {
      puts "Enter filename to save current VMD state:"
      set file [gets stdin]
    }
  }

  if { ![string compare $file ""] } {
    return
  }
  
  set fildes [open $file w]

  switch $version {

    1.0 {
      ::jmvexport::jmv_file_10_header $fildes
      foreach mol [molinfo list] {
        ::jmvexport::jmv_file_10_mol $fildes $mol
      }
      ::jmvexport::jmv_file_10_trailer $fildes
    }
    2.0 {
      ::jmvexport::jmv_file_20_header $fildes
      foreach mol [molinfo list] {
        ::jmvexport::jmv_file_20_mol $fildes $mol
      }
      ::jmvexport::jmv_file_20_trailer $fildes
    }
  }

  close $fildes
}


######### VERSION 1 JMV FILE FORMAT #########
proc ::jmvexport::jmv_file_10_header { fildes } {
  puts $fildes "1.00"
}

proc ::jmvexport::jmv_file_10_trailer { fildes } {
}

proc ::jmvexport::jmv_file_10_mol { fildes mol } {
  variable representations
  variable viewpoints

  set filetype [molinfo $mol get filetype]
  set filename [molinfo $mol get filename]
  switch $filetype {
    webpdb - 
    pdb {
      if [info exists representations($mol)] {
        switch $filetype {
          webpdb {
            puts $fildes "http://www.rcsb.org/pdb/cgi/export.cgi/$filename.pdb?pdbId=$filename;format=PDB"
          }
          pdb {
            puts $fildes "$filename"
          }
        }

        puts $fildes "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0"
        foreach rep $representations($mol) {
          lassign $rep r s c m
          set lrep [concat $r]
          set reptype [::jmvexport::jmv_representation_type [lindex $lrep 0]]
          set colorby [::jmvexport::jmv_colorscheme_type $c]
          # convert selection to atom list
          set newsel [atomselect $mol $s] 
          set selatomlist [$newsel get index]
          $newsel delete

          puts $fildes "$reptype $colorby RGB $selatomlist" 
        }
      }
    }
    default {
      puts "Molecules of type $filetype cannot be loaded by JMV yet."
    }
  }
}


######### VERSION 2 JMV FILE FORMAT #########
proc ::jmvexport::jmv_file_20_header { fildes } {
  puts $fildes "<JMVSAVEDSTATE>"
  puts $fildes "<VERSION>2.0</VERSION>"
}

proc ::jmvexport::jmv_file_20_trailer { fildes } {
  puts $fildes "</JMVSAVEDSTATE>"
}

proc ::jmvexport::jmv_file_20_mol { fildes mol } {
  variable representations
  variable viewpoints

  set filetype [molinfo $mol get filetype]
  set filename [molinfo $mol get filename]
  switch $filetype {
    webpdb - 
    pdb {
      if [info exists representations($mol)] {
        switch $filetype {
          webpdb {
            puts $fildes "<FILENAME>http://www.rcsb.org/pdb/cgi/export.cgi/$filename.pdb?pdbId=$filename;format=PDB</FILENAME>"
          }
          pdb {
            puts $fildes "<FILENAME>$filename</FILENAME>"
          }
        }

        puts $fildes "<TRANSFORMATION>1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0</TRANSFORMATION>"
        puts $fildes "<BACKGROUND>0.0 0.0 0.0</BACKGROUND>"

        foreach rep $representations($mol) {
          lassign $rep r s c m
          set lrep [concat $r]
          set reptype [::jmvexport::jmv_representation_type [lindex $lrep 0]]
          set colorby [::jmvexport::jmv_colorscheme_type $c]
          # convert selection to atom list
          set newsel [atomselect $mol $s] 
          set selatomlist [$newsel get index]
          $newsel delete

          puts $fildes "<MOLECULEITEM>"
          puts $fildes "<REPRESENTATION>$reptype</REPRESENTATION>"
          puts $fildes "<COLORSCHEME>$colorby</COLORSCHEME>"
          puts $fildes "<GRADIENT>RGB</GRADIENT>"
          puts $fildes "<ATOMLIST>$selatomlist</ATOMLIST>"
          puts $fildes "</MOLECULEITEM>"
        }
      }
    }
    default {
      puts "Molecules of type $filetype cannot be loaded by JMV yet."
    }
  }
}


proc ::jmvexport::jmv_representation_type { rep } { 
  set reptype [string toupper $rep]
 
  switch $reptype {
    HBONDS -
    LINES { 
      return LINES
    }

    TRACE {
      return TRACE
    }

    CARTOON -  
    RIBBONS -
    TUBE {
      return TUBE
    }

    DYNAMICBONDS -
    BONDS {
      return BONDS
    }

    LICORICE {
      return LICORICE 
    }

    CPK {
      return CPK
    }

    MSMS -
    SURF -
    DOTTED -
    SOLVENT -
    POINTS -
    VDW {
      return VDW
    }     

    ISOSURFACE -
    VOLUMESLICE -
    OFF {
      return LINES
    }
  } 

  #default 
  return LINES
}

proc ::jmvexport::jmv_colorscheme_type { method } { 
  set colortype [string toupper $method]

  switch $colortype {
    STRUCTURE {
      return STRUCTURE
    }

    SEGNAME {
      return SEGNAME
    }

    CHAIN {
      return CHAIN
    }

    NAME { 
      return NAME
    }
 
    INDEX {
      return INDEX
    }

    RESNAME {
      return RESNAME
    }
  } 

  return NAME
}
