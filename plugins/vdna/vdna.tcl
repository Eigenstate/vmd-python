##################################
## vdna Version 2.0:   
##  TcBishop, Tulane University
##################################
## $Revision: 1.6 $ 
## $Id: vdna.tcl,v 1.6 2010/03/03 22:14:30 johns Exp $
## 
## Please e-mail improvements or comments to
##  
##  Author: Tom Bishop 
##    bishop@tulane.edu 
##    504-862-3370
##    http://dna.ccs.tulane.edu
###### Please Cite the following if you use VDNA
###     T. C. Bishop, VDNA: The Virtual DNA plug-in for VMD,
###     Bioinformatics  2009, DOI 10.1093/bioinformatics/btp566.

######################################
### USAGE:
### requires VMD 1.8.2  or later from www.ks.uiuc.edu
######################################
###   Version 2.0 now does following:
####     1)uses any valid math expression as helix parameters
###      2) has default values for all sorts of DNA structures
###      3) can draw nucleosome cores
###      4) can make an xyz output file... vdna.xyz
###      5) will make some plots of helical parameters (needs work)
###      6) will save helixparms to file.. vdna.par
### TODO list  
###    helix par output:
###      1) needs broswer to select ouput file currently writes to tmp.par
####   plotting
####     1) can be much improved
####   sequence reader
###     1) needs sequence input area: currently makes up random sequence for par files
####    2) associate some default values with a defined sequence; omega gama would them
###           have to be some lookup table


## Tell Tcl that we're a package and any dependencies we may have
package provide vdna 2.2


namespace eval ::VDNA:: {
  variable w
  variable TiltStrng
  variable RollStrng
  variable TwistStrng
  variable ShiftStrng
  variable SlideStrng
  variable RiseStrng
  variable Geometry
  variable s
  variable sMax
  variable t
  variable tMax
  variable Lk
  variable Nuc
  variable V1Strng
  variable V2Strng
  variable IsNucStrng
  #################################

  ##################################
  ### derived and fixed inputs
  ##################################
  variable Pi 3.14159265358979
  ##################################
  ### cross-section of the rod for drawing
  ##################################
  ### ideal (Wdth)^2 + (Dpth)^2 = (20)^2
  ###  and Dpth = 1.7*Wdth
  variable Wdth       #  width of the rod graphic
  variable Dpth       #  depth of the rod graphic
}

##########################################################
#	PROCEDURE DEFINITIONS
#  1) OMEGA  
#  2) GAMMA
#  3) DRAW_CUBE
#  4) INTEGRATE
#  5) MAKEEND
#  6) RESETVARS
##########################################################
proc ::VDNA::DefOmega {  } {
      variable Pi
      variable TiltStrng
      variable RollStrng
      variable TwistStrng
      variable Lk
      variable Nuc
      variable V1Strng
      variable V2Strng
      variable t
##    puts "DefOmega: $TiltStrng $RollStrng $TwistStrng"

    proc ::VDNA::Omega { s } {
      variable Pi
      variable TiltStrng
      variable RollStrng
      variable TwistStrng
      variable Lk
      variable Nuc
      variable V1Strng
      variable V2Strng
      variable t
     set V1 [expr  $V1Strng  ] 
     set V2 [expr  $V2Strng ] 
     set Tw [expr  $TwistStrng  ] 
     set O3 [expr  $Pi / 180.0 * ( $TwistStrng )  ] 
     set O2 [expr  $Pi / 180.0 * ( $RollStrng )   ] 
     set O1 [expr  $Pi / 180.0 * ( $TiltStrng )   ] 
     return [ list $O1 $O2 $O3  ] 
   }
}

proc ::VDNA::DefGamma {  } {
      variable Pi
      variable ShiftStrng
      variable SlideStrng
      variable RiseStrng
      variable TwistStrng
      variable Lk
      variable Nuc
      variable V1Strng
      variable V2Strng
      variable t

    proc ::VDNA::Gamma { s } {
      variable Pi
      variable ShiftStrng
      variable SlideStrng
      variable RiseStrng
      variable TwistStrng
      variable Lk
      variable Nuc
      variable V1Strng
      variable V2Strng
      variable t
     set V1 [expr $V1Strng ]
     set V2 [expr $V2Strng ]
     set Tw [expr  $TwistStrng ] 
     set G1 [expr  $ShiftStrng ] 
     set G2 [expr  $SlideStrng ] 
     set G3 [expr  $RiseStrng  ]  
     return [ list $G1 $G2 $G3  ] 
   }
}
proc ::VDNA::DefIsNuc {  } { 
    variable IsNucStrng
    variable Lk
    variable Nuc
    variable V1Strng
    variable V2Strng
      variable t

    proc ::VDNA::IsNuc { s } {
     variable IsNucStrng
     variable Lk
     variable Nuc
     variable V1Strng
     variable V2Strng
      variable t
     return [ expr $IsNucStrng  ] 
   }
}


#######################################
### DRAW_SPHERE  at R
#######################################
### R is a  3-vectors 
proc ::VDNA::draw_sphere { R } {
  variable graphmol

#  if {[llength $R] != 4 } { return "ERROR: draw_sphere inproper data format R"}
    graphics $graphmol color [expr {2%2}]
    graphics $graphmol sphere $R radius 1 resolution 10
} 
### end of DRAW_CUBE
#######################################
### DRAW_CUBE from END1 to END2
#######################################
### End1 and End2 are lists of 3-vectors 
### each End contains the coordinates for 4 points in space
proc ::VDNA::draw_cube { End1 End2 } {
  variable graphmol

#  if {[llength $End1] != 4 } { return "ERROR: draw_cube inproper data format End1"}
#  if {[llength $End2] != 4 } { return "ERROR: draw_cube inproper data format End2"}

  for {set k 0 } { $k <= 3 } { incr k } { 
    set Pt(1,$k) [lindex $End1 $k]
    set Pt(2,$k) [lindex $End2 $k]
  }

  for { set k 0 } { $k <=3 } { incr k } {
    set kp  [expr {($k+1)%4}]
    set kpp [expr {($k+2)%4}]
    set km  [expr {($k-1)%4}]

    # normals for first set of triangles
    set n11  [vecnorm [vecsub $Pt(1,$km)  $Pt(1,$k)]]
    set n12  [vecnorm [vecsub $Pt(2,$kpp) $Pt(2,$kp)]]
    set n13  [vecnorm [vecsub $Pt(1,$kpp) $Pt(1,$kp)]]

    # normals for second set of triangles
    set n21  [vecnorm [vecsub $Pt(1,$km)  $Pt(1,$k)]]
    set n22  [vecnorm [vecsub $Pt(2,$km)  $Pt(2,$k)]]
    set n23  [vecnorm [vecsub $Pt(2,$kpp) $Pt(2,$kp)]]

    # use trinorms so that graphics are smooth
    graphics $graphmol color [expr {$k%2}]
    graphics $graphmol trinorm $Pt(1,$k) $Pt(2,$kp) $Pt(1,$kp) $n11 $n12 $n13 
    graphics $graphmol trinorm $Pt(1,$k) $Pt(2,$k)  $Pt(2,$kp) $n21 $n22 $n23
  }
} 
### end of DRAW_CUBE

#######################################
###  INTEGRATE Rod Centerline and Directors
#######################################
###   3-vectors: Omega,Gamma,R
###   9-vector:  D = d1,d2,d3
proc ::VDNA::Integrate { W G R D } {

## translation of centerline to next increment 
###    R(s+ds) = R(s) +  D*G*ds
###    D(s+ds) = D(s) +  D*W*ds
## where D is matrix of directors
### the following algorithm 
### is  adapted from  Hassan and Calladine, JMB, 1995 
###   its not clean but it works
 # director 1 ex

  set XYZ00 [lindex [lindex $D 0 ] 0 ]
  set XYZ01 [lindex [lindex $D 0 ] 1 ] 
  set XYZ02 [lindex [lindex $D 0 ] 2 ]
  # director 2 ey
  set XYZ10 [lindex [lindex $D 1 ] 0 ] 
  set XYZ11 [lindex [lindex $D 1 ] 1 ] 
  set XYZ12 [lindex [lindex $D 1 ] 2 ]
  # director 3 ez
  set XYZ20 [lindex [lindex $D 2 ] 0 ] 
  set XYZ21 [lindex [lindex $D 2 ] 1 ] 
  set XYZ22 [lindex [lindex $D 2 ] 2 ] 

  set R0  [lindex $R 0 ]
  set R1  [lindex $R 1 ]
  set R2  [lindex $R 2 ]


##    set d(1) [lindex $D 0]
##    set d(2) [lindex $D 1]
##    set d(3) [lindex $D 2]

##shift slide rise
##    puts "Integration: setting shift,slide,rise"
    set dRl0 [lindex $G 0] 
    set dRl1 [lindex $G 1]
    set dRl2 [lindex $G 2]
##    puts   "   dRl1,2,3 $dRl0 $dRl1 $dRl2 "
## tilt,roll, twist
    set tilt  [expr {[lindex $W  0 ]}]
    set roll  [expr {[lindex $W  1 ]}]
    set twist [expr {[lindex $W  2 ]}]
##    puts   "   Ti,Ro,Tw $tilt $roll $twist "

 set bend [expr {sqrt($roll*$roll+$tilt*$tilt)}]
 set c2O [expr {cos(0.5*$twist)}]
 set s2O [expr {sin(0.5*$twist)}]
 set cO [expr {cos($twist)}]
 set sO [expr {sin($twist)}]

if [expr $bend != 0] {
  # T[i+1] = M &* T[i]
  # transpose(M)
  set c2G [expr {cos(0.5*$bend)}]
  set s2G [expr {sin(0.5*$bend)}]
  set cG [expr {cos($bend)}]
  set sG [expr {sin($bend)}]
  set tG [expr {$tilt/$bend}]
  set rG [expr {$roll/$bend}]
  set cgp [expr {$cG+1.0}]
  set cgm [expr {$cG-1.0}]
  set trc [expr {$tG*$rG*$cgm}]
  set rct [expr {$rG*$c2O+$tG*$s2O}]
  set rmt [expr {$rG*$s2O-$tG*$c2O}]
  set tr [expr {$tG*$tG-$rG*$rG}]
  set ccg [expr {$cO*$cgp}]
  set tcm [expr {$tr*$cgm}]
  set Mt00 [expr {0.5*($ccg-$tcm)}]
  set Mt01 [expr {0.5*$sO*$cgp-$trc}]
  set Mt02 [expr {$sG*($tG*$s2O-$rG*$c2O)}]
  set Mt10 [expr {-0.5*$sO*$cgp-$trc}]
  set Mt11 [expr {0.5*($ccg+$tcm)}]
  set Mt12 [expr {$sG*($rG*$s2O+$tG*$c2O)}]
  set Mt20 [expr {$sG*$rct}]
  set Mt21 [expr {$sG*$rmt}]
  set Mt22 [expr {$cG}]
  # middle-frame
  set c2gm [expr {$c2G-1.0}]
  set rcg [expr {$rG*$c2gm}]
  set tcg [expr {$tG*$c2gm}]
  set Mm00 [expr {$c2O+$rcg*$rct}]
  set Mm01 [expr {-$s2O-$tcg*$rct}]
  set Mm02 [expr {$s2G*$rct}]
  set Mm10 [expr {$s2O+$rcg*$rmt}]
  set Mm11 [expr {$c2O-$tcg*$rmt}]
  set Mm12 [expr {$s2G*$rmt}]
  set Mm20 [expr {-$s2G*$rG}]
  set Mm21 [expr {$s2G*$tG}]
  set Mm22 [expr {$c2G}]
  #Tm[bp]:=evalm( transpose(XYZ[bp-1]) &* Mm[bp] );
  set Tm00 [expr {$XYZ00 * $Mm00 + $XYZ10 * $Mm10 + $XYZ20 * $Mm20}]
  set Tm01 [expr {$XYZ00 * $Mm01 + $XYZ10 * $Mm11 + $XYZ20 * $Mm21}]
  set Tm02 [expr {$XYZ00 * $Mm02 + $XYZ10 * $Mm12 + $XYZ20 * $Mm22}]
  set Tm10 [expr {$XYZ01 * $Mm00 + $XYZ11 * $Mm10 + $XYZ21 * $Mm20}]
  set Tm11 [expr {$XYZ01 * $Mm01 + $XYZ11 * $Mm11 + $XYZ21 * $Mm21}]
  set Tm12 [expr {$XYZ01 * $Mm02 + $XYZ11 * $Mm12 + $XYZ21 * $Mm22}]
  set Tm20 [expr {$XYZ02 * $Mm00 + $XYZ12 * $Mm10 + $XYZ22 * $Mm20}]
  set Tm21 [expr {$XYZ02 * $Mm01 + $XYZ12 * $Mm11 + $XYZ22 * $Mm21}]
  set Tm22 [expr {$XYZ02 * $Mm02 + $XYZ12 * $Mm12 + $XYZ22 * $Mm22}]

  # XYZ[bp]:=evalm( Mt[bp] &* XYZ[bp-1]);
  set xyz00 $XYZ00
  set XYZ00 [expr {$Mt00*$xyz00+$Mt01*$XYZ10+$Mt02*$XYZ20}]
  set xyz01 $XYZ01
  set XYZ01 [expr {$Mt00*$xyz01+$Mt01*$XYZ11+$Mt02*$XYZ21}]
  set xyz02 $XYZ02
  set XYZ02 [expr {$Mt00*$xyz02+$Mt01*$XYZ12+$Mt02*$XYZ22}]
  set xyz10 $XYZ10
  set XYZ10 [expr {$Mt10*$xyz00+$Mt11*$xyz10+$Mt12*$XYZ20}]
  set xyz11 $XYZ11
  set XYZ11 [expr {$Mt10*$xyz01+$Mt11*$xyz11+$Mt12*$XYZ21}]
  set xyz12 $XYZ12
  set XYZ12 [expr {$Mt10*$xyz02+$Mt11*$xyz12+$Mt12*$XYZ22}]
  set XYZ20 [expr {$Mt20*$xyz00+$Mt21*$xyz10+$Mt22*$XYZ20}]
  set XYZ21 [expr {$Mt20*$xyz01+$Mt21*$xyz11+$Mt22*$XYZ21}]
  set XYZ22 [expr {$Mt20*$xyz02+$Mt21*$xyz12+$Mt22*$XYZ22}]
 } else {
  set Tm00 [expr {$XYZ00*$c2O+$XYZ10*$s2O}]
  set Tm01 [expr {-$XYZ00*$s2O+$XYZ10*$c2O}]
  set Tm02 [expr {$XYZ20}]
  set Tm10 [expr {$XYZ01*$c2O+$XYZ11*$s2O}]
  set Tm11 [expr {-$XYZ01*$s2O+$XYZ11*$c2O}]
  set Tm12 [expr {$XYZ21}]
  set Tm20 [expr {$XYZ02*$c2O+$XYZ12*$s2O}]
  set Tm21 [expr {-$XYZ02*$s2O+$XYZ12*$c2O}]
  set Tm22 [expr {$XYZ22}]

  set xyz00 $XYZ00
  set XYZ00 [expr {$cO*$xyz00+$sO*$XYZ10}]
  set xyz01 $XYZ01
  set XYZ01 [expr {$cO*$xyz01+$sO*$XYZ11}]
  set xyz02 $XYZ02
  set XYZ02 [expr {$cO*$xyz02+$sO*$XYZ12}]
  set XYZ10 [expr {-$sO*$xyz00+$cO*$XYZ10}]
  set XYZ11 [expr {-$sO*$xyz01+$cO*$XYZ11}]
  set XYZ12 [expr {-$sO*$xyz02+$cO*$XYZ12}]
 }

 # dR[bp]:=evalm( Tm[bp] &* dRl[bp] );
 set dR0 [expr {$Tm00*$dRl0+$Tm01*$dRl1+$Tm02*$dRl2}]
 set dR1 [expr {$Tm10*$dRl0+$Tm11*$dRl1+$Tm12*$dRl2}]
 set dR2 [expr {$Tm20*$dRl0+$Tm21*$dRl1+$Tm22*$dRl2}]
 set Rold "$R0 $R1 $R2"
 # R[bp]:=evalm(R[bp-1]+dR[bp]);
 set R0 [expr {$R0+$dR0}]
 set R1 [expr {$R1+$dR1}]
 set R2 [expr {$R2+$dR2}]
 set R "$R0 $R1 $R2"

  set d(1)  " $XYZ00 $XYZ01 $XYZ02 " 
  set d(2)  " $XYZ10 $XYZ11 $XYZ12 "
  set d(3)  " $XYZ20 $XYZ21 $XYZ22 "

     return [list  $R $d(1) $d(2) $d(3) ]
}

###  end of Integrate 

#######################################
###  MAKEEND
#######################################
## inputs: R (centerline) and D (directors)
##  return the 4pts for the END
proc ::VDNA::MakeEnd { RandD } { 

   variable Wdth
   variable Dpth

   set R  [lindex $RandD 0]
   set d1 [lindex $RandD 1]
   set d2 [lindex $RandD 2]
   set d3 [lindex $RandD 3]
  
  set sign(0)  1
  set sign(1)  1
  set sign(2) -1
  set sign(3) -1
  set End ""
  for { set j 0 } { $j <= 3} { incr j } {
    set scale [expr $sign($j)/2.0 * $Wdth]
    set vec1  [vecscale $scale $d1 ]
    set scale [expr $sign([expr ($j-1)%4 ])/2.0 * $Dpth]
    set vec2  [vecscale $scale $d2]
    set Pt($j) [vecadd $vec1 $vec2 ]
    set Pt($j) [vecadd $R $Pt($j) ]
    set End [ lappend End1 $Pt($j)]
  }
   return $End
}

#######################################
###       GEOMETRIES
#######################################
proc ::VDNA::geom args {
  variable Geometry
  puts "$Geometry"
}

#######################################
###      RESETVARS
#######################################
proc ::VDNA::setDefault args {
  variable Pi 3.14159265358979
  variable Wdth 5.0
  variable Dpth 17.2
  variable s  0
  variable sMax 146
  variable t   0
  variable tMax 0
  variable Lk   30
  variable Nuc 146 
  variable V1Strng   4.3
  variable V2Strng   -0.25
  variable IsNucStrng 0
  variable Geometry "Default"

##Values obtained from ABC3, Lavery et. al PCCP  and used for thermal and default models
## parm: avg: std: range: min: max:
### units are degrees and angstrom
##Shift	-0.05	0.76	9.0	-4.4	4.6
##Slide -0.44	0.68	8.7	-3.7	5.0
##Rise	3.32	0.37	4.5	1.4	5.9
##Tilt	-0.3	4.6	58.0	-27.8	28.8
##Roll	3.6	7.2	82.0	-37.3	44.7
##Twist	32.6	7.3	76.0	-17.5	60.4

### Default is set to be the Shear Helix from Bishop BJ 2008 with twist for nucleosome
###   set to average obtained from all 24 structures. Since this is a shear helix the 
###   value of twist does not affect the pitch of superhelix values for free DNA agree w/ ABCII results

 variable ShStd 0.76
 variable SlStd 0.68
 variable RiStd 0.37
 variable TiStd 4.6
 variable RoStd 7.2
 variable TwStd 7.3
}

proc ::VDNA::setvars args {
  variable Pi
  variable s
  variable sMax
  variable t
  variable tMax
  variable TiltStrng
  variable RollStrng
  variable TwistStrng
  variable ShiftStrng
  variable SlideStrng
  variable RiseStrng
  variable Geometry
  variable Lk
  variable Nuc
  variable V1Strng
  variable V2Strng
  variable IsNucStrng
  puts "Geometry is $VDNA::Geometry "
  puts " procedure VDNA::geom "
  VDNA::geom $VDNA::Geometry
   switch $VDNA::Geometry {

      "Straight" {
          puts "Straight Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 0 "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      50.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }

      "Bend(Roll)" {
          puts " Rolled Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 2.0"
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      40
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }

      "Bend(Tilt)" {
          puts " Tilted Rod "     
          set TiltStrng   " 2.0 "
          set RollStrng   " 0 "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      40
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }


      "Tilt-a-gon" {
          puts " Tilted Polygon"     
          set TiltStrng   " 360/\$V1"
          set RollStrng   " 0 "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 20 "
          set sMax      10
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  5
          set V2Strng  0
          set IsNucStrng  0
       }

      "Roll-a-gon" {
          puts " Rolled Polygon"     
          set TiltStrng   " 0 "
          set RollStrng   " 360/\$V1"
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 10 "
          set sMax      6
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  3
          set V2Strng  0
          set IsNucStrng  0
       }

      "RollTilt-a-gon" {
          puts " Polygon with Roll and Tilt"     
          set TiltStrng   " 360/( \$V1 * sqrt(2.0))"
          set RollStrng   " 360/( \$V1 * sqrt(2.0))"
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 20 "
          set sMax      16
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  8 
          set V2Strng  0
          set IsNucStrng  0
       }

      "Shear(Shift)" {
          puts " Shifted Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 0 "
          set TwistStrng  " 0 "
          set ShiftStrng  " 2 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      50.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }

      "Shear(Slide)" {
          puts " Slided Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 0 "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 2 "
          set RiseStrng   " 3.4 "
          set sMax      50.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }

      "Twist(DNA)" {
          puts " Twisted Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 0 "
          set TwistStrng  " 35.2 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      50.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }

      "Untwisted DNA" {
          puts " Twisted Rod "     
          set TiltStrng   " 0 "
          set RollStrng   " 0 "
          set TwistStrng  " 36.0 * sin ( 2*\$Pi / 70 * \$s ) *sin ( 2*\$Pi / 70 * \$s ) * \$s/ 35.0"
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      70.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  0
          set IsNucStrng  0
       }
      "Circle(Roll)" {
          puts " Rolled Circle "     
          set TiltStrng   " 0 "
          set RollStrng   " 360.0 / \$V1  "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      49.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  50
          set V2Strng  0
          set IsNucStrng  0
       }

      "Circle(Tilt)" {
          puts " Tilted Circle "     
          set TiltStrng   " 360.0 / \$V2 "
          set RollStrng   " 0   "
          set TwistStrng  " 0 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      49.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  0
          set V2Strng  50
          set IsNucStrng  0
       }

      "Circular DNA" {
          puts " DNA Circle "     
          set TiltStrng   " 360.0 / \$V1 * sin (\$Pi/180.0 * \$Tw * \$s ) "
          set RollStrng   " 360.0 / \$V1 * cos (\$Pi/180.0 * \$Tw * \$s ) "
          set TwistStrng  " \$V2*360.0/\$V1 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      69.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  70
          set V2Strng  6
          set IsNucStrng  0
       }

      "Torsion Helix(+)" {
          puts " R.H. Torsion Helix "     
          set TiltStrng   " 360.0 / \$V1 * sin (\$Pi/180.0 * (\$Tw + \$V2) * \$s ) "
          set RollStrng   " 360.0 / \$V1 * cos (\$Pi/180.0 * (\$Tw + \$V2) * \$s ) "
          set TwistStrng  " 35.2 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      140.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  70
          set V2Strng   -0.5
          set IsNucStrng  0
       }
 
      "Torsion Helix(-)" {
          puts " L.H. Torsion Helix "     
          set TiltStrng   " 360.0 / \$V1 * sin (\$Pi/180.0 * (\$Tw + \$V2) * \$s ) "
          set RollStrng   " 360.0 / \$V1 * cos (\$Pi/180.0 * (\$Tw + \$V2) * \$s ) "
          set TwistStrng  " 35.2 "
          set ShiftStrng  " 0 "
          set SlideStrng  " 0 "
          set RiseStrng   " 3.4 "
          set sMax      140.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  70
          set V2Strng   0.5
          set IsNucStrng  0
       }
 
      "Shear Helix(+)" {
          puts " R.H. Sheared Helix "     
          set TiltStrng   " 360.0 / \$V1 * sin (\$Pi/180.0 * \$Tw * \$s ) "
          set RollStrng   " 360.0 / \$V1 * cos (\$Pi/180.0 * \$Tw * \$s ) "
          set TwistStrng  " 36.0 "
          set ShiftStrng  " \$V2 * sin (\$Pi/180.0 * \$Tw * \$s ) "
          set SlideStrng  " \$V2 * cos (\$Pi/180.0 * \$Tw * \$s ) "
          set RiseStrng   " 3.4 "
          set sMax      140.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  70
          set V2Strng  0.3
          set IsNucStrng  0
       }
 
      "Shear Helix(-)" {
          puts " L.H. Sheared Helix "     
          set TiltStrng   " 360.0 / \$V1 * sin (\$Pi/180.0 * $\Tw * \$s ) "
          set RollStrng   " 360.0 / \$V1 * cos (\$Pi/180.0 * \$Tw * \$s ) "
          set TwistStrng  " 36.0 "
          set ShiftStrng  " \$V2 * sin (\$Pi/180.0 * $\Tw * \$s ) "
          set SlideStrng  " \$V2 * cos (\$Pi/180.0 * $\Tw * \$s ) "
          set RiseStrng   " 3.4 "
          set sMax      140.0
          set tMax      0
          set Lk       0
          set Nuc      0
          set V1Strng  70
          set V2Strng  -0.3
          set IsNucStrng  0
       }
      "Thermal" {
          puts "Thermal"     
          set TiltStrng   " -0.3 +  4.6 * \[ gdist \] "
          set RollStrng   "  3.6 +  7.2 * \[ gdist \] "
          set TwistStrng  " 32.6 +  7.3 * \[ gdist \]"
          set ShiftStrng  " -0.05 + 0.76 * \[ gdist \] "
          set SlideStrng  " -0.44 + 0.68 * \[ gdist \] "
          set RiseStrng   "  3.32 + 0.37 * \[ gdist \]"
          set sMax      150
          set tMax      25
          set Lk       0
          set Nuc      0
          set V1Strng  0.0
          set V2Strng   0.0
          set IsNucStrng  0

       }

       "Trajectory" {
          puts "Trajectory"     
          set TiltStrng   "  (\[IsNuc  \$s \] ) ? \$V1 * sin ( \$Pi/180.0 * 36.0 * \$s  ) : 0 "
          set RollStrng   "  (\[IsNuc  \$s \] ) ? \$V1 * cos ( \$Pi/180.0 * 36.0 * \$s  ) : 0 "
          set TwistStrng  "   36.0 "
          set ShiftStrng  "  (\[IsNuc  \$s \] ) ? \$V2 * sin ( \$Pi/180.0 * 36.0 * \$s) : 0"
          set SlideStrng  "  (\[IsNuc  \$s \] ) ? \$V2 * cos ( \$Pi/180.0 * 36.0 * \$s) : 0 "
          set RiseStrng   "  3.3 "
          set sMax      170
          set tMax      20
          set Lk       30
          set Nuc      146
          set V1Strng   4.3
          set V2Strng   -0.25
          set IsNucStrng   " (\$s < (\$Nuc - \$t * 1)  && \$s > \$t *1 ) ? 1 : 0 "

       }

      "Chromatin1" {
          puts "Chromatin"     
          set Lk       28
          set Nuc      146
          set Tw  34.5
          set V1Strng  4.2
          set V2Strng  -0.25
          set TiltStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ?  \$V1 * sin (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 0 " 
          set RollStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ?  \$V1 * cos (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 0 " 
          set TwistStrng  " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?  34.5 : 35.0"
 
          set ShiftStrng   " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?   \$V2 * sin (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 0  "
          set SlideStrng   " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?   \$V2 * cos (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc
) * (\$Lk + \$Nuc )))  ) : 0  "
          set RiseStrng   " 3.4 "
          set sMax      3000
          set tMax      0
          set IsNucStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ? 1 : 0 " 
       }

      "Chromatin2" {
          puts "Chromatin"     
          set Lk       33
          set Nuc      146
          set Tw       34.5
          set V1Strng  "1.70 * 360.0 / 146.0 "
          set V2Strng  "-20.6 * 1.70   / 146.0 "
          set IsNucStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ? 1 : 0 " 
          set TiltStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ?  \$V1 * sin (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 2 " 
          set RollStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ?  \$V1 * cos (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 0 " 
          set TwistStrng  " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?  34.5 : 0 "
 
          set ShiftStrng   " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?   \$V2 * sin (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc ))) ) : 0  "
          set SlideStrng   " (\$s < int (\$s / (\$Nuc + \$Lk) ) *  ( \$Nuc + \$Lk )  + \$Nuc ) ?   \$V2 * cos (\$Pi/180.0 * \$Tw * (\$s -  int (\$s / (\$Lk + \$Nuc
) * (\$Lk + \$Nuc )))  ) : 0  "
          set RiseStrng   " 3.32 "
          set sMax      3000
          set tMax      0
          set IsNucStrng "(\$s < int (\$s / (\$Lk + \$Nuc ) * (\$Lk + \$Nuc )  + \$Nuc)) ? 1 : 0 " 
       }

        "Default" { 
          puts " Default Rod "

          set TiltStrng   " (\[IsNuc  \$s \] ) ? \$V1 * sin ( \$Pi/180.0 * 34.95 * \$s  ) : -0.3 +  4.6 * \[ gdist \] "
          set RollStrng   " (\[IsNuc  \$s \] ) ? \$V1 * cos ( \$Pi/180.0 * 34.95 * \$s  ) :  3.6 +  7.2 * \[ gdist \] "
          set TwistStrng  " (\[IsNuc  \$s \] ) ?  34.95 : 32.6 +  7.3 * \[ gdist \] "
          set ShiftStrng  " (\[IsNuc  \$s \] ) ? \$V2 * sin ( \$Pi/180.0 * 34.95 * \$s) : -0.05 + 0.76 * \[ gdist \] "
          set SlideStrng  " (\[IsNuc  \$s \] ) ? $\V2 * cos ( \$Pi/180.0 * 34.95 * \$s) : -0.44 + 0.68 * \[ gdist \] "
          set RiseStrng   " (\[IsNuc  \$s \] ) ?  3.4 :  3.32 + 0.37 * \[ gdist \] "
          set Lk       30
          set Nuc      146
          set sMax     " 230"
          set tMax      0
          set IsNucStrng  " (\$s < (\$Nuc + 40 -  \$t )  && \$s > 40 + \$t ) ? 1 : 0  "
       }
      }
}

#######################################
###       Make XYZ File
#######################################
proc ::VDNA::hp2xyz { } {
  variable Pi 
  variable TiltStrng
  variable RollStrng
  variable TwistStrng
  variable ShiftStrng
  variable SlideStrng
  variable RiseStrng
  variable Geometry
  variable s
  variable sMax
  variable t
  variable tMax
  variable Lk
  variable Nuc

  VDNA::DefIsNuc
  VDNA::DefOmega 
  VDNA::DefGamma

  set Cfrmt  "CA     %20.5f %20.5f %20.5f"
  set H1frmt "H1      %20.5f %20.5f %20.5f"
  set H2frmt "H2      %20.5f %20.5f %20.5f"
  set H3frmt "H3     %20.5f %20.5f %20.5f"
  set fp [open "vdna.xyz" "w" ]
  set t 0
  while {$t <= $tMax } { 
  puts $fp [format "%d "  [expr int (($sMax + 1 ) * 4.0) ] ]
  puts $fp "  COMMENT  VDNA 2.0 par2xyz "
  set R  { 0.0 0.0 0.0 }
  set d1 { 1.0 0.0 0.0 }
  set d2 { 0.0 1.0 0.0 }
  set d3 { 0.0 0.0 1.0 }
  set d3zero $d3
  set D [list $d1 $d2 $d3 ]
  set RandD [list $R $d1 $d2 $d3 ]
  set H1 [vecadd $R  $d1 ]
  set H2 [vecadd $R  $d2 ]
  set H3 [vecadd $R  $d3 ]
  puts  $fp [format $Cfrmt [lindex $R 0 ] [lindex $R 1 ] [lindex $R 2] ]
  puts  $fp [format $H1frmt [lindex $H1  0 ] [lindex $H1 1 ] [lindex $H1 2] ]
  puts  $fp [format $H2frmt [lindex $H2  0 ] [lindex $H2 1 ] [lindex $H2 2] ]
  puts  $fp [format $H3frmt [lindex $H3  0 ] [lindex $H3 1 ] [lindex $H3 2] ]
  for {set s 1} {$s <= $sMax} {incr s 1} {
    set Om3  [ Omega $s ]
    set Ga3  [ Gamma $s ]
    set RandD [ Integrate $Om3 $Ga3 $R $D ]
    set R [lindex $RandD 0]
    set D [lrange $RandD 1 end]
    set d1 [lindex $RandD 1 ]
    set d2 [lindex $RandD 2 ]
    set d3 [lindex $RandD 3 ]
     set H1 [vecadd $R  $d1 ]
     set H2 [vecadd $R  $d2 ]
     set H3 [vecadd $R  $d3 ]
  puts  $fp [format $Cfrmt [lindex $R 0 ] [lindex $R 1 ] [lindex $R 2] ]
  puts  $fp [format $H1frmt [lindex $H1  0 ] [lindex $H1 1 ] [lindex $H1 2] ]
  puts  $fp [format $H2frmt [lindex $H2  0 ] [lindex $H2 1 ] [lindex $H2 2] ]
  puts  $fp [format $H3frmt [lindex $H3  0 ] [lindex $H3 1 ] [lindex $H3 2] ]
  }
  incr t
}
  close $fp
    mol new vdna.xyz type xyz first 0 last -1 step 1 waitfor 1
    mol modselect 0 [molinfo top] "name CA H1"
    mol modstyle 0 [molinfo top]  VDW 1.0 8.0
}
#######################################
###       Export Helix Parameters
#######################################
proc ::VDNA::SaveHP { } {
  variable sMax
  variable tMax
  variable Pi 
  VDNA::DefIsNuc
  VDNA::DefOmega 
  VDNA::DefGamma
   set seq(0) "A-T "
   set seq(1) "T-A "
   set seq(2) "C-G "
   set seq(3) "G-C "
  set frmt "%6d %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f"
  set scale [expr 180.0/$Pi ]
  set fp [open "vdna.par" "w" ]
  puts $fp [format "%4d base-pairs"  [expr int ($sMax) ] ]
  puts $fp "   0  ***local base-pair & step parameters*** "
  puts $fp "       Shear  Stretch  Stagger Buckle Prop-Tw Opening   Shift  Slide    Rise    Tilt    Roll   Twist "
  for {set j 0} {$j < $sMax} {incr j 1} {
    set seqno [expr  int(floor(4*rand())) ]
    set om [VDNA::Omega $j ]
    set om [vecscale $scale  $om ]
    set gam [VDNA::Gamma $j ]
    puts -nonewline $fp $seq($seqno)
    for {set k 1 } { $k <= 6 } { incr k } {
      puts -nonewline $fp [format "%8.2f" 0.0 ]
    }
    for {set k 0 } { $k < 3 } { incr k } {
      puts -nonewline $fp [format "%8.2f" [lindex $gam $k ]]
    }
    for {set k 0 } { $k < 3 } { incr k } {
      puts -nonewline $fp [format "%8.2f" [lindex $om  $k ]]
    }
    puts $fp " " 
  }
  close $fp

}

#######################################
###       Plotting Helix Parameters
#######################################

proc ::VDNA::index2rgb {i} {
  set len 2
  lassign [colorinfo rgb $i] r g b
  set r [expr int($r*255)]
  set g [expr int($g*255)]
  set b [expr int($b*255)]
  #puts "$i      $r $g $b"
  return [format "#%.${len}X%.${len}X%.${len}X" $r $g $b]
}

proc ::VDNA::PlotHP {} {
  variable sMax
  variable tMax
  variable Pi 
  VDNA::DefIsNuc
  VDNA::DefOmega 
  VDNA::DefGamma
  set scale [expr 180.0/$Pi ]
  for {set j 0} {$j <= $sMax} {incr j 1} {
    lappend x  $j
    set om [VDNA::Omega $j ]
    puts "Plot $j $Pi $om "
    puts [vecscale  $scale $om ]
    set om [vecscale $scale  $om ]
    set gam [VDNA::Gamma $j ]
    puts "Plot $j $Pi $gam " 
    for {set k 0 } { $k < 3 } { incr k } {
      lappend y($k) [lindex $om $k ]
      set m [expr $k + 3]
      lappend y($m) [lindex $gam $k]
    }
  }

  if [catch {package require multiplot} msg] {
     showMessage "Plotting in Multiplot not available: package multiplot not installed!\nDo you have the latest VMD version?"
      return
   }
   
    set labels {"Tilt(deg/bp)" "Roll(deg/bp)" "Twist(deg/bp)" "Shift(A/bp)" "Slide(A/bp)" "Rise(A/bp)" } 
  
    set title "Rotational Parameters" 
    set xlab "Position"
    set rotpltp [multiplot -title $title -xlabel $xlab -nostats]
   for {set k 0} {$k < 3} {incr k 1} {
    set color [index2rgb $k]
    puts $color
    set leg [lindex $labels $k]
    $rotpltp add $x $y($k) -marker point -radius 3 -fillcolor $color -linecolor $color -nostats  -legend $leg
     }
   $rotpltp replot

  
    set title "Translational Parameters" 
    set xlab "Position"
    set transpltp [multiplot -title $title -xlabel $xlab -nostats]
   for {set k 3} {$k < 6} {incr k 1} {
    set color [index2rgb $k]
    puts $color
    set leg [lindex $labels $k]
    $transpltp add $x $y($k) -marker point -radius 3 -fillcolor $color -linecolor $color -nostats  -legend $leg
     }
   $transpltp replot
}


#######################################
###      Generate a random number
##          the random numbers have a gaussian
##           or normal distribution with
##             stdev (width 1) and average value 0
#######################################
proc ::VDNA::gdist {} {
set w  1
 while { $w >= 1 } {
 set x1  [expr {2.0 * rand() - 1.0}] 
 set x2  [expr {2.0 * rand() - 1.0}] 
 set w   [expr {$x1 * $x1 + $x2 * $x2} ]
}
 set  w [expr  {sqrt( (-2.0 * log( $w ) ) / $w )} ]
 set y1 [expr  {$x1 * $w } ]
 set y2 [expr  {$x2 * $w } ]
 return $y1
}


 

#########################
### INITIALIZE MAIN  LOOP
#########################

proc ::VDNA::vdna {} {
  variable Pi 
  variable graphmol
  variable TiltStrng
  variable RollStrng
  variable TwistStrng
  variable ShiftStrng
  variable SlideStrng
  variable RiseStrng
  variable Geometry
  variable s
  variable sMax
  variable t
  variable tMax
  variable Lk
  variable Nuc

  ### initialize the graphics
  if {[info exists graphmol]} { mol delete $graphmol}
  mol new
  set graphmol [molinfo top]

  VDNA::DefIsNuc
  puts " Defining Omega and Gamma"
  puts  "Tilt, Roll, Twist : "
  puts    " $TiltStrng $RollStrng $TwistStrng"
  VDNA::DefOmega 
  puts  "Shift, Slide, Rise : "
  puts    " $ShiftStrng $SlideStrng $RiseStrng"
  VDNA::DefGamma
  puts " Omega at S = 0 "
  puts [ Omega 0  ] 
  puts " Gamma at S = 0 "
  puts [ Gamma 0  ]

 set t 0 
 while {$t <= $tMax} { 
  ##################################
  ##### orientation of first basepair
  ##################################
  set R { 0.0 0.0 0.0 }   
  set d1 { 1.0 0.0 0.0 }  
  set d2 { 0.0 1.0 0.0 }  
  set d3 { 0.0 0.0 1.0 }  
  set D [list $d1 $d2 $d3 ]
  set RandD [list $R $d1 $d2 $d3 ]

  set End1 [MakeEnd $RandD ]

  set NucCenter { 0.0 0.0 0.0 }
  set nuccnt 0
  set iNuc 0

  
  #########################
  ###  MAIN  LOOP
  #########################
  ### there is one main   loops
  ###   For (s =0;s< sMax;s++)
  ###         draw basepairs
  ###   End
  ################################

  display update off
   set s 0 
   while {$s < $sMax} { 
    set nuc [IsNuc $s]
    set sp [expr { $s +1 } ]

    set Om3  [ Omega $sp ] 
    set Ga3  [ Gamma $sp ]

    set RandD [ Integrate $Om3 $Ga3 $R $D ] 
    set R [lindex $RandD 0]
    set D [lrange $RandD 1 end]

    set End2  [MakeEnd $RandD ]
    draw_cube $End1 $End2
    if {$nuc == 1} {
         set NucCenter [vecadd $NucCenter $R ]
         incr nuccnt
    } elseif {$nuc ==0} {
       if {$nuccnt > 0 } {
        incr iNuc
        set NucCenter [vecscale [expr 1.0/$nuccnt ] $NucCenter  ]
        draw_sphere $NucCenter
        graphics $graphmol color [expr {int($iNuc)%7 }]
        graphics $graphmol sphere $NucCenter radius 40 resolution 20
        }
       set NucCenter { 0.0 0.0 0.0 }
       set nuccnt 0
    }
	
 ## set up for next iteration
    set End1 $End2
    set s $sp
 }
  puts " current time is $t "
  incr t
}
  incr t -1
  display projection orthographic
  display update on
  display resetview
  rock x by .1
  rock y by .1
}

proc ::VDNA::vdnatk {} {
  variable Pi
  variable Wdth 
  variable Dpth 
  variable w
  variable TiltStrng
  variable RollStrng
  variable TwistStrng
  variable ShiftStrng
  variable SlideStrng
  variable RiseStrng
  variable Geometry
  variable s
  variable sMax 
  variable t
  variable tMax
  variable Lk
  variable Nuc


  # If already initialized, just turn on
  if { [winfo exists .vdnatk] } {
    wm deiconify $w
    return
  }
  trace add variable VDNA::Geometry write VDNA::setvars 
  VDNA::setDefault

  set w [toplevel ".vdnatk"]
  wm title $w "vdna Tool Kit" 
  wm resizable $w 1 0

  ##
  ## make the menu bar
  ##
  frame $w.menubar -relief raised -bd 2 ;# frame for menubar
  pack $w.menubar -padx 1 -fill x

  menubutton $w.menubar.help -text "Help" -underline 0 -menu $w.menubar.help.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.help config -width 5

  ##
  ## help menu
  ##
  menu $w.menubar.help.menu -tearoff no
  $w.menubar.help.menu add command -label "Help @ UIUC" -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/vdna"

  $w.menubar.help.menu add command -label "VDNA Home" -command "vmd_open_url http://dna.ccs.tulane.edu/"

  pack $w.menubar.help -side right


  ##
  ##  Tilt
  ##
  frame $w.tilt ;#  
  label $w.tilt.label -text "Tilt:  "
  entry $w.tilt.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::TiltStrng
  pack $w.tilt.label $w.tilt.entry -side left -anchor w

  ##
  ##  Roll
  ##
  frame $w.roll ;# 
  label $w.roll.label -text "Roll:  "
  entry $w.roll.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::RollStrng
  pack $w.roll.label $w.roll.entry -side left -anchor w

  ##
  ##  Twist
  ##
  frame $w.twist ;#  
  label $w.twist.label -text "Twist: "
  entry $w.twist.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::TwistStrng
  pack $w.twist.label $w.twist.entry -side left -anchor w


  ##
  ##  Shift
  ##
  frame $w.shift ;#  
  label $w.shift.label -text "Shift: "
  entry $w.shift.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::ShiftStrng
  pack $w.shift.label $w.shift.entry -side left -anchor w

  ##
  ##  Slide
  ##
  frame $w.slide ;#  
  label $w.slide.label -text "Slide: "
  entry $w.slide.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::SlideStrng
  pack $w.slide.label $w.slide.entry -side left -anchor w

  ##
  ##  Rise
  ##
  frame $w.rise ;#  
  label $w.rise.label -text "Rise: "
  entry $w.rise.entry -width 40 -relief sunken -bd 2 \
    -textvariable ::VDNA::RiseStrng
  pack $w.rise.label $w.rise.entry -side left -anchor w


  ##
  ##  OTher Variables
  ##
  frame $w.vars ;#  

  label $w.vars.slabel -text "Max S: "
  entry $w.vars.sentry -width 5 -relief sunken -bd 2 \
    -textvariable ::VDNA::sMax

  label $w.vars.tlabel -text "Max T: "
  entry $w.vars.tentry -width 5 -relief sunken -bd 2 \
    -textvariable ::VDNA::tMax

  label $w.vars.lklabel -text "Lk: "
  entry $w.vars.lkentry -width 5 -relief sunken -bd 2 \
    -textvariable ::VDNA::Lk

  label $w.vars.nlabel -text "Nuc: "
  entry $w.vars.nentry -width 5 -relief sunken -bd 2 \
    -textvariable ::VDNA::Nuc
  pack $w.vars.slabel $w.vars.sentry $w.vars.tlabel $w.vars.tentry $w.vars.lklabel $w.vars.lkentry \
      $w.vars.nlabel $w.vars.nentry -side left -anchor w

  ##
  ##  Extra Variables
  ##
  frame $w.xvars ;#  

  label $w.xvars.v1label -text "V1: "
  entry $w.xvars.v1entry -width 10 -relief sunken -bd 2 \
    -textvariable ::VDNA::V1Strng

  label $w.xvars.v2label -text "V2: "
  entry $w.xvars.v2entry -width 10 -relief sunken -bd 2 \
    -textvariable ::VDNA::V2Strng

  label $w.xvars.v3label -text "Cores: "
  entry $w.xvars.v3entry -width 20 -relief sunken -bd 2 \
    -textvariable ::VDNA::IsNucStrng

  pack $w.xvars.v1label $w.xvars.v1entry \
       $w.xvars.v2label $w.xvars.v2entry \
       $w.xvars.v3label $w.xvars.v3entry \
      -side left -anchor w

  ##
  ##  Go and Reset buttons
  ##
  frame $w.goreset        ;# frame for Go buttons
  button $w.goreset.gobutton     -text "Draw It" -command VDNA::vdna
  tk_optionMenu $w.goreset.geometry  VDNA::Geometry \
       "Straight"      \
       "Bend(Roll)"    \
       "Bend(Tilt)"    \
       "Shear(Shift)"  \
       "Shear(Slide)"  \
       "Roll-a-gon"    \
       "Tilt-a-gon"    \
       "RollTilt-a-gon"    \
       "Circle(Roll)" \
       "Circle(Tilt)"  \
       "Twist(DNA)"    \
       "Untwisted DNA"       \
       "Circular DNA"  \
       "Torsion Helix(+)"         \
       "Torsion Helix(-)"         \
       "Shear Helix(+)"         \
       "Shear Helix(-)"         \
       "Chromatin1"        \
       "Chromatin2"        \
       "Thermal"            \
       "Trajectory"           
  button $w.goreset.resetbutton  -text "Reset All" -command  VDNA::setDefault

  pack $w.goreset.gobutton $w.goreset.geometry $w.goreset.resetbutton \
   -side left -anchor w

  ##
  ##   Plot; Save; XYZ buttons
  ##
  frame $w.plotsave        ;# frame for Go buttons
  button $w.plotsave.plot     -text "Plot Parameters" -command VDNA::PlotHP
  button $w.plotsave.save     -text "Save Parameters" -command VDNA::SaveHP
  button $w.plotsave.xyz     -text "Make Molec" -command VDNA::hp2xyz

  pack $w.plotsave.plot $w.plotsave.save $w.plotsave.xyz\
   -side left -anchor w

  ##
  ## Progress area
  ##

  frame $w.status
  label $w.status.label -text " Position: "
  label $w.status.step  -textvariable ::VDNA::s
  label $w.status.slash -text " of "
  label $w.status.steps -textvariable ::VDNA::sMax
  pack  $w.status.label $w.status.step $w.status.slash \
    $w.status.steps -side left -anchor w

  frame $w.tstatus
  label $w.tstatus.label -text " Time: "
  label $w.tstatus.step  -textvariable ::VDNA::t
  label $w.tstatus.slash -text " of "
  label $w.tstatus.steps -textvariable ::VDNA::tMax
  pack  $w.tstatus.label $w.tstatus.step $w.tstatus.slash \
    $w.tstatus.steps -side left -anchor w
  ##
  ## pack up the main frame
  ##
  pack $w.tilt  $w.roll  $w.twist \
       $w.shift $w.slide $w.rise \
       $w.vars \
       $w.xvars \
       $w.status \
       $w.tstatus \
       $w.goreset \
       $w.plotsave \
       -side top -pady 10 -fill x -anchor w
}

# This gets called by VMD the first time the menu is opened.
proc vdna_tk_cb {} {
  ::VDNA::vdnatk
  return $VDNA::w
}


#VDNA::vdnatk
