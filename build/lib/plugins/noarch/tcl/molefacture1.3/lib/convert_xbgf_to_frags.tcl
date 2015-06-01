package require utilities

proc convert_xbgf_to_mfrag {xbgffile fragprefix longname shortname} {
  set fragmol [mol new $xbgffile]
  set fragsel [atomselect $fragmol all]
  $fragsel moveby [vecscale -1 [measure center $fragsel weight mass]]

  set ofile [open "${fragprefix}.mfrag" w]

  puts $ofile "FRAGMENT $longname $shortname"

  foreach name [$fragsel get name] x [$fragsel get x] y [$fragsel get y] z [$fragsel get z] element [$fragsel get element] {
    puts $ofile "$name $x $y $z $element"
  }

  puts $ofile "BONDS"

  foreach bondlist [$fragsel getbonds] bolist [$fragsel getbondorders] ind [$fragsel get index] {
    foreach bond $bondlist bo $bolist {
      puts $ofile "$ind $bond $bo"
    }
  }

  close $ofile
  $fragsel delete
  mol delete $fragmol
}

proc convert_xbgf_to_afrag {xbgffile fragprefix fragname delh conatom} {
  set fragmol [mol new $xbgffile]
  set fragsel [atomselect $fragmol "not index $delh"]
  set allsel [atomselect $fragmol all]
  set hsel [atomselect $fragmol "index $delh"]
  set csel [atomselect $fragmol "index $conatom"]
  set indzerosel [atomselect $fragmol "index 0"]
  set diffvec [vecsub [join [$csel get {x y z}]] [join [$hsel get {x y z}]]]
  $allsel move [transvecinv $diffvec]
  $allsel moveby [vecscale -1 [join [$hsel get {x y z}]]]

  # properly work out all the bonds and bond orders
  set vals [$indzerosel get {name x y z element}]
  set bondvals [$indzerosel getbonds]
  set bondordvals [$indzerosel getbondorders]
  set conatombonds [$csel getbonds]
  set conatombondorders [$csel getbondorders]

  set bondvals [::util::lmap $bondvals "regsub -all $conatom \$elem 0"]
  set conatombonds [::util::lmap $conatombonds "regsub -all 0 \$elem $conatom"]

  foreach ind $bondvals {
    set sel [atomselect $fragmol "index $ind"]
    $sel setbonds [::util::lmap [$sel getbonds] "regsub -all $conatom \$elem 0"]
    $sel setbonds [::util::lmap [$sel getbonds] "regsub -all 0 \$elem $conatom"]
    $sel delete
  }

  foreach ind $conatombonds {
    set sel [atomselect $fragmol "index $ind"]
    $sel setbonds [::util::lmap [$sel getbonds] "regsub -all $conatom \$elem 0"]
    $sel setbonds [::util::lmap [$sel getbonds] "regsub -all 0 \$elem $conatom"]
    $sel delete
  }

  $csel setbonds $bondvals
  $csel setbondorders $bondordvals
  $indzerosel setbonds $conatombonds
  $indzerosel setbondorders $conatombondorders

  $indzerosel set {name x y z element} [$csel get {name x y z element}]
  $csel set {name x y z element} $vals

  $fragsel writexbgf tmp.xbgf
  $fragsel delete
  set newmol [mol new tmp.xbgf]

  set fragsel [atomselect $newmol all]

  set ofile [open "${fragprefix}.frag" w]
  puts $ofile "FRAGMENT $fragname"

  foreach name [$fragsel get name] x [$fragsel get x] y [$fragsel get y] z [$fragsel get z] element [$fragsel get element] {
    puts $ofile "$name $x $y $z $element"
  }
  puts $ofile "BONDS"

  foreach bondlist [$fragsel getbonds] bolist [$fragsel getbondorders] ind [$fragsel get index] {
    foreach bond $bondlist bo $bolist {
      puts $ofile "$ind $bond $bo"
    }
  }

  close $ofile

  $fragsel delete
  $allsel delete
  $hsel delete
  $csel delete

  mol delete $fragmol
  mol delete $newmol
}



