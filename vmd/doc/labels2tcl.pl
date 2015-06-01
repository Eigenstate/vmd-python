#!/usr/local/bin/perl

if($#ARGV != 1) {
  print STDERR <<END;
usage: labels2tcl.pl labels.pl "http://www.ks.uiuc.edu/...ug" > labels.tcl
END
  exit 1;
}

$labelfile = $ARGV[0];
$URL = $ARGV[1];

if(! -r $labelfile) {
  print STDERR "Can't read $labelfile\n";
  exit 1;
}

# read in the labels
do $labelfile;

# print out the TCL array definition
print "array set external_labels {\n";
foreach $key (keys(%external_labels)) {
  print "  $key $external_labels{$key}#$key\n";
}
print "}\n";
