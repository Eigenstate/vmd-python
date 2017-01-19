#! /bin/perl
# Perl script that replaces the title in the pbctools HTML file with
# the VMD homepage template 

while (<>) {
    if (/END OF MAKETITLE/) {
	last;
    }
    if (/^<div class="author"\s*>(.*)<\/div>$/) {
	$authors = $1;
	$authors =~ s/ <br class="and" \/>/, /g;
    }
    if (/^<div class="date"\s*>Version (.*)<\/div>$/) {
	$version = $1;
    }
}

# HEADER
print <<EOT;
<!--#set var="TITLE" value="PBCTools Plugin, Version $version"-->
<!--#include virtual="/Research/vmd/include/header-vmd.shtml"-->

<!-- HTML docs as used on the VMD homepage in
  http://www.ks.uiuc.edu/Research/vmd/plugins/pbctools/
-->

<div style="text-align:right;font-weight:bold">by $authors</div>

<p style="border:thin solid black;background:yellow;padding:0.5em;font-weight:bold;margin-left:5%;margin-right:5%">
Please note that this is the documentation of the PBCTools plugin as
it comes with the current development version of VMD, and it may
therefore not match the plugin version you are using. If you want to
find the documentation of a specific version of the plugin, please
refer to the <a
href="http://github.com/olenz/pbctools/downloads">PBCTools development
page</a>.
</p>

EOT

# CONTENTS
# copy the rest, stop when the ending tags are found
while (<>) {
    last if /<\/body><\/html>/;
    print;
}

# FOOTER
print <<EOT;
<!--#include virtual="/Research/vmd/include/footer-vmd.shtml"-->
EOT
