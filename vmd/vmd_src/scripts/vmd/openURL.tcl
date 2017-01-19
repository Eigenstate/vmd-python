############################################################################
#cr                                                                       
#cr            (C) Copyright 1995-2007 The Board of Trustees of the            
#cr                        University of Illinois                         
#cr                         All Rights Reserved                           
#cr                                                                       
############################################################################

############################################################################
# RCS INFORMATION:
#
# 	$RCSfile: openURL.tcl,v $
# 	$Author: johns $	$Locker:  $		$State: Exp $
#	$Revision: 1.17 $	$Date: 2011/02/25 04:32:51 $
#
############################################################################
# DESCRIPTION:
#      Point a browser to the given URL, or get a url to a file
#
############################################################################

#  The browser name is retreived from the environment variable
#  VMDHTMLVIEWER.  If it looks like you are running mozilla, and
# it fails, try again with just the command name (this assumes you
# are trying funky signalling between Mozilla browsers.
#  There is a puts and an error because this option will probably
# be used in buttons/menus w/o recourse to printing to the screen

proc vmd_open_url {url} {
  global env
  set callstr [format $env(VMDHTMLVIEWER) $url $url $url $url $url $url]
  # open the URL -- this is fraught with SECURITY HOLES
  puts "Opening $url ..."

  #
  # Use the exectool plugin to run the web browser if possible...
  # 
  if [catch {package require exectool} msg] {
    # no exectool plugin, do it the old fashioned way
    if [catch {eval exec $callstr} ermsg] { ;# do not call in background
      set browser [lindex [split $env(VMDHTMLVIEWER)] 0]
      #special case for mozilla-based browsers
      if {$browser == "mozilla" || $browser == "netscape" || $browser == "firefox"} {
        puts stderr "Calling $env(VMDHTMLVIEWER) failed, trying $browser by itself ..."
          if {[catch {eval "exec $browser $url &"} errmsg]} {
            puts stderr "vmd_open_url failed: $errmsg"
            error "vmd_open_url failed: $errmsg"
          }
          return
      }
      puts stderr "vmd_open_url failed: $errmsg"
      error "vmd_open_url failed: $errmsg"
    }
  } else {
    # Use the exectool plugin's ::ExecTool::exec method
    if [catch {eval ::ExecTool::exec $callstr} ermsg] { ;# do not call in background
      set browser [lindex [split $env(VMDHTMLVIEWER)] 0]
      #special case for mozilla-based browsers
      if {$browser == "mozilla" || $browser == "netscape" || $browser == "firefox"} {
        puts stderr "Calling $env(VMDHTMLVIEWER) failed, trying $browser by itself ..."
          if {[catch {eval "::ExecTool::exec $browser $url &"} errmsg]} {
            puts stderr "vmd_open_url failed: $errmsg"
            error "vmd_open_url failed: $errmsg"
          }
          return
      }
      puts stderr "vmd_open_url failed: $errmsg"
      error "vmd_open_url failed: $errmsg"
    }
  }
}


#  Implement the help system here as a simple lookup from topic
#  to URL.
set vmd_help_basepage [string trimright [vmdinfo www] /]
set vmd_help_versionnum [vmdinfo version]

array set vmd_help_lookup "
    homepage    {$vmd_help_basepage/}
    quickhelp   {$vmd_help_basepage/vmd_help.html}
    userguide   {$vmd_help_basepage/vmd-$vmd_help_versionnum/ug/ug.html}
    tutorial    {$vmd_help_basepage/vmd-$vmd_help_versionnum/docs.html#tutorials}
    faq         {$vmd_help_basepage/allversions/vmd_faq.html}
    maillist    {$vmd_help_basepage/mailing_list/}
    software    {$vmd_help_basepage/allversions/related_programs.html}
    scripts     {$vmd_help_basepage/script_library/}
    plugins     {$vmd_help_basepage/plugins/}
    biocore     {http://www.ks.uiuc.edu/Research/biocore/}
    namd        {http://www.ks.uiuc.edu/Research/namd/}
    tcl         {http://www.tcl.tk/}
    python      {http://www.python.org/}
    msms        {http://www.scripps.edu/pub/olson-web/people/sanner/html/msms_home.html}
    babel       {http://www.eyesopen.com/babel/}
    raster3d    {http://www.bmsc.washington.edu/raster3d/}
    povray      {http://www.povray.org/}
    radiance    {http://radsite.lbl.gov/radiance/HOME.html}
    rayshade    {http://www-graphics.stanford.edu/~cek/rayshade/rayshade.html}
    tachyon     {http://www.photonlimited.com/~johns/tachyon/}
    vrml        {http://www.web3d.org/}
"

# The help interface translates the keyword to a URL.  That's all
proc vmd_help { {keyword list} } {
    global vmd_help_lookup
    if {$keyword == "list"} {
	set msg "For help, type help and the name of a topic from the following list:\n" 
	foreach topic [array names vmd_help_lookup] {
          set msg [format "%s%-10s %s\n" $msg $topic $vmd_help_lookup($topic)]
	}
	return $msg
    }
    if {$keyword == "print"} {
	foreach topic [array names vmd_help_lookup] {
	  puts [format "%-10s => %s" $topic $vmd_help_lookup($topic)]
	}
	return
    }
    if {![info exists vmd_help_lookup($keyword)]} {
	# do this since this will be called from pop-up options and
	error "Cannot find a URL for the topic '$keyword':\n\
other options are 'list' and 'print'"
    }
    vmd_open_url $vmd_help_lookup($keyword)
}


