~~Installation~~ Upgrade instructions
========================================

The official distribution of VMD may contain an outdated version of
Plumed-GUI, so upgrading to one of the releases in
https://github.com/tonigi/vmd_plumed is highly recommended.
To upgrade, please follow the directions in this file.

There are various ways to install, depending on whether you have write
privileges in VMD's program directory (i.e. you are the administrator,
or installed VMD as a user). 

The plugin should appear in the _Extensions > Analysis > Collective
Variable Analysis (PLUMED)_ menu upon VMD's next start. Verify the
version you are running from _Help > About_.



## Method 1: edit VMD's startup file

You may download and extract the plugin in any directory. Then add the
following lines to your `.vmdrc` startup file. Note that name and location
[differ under Windows](http://www.ks.uiuc.edu/Research/vmd/vmd-1.7/ug/node197.html) !

        set auto_path [linsert $auto_path 0 /PATH/TO/EXTRACTED/VMD_PLUMED]
        menu main on


**Note.** If you are using VMD 1.9.2's new preference manager,
`.vmdrc` can't be edited by hand, so this method is not
applicable. Although it may be possible to set `auto_path` in the
*custom* section of the Preferences editor, such commands are executed
*after* the old version of the package is loaded. This is being
investigated.



## Method 2: set the TCLLIBPATH environment variable

This is suitable e.g. if you use *modulefiles*. Note that, unlike
other Unix paths, multiple path components should be space-separated.

        export TCLLIBPATH="/PATH/TO/EXTRACTED/VMD_PLUMED $TCLLIBPATH"



## Method 3: replace the plugin directory

This method is sure-fire if you have write access to the directory where
VMD is installed. Just replace the _plugins/noarch/tcl/plumedX.Y_
directory in VMD's installation with the archive downloaded from
GitHub.

To identify your VMD installation directory, the easiest is to issue the
following command in VMD's TkConsole:

          puts $env(VMDDIR)


