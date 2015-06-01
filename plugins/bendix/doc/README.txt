Hello!

Thank you for your interest in Bendix. Below follows a 3-step installation for UNIX, Windows and MAC OS. The installation should be very similar for other operating systems.
Basically you need to save all files in a dedicated VMD plugin directory and edit your user-settings to enable Bendix at start-up.

Bendix is a plugin for VMD. This means that it needs VMD to work. If you do not already have VMD, you can download it for free from this website (a short registration process is needed):
http://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD

Bendix has a dedicated website: http://sbcb.bioch.ox.ac.uk/Bendix
Please visit it for tutorials, contact details, Q&A and general info.


::::: Bendix installation notes ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
N.B. I found that I had to log in as Administrator to take control of my Windows7 laptop, or it wouldn't let me alter program files.

1. Move the bendix1.1 folder, with all files, to VMD's dedicated plugin directory
	This is a subdirectory [../plugins/noarch/tcl] in your VMD installation path. 
	The exact path will differ depending on where you chose to install VMD, but examples are:
		LINUX:		/computer/packages/vmd/1.9.1/lib/plugins/noarch/tcl/
		MAC:		/Applications/VMD\ 1.9.1.app/Contents/vmd/plugins/noarch/tcl/
					(to access this path, right-click on the VMD icon and choose "Show Package Contents")
		WINDOWS:	C:\Program_files\University_of_Illinois\VMD\plugins\noarch\tcl\
	

2. Locate (or create) the VMD startup file called vmdrc
	This is a hidden file called .vmdrc in LINUX and MAC OS. If you don't have this file, create it.
		LINUX:		In your home directory, where you end up if you type cd, list all hidden files by ls -ltra
		MAC:		In a VMD subdirectory [../Contents/vmd/.vmdrc], 
					e.g. /Applications/VMD\ 1.9.1.app/Contents/vmd/.vmdrc
					Alternatively in your home directory.
		WINDOWS:	This is a standard (not hidden) file in the VMD main directory.
					e.g. C:\Program_files\University_of_Illinois\VMD\vmd.rc

3. Open the vmdrc file in an editor and add these two lines to the end of the file

	    vmd_install_extension bendix bendix "Analysis/Bendix"
    	menu main on


Start VMD, and Bendix should appear in the VMD Main window under the menu Extensions > Analysis > Bendix


:::: Troubleshoot ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
If installation fails, see the Bendix website http://sbcb.bioch.ox.ac.uk/Bendix/faq.html
Notably, if you're running a version of VMD that is older than version 1.9, this may prohibit Bendix and you should consider a VMD upgrade.

If plugin-installation fails, or you wish to not run bendix as a plugin:
	1. Save the files in a directory of your choice
	2. Open VMD and in the VMD Main window, under the menu Extensions, open Tk Console. 
	3. Type the following into the Tk Console terminal:
			source /dir/where/you/saved/bendix1.1/bendix.tcl
			bendix

However, in non-plugin mode images do not load, so you may with to keep the GUI-tips.gif image open as a reference.
Alternatively keep a browser open with bendix' website: http://sbcb.bioch.ox.ac.uk/Bendix/


Best regards,
Caroline

==========================================================================================
A. Caroline E. Dahl
D.Phil candidate in the Structural Bioinformatics and Computational Biochemistry Unit, University of Oxford

caroline.dahl@dtc.ox.ac.uk
http://sbcb.bioch.ox.ac.uk/currentmembers/dahl.php

