Prerequisite: PLUMED engine
========================================

Plumed-GUI **requires** a _driver_ executable for your architecture to
be located somewhere in the executable path.  The executables are
named _plumed_ (PLUMED 2.1, recommended) and _driver_ (PLUMED 1.3).
You can have either or both installed.

 * **Linux/Unix** and **OSX**: download the code from the [PLUMED home
     page](http://www.plumed.org) and build it according to the
     instructions.

 * **Windows**: Plumed-GUI provides a _Help > Attempt
   download of prebuilt Windows driver binaries_.  menu entry which
   attempts to get precompiled binaries and to install them in a
   (system specific) directory. The same directory is temporarily
   added to the search path. The method requires enough permissions
   and network access.


If the executables are correctly installed, their location will appear
in the "Path to executable" box; the path can be adjusted manually
(not recommended).

If the Windows-specific automated installation above fails, try to
manually download the pre-compiled binaries for Win32
([driver.exe](http://www.multiscalelab.org/utilities/PlumedGUI?action=AttachFile&do=get&target=driver.exe)
and
[plumed.exe](http://www.multiscalelab.org/utilities/PlumedGUI?action=AttachFile&do=get&target=plumed.exe))
and copy them e.g. in `c:\windows` or in VMD's directory.  If all
else fails, you can try to cross-compile yourself, as shown in the
[PLUMED-WIN32.md](PLUMED-WIN32.md) file.

