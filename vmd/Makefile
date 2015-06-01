##
## Top level "makefile" for VMD.
## $Id: Makefile,v 1.106 2015/05/20 22:42:10 johns Exp $
##
##
## This makefile just presents the user with a simple list of 
## target platforms that the makefile knows about.  The makefile
## calls the VMD configure script to generate appropriate files
## and do the real work.  This makefile is mainly intended to reduce
## the number of config files, and organize all of the platform settings
## in a single file for easy editing and eventually for site specific
## configuration.
##
default:
	@echo "VMD Build Options"
	@echo "-----------------"
	@echo "  Hardware accelerated versions:                   "
	@echo "      aix4.opengl       - AIX 4.3                    "
	@echo "      aix5.opengl       - AIX 5                      "
	@echo "   aix5.64.opengl       - AIX 5, 64-bit              "
	@echo "   aix5.64.text         - AIX 6, 64-bit, text-only   "
	@echo "   aix6.64.text         - AIX 6, 64-bit, text-only   "
	@echo "    hpux11.opengl       - HP-UX 11.0                 "
	@echo "     irix6.opengl       - IRIX 6.5.x                 "
	@echo "  irix6.64.opengl       - IRIX 6.5.x, 64-bit         "
	@echo "     linux.opengl       - Redhat Linux (32-bit)      "
	@echo "     linux.cuda.opengl  - Redhat Linux (32-bit), NVIDIA CUDA"
	@echo "     linux.amd64.opengl - Redhat Linux (64-bit)      "
	@echo "     linux.ia64.opengl  - Redhat Linux Itanium EL3 (64-bit)  "
	@echo "    macosx.opengl       - MacOS X 10.2               "
	@echo "    macosx.opengl.x11   - MacOS X 10.3               "
	@echo "   solaris.opengl       - Solaris 9                  "
	@echo "solaris.64.opengl       - Solaris 9, 64-bit          "
	@echo "     tru64.opengl       - Tru64 Unix 4.0F            "
	@echo ""
	@echo "  Non-accelerated versions:                    "
	@echo "        aix4.mesa - AIX 4.3                    "
	@echo "     freebsd.mesa - FreeBSD 5.x                "
	@echo "      hpux11.mesa - HP-UX 11.0                 "
	@echo "       linux.mesa - Redhat Linux 7.x           "
	@echo "  linuxalpha.mesa - Linux/Alpha                "
	@echo "    linuxppc.mesa - Linux/PowerPC              "
	@echo "      macosx.mesa - MacOS X 10.1               "
	@echo "     solaris.mesa - Solaris 8                  "
	@echo "       tru64.mesa - Tru64 Unix 4.0F            "
	@echo "-----------------------------------------------"
	@echo "Type 'make platform' to configure VMD for any of   "
	@echo "the target platforms listed above.                 " 
	@echo ""
	@echo "Once VMD is configured for compilation, do this:   "
	@echo "  cd src; make veryclean; make                     "
	@echo ""

##
## catch people that don't read the README closely enough
##
install:
	@echo "Please run 'make install' from within the 'src' directory"
	@echo "e.g.:  cd src; make install"

##
## AIX 4.x and 5.x builds
##
aix4.mesa:
	./configure AIX4 MESA FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON NETCDF

aix4.opengl:
	./configure AIX4 OPENGL FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON NETCDF

aix5.opengl:
	./configure AIX5 OPENGL FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON NETCDF

aix5.text:
	./configure AIX5 IMD SILENT TCL PTHREADS LIBTACHYON

aix5.64.opengl:
	./configure AIX5_64 OPENGL FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON NETCDF LP64

aix5.64.text:
	./configure AIX5_64 IMD SILENT TCL PTHREADS LIBTACHYON LP64

aix6.64.text:
	./configure AIX6_64 IMD SILENT TCL PTHREADS LIBTACHYON LP64

##
## Android builds
##
android.armv7a.min:
	./configure ANDROIDARMV7A IMD PTHREADS SILENT

android.armv7a.text:
	./configure ANDROIDARMV7A IMD TCL PTHREADS LIBTACHYON SILENT

##
## FreeBSD builds
##
freebsd.mesa:
	./configure FREEBSD MESA FLTK TK TCL IMD VRPN PTHREADS LIBTACHYON NETCDF SILENT

freebsd.opengl:
	./configure FREEBSD OPENGL FLTK TK IMD VRPN SILENT LIBSBALL TCL PTHREADS ACTC LIBTACHYON NETCDF



##
## HP-UX 11.x builds
##
hpux11.mesa:
	./configure HPUX11 MESA FLTK TK IMD ACC SILENT TCL PTHREADS LIBTACHYON NETCDF LIBSBALL

hpux11.opengl:
	./configure HPUX11 OPENGL FLTK TK IMD ACC SILENT TCL PTHREADS LIBTACHYON NETCDF LIBSBALL


##
## IRIX 6.5.x builds
##
irix6.opengl:
	./configure IRIX6 CAVE OPENGL FLTK TK IMD SILENT LIBSBALL TCL PTHREADS LIBTACHYON ACTC NETCDF

irix6.64.opengl:
	./configure IRIX6_64 CAVE OPENGL FLTK TK IMD SILENT LIBSBALL TCL PTHREADS LIBTACHYON ACTC NETCDF LP64


##
## Linux builds
##
linux.carma.text:
	./configure LINUXCARMA IMD TCL PTHREADS CUDA SILENT

linux.carma.opengl:
	./configure LINUXCARMA OPENGL FLTK TK IMD TCL PTHREADS CUDA ACTC LIBTACHYON SILENT

linux.mesa:
	./configure LINUX MESA FLTK TK IMD VRPN SILENT TCL PTHREADS ACTC LIBTACHYON

linux.opengl:
	./configure LINUX ICC OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF CUDA XINERAMA

linux.opengl.sharedlib:
	./configure LINUX OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF CUDA SHARED

linux.icc.opengl:
	./configure LINUX OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF ICC CUDA


linux.cuda.opengl:
	./configure LINUX OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF CUDA

linux.amd64.opengl:
	./configure LINUXAMD64 ICC OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON LIBOPTIX NETCDF PYTHON NUMPY CUDA XINERAMA

linux.iccamd64.opengl:
	./configure LINUXAMD64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF ICC CUDA

linux.amd64.opengl.sharedlib:
	./configure LINUXAMD64 COLVARS IMD TCL PYTHON PTHREADS NUMPY SILENT SHARED

linux.amd64.mpi:
	./configure LINUXAMD64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF PYTHON NUMPY CUDA XINERAMA MPI

linux.cuda64.opengl:
	./configure LINUXAMD64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL XINPUT TCL PTHREADS ACTC LIBTACHYON NETCDF PYTHON NUMPY CUDA

linux.ia64.opengl:
	./configure LINUXIA64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD SILENT ACTC NETCDF TCL PTHREADS

linuxalpha.mesa:
	./configure LINUXALPHA MESA FLTK TK IMD LIBTACHYON NETCDF SILENT TCL

linuxppc.mesa:
	./configure LINUXPPC MESA FLTK TK IMD LIBTACHYON NETCDF SILENT TCL

##
## Special Linux build for clustermatic/scyld, or plain text mode usage
##
linux.text:
	./configure LINUX COLVARS IMD SILENT TCL PTHREADS ACTC LIBTACHYON NETCDF

linux.amd64.text:
	./configure LINUXAMD64 COLVARS IMD SILENT TCL PTHREADS ACTC LIBTACHYON NETCDF

##
## Indiana BigRed cluster build
##
linuxppc64.text:
	./configure LINUXPPC64 IMD SILENT TCL PTHREADS 

##
## NCSA Blue Waters supercomputer builds
##
bluewaters.text:
	./configure BLUEWATERS OPENGLPBUFFER CUDA COLVARS IMD SILENT TCL ACTC LIBTACHYON PTHREADS 

bluewaters.mpi:
	./configure BLUEWATERS OPENGLPBUFFER CUDA COLVARS IMD SILENT TCL ACTC LIBOPTIX LIBTACHYON PTHREADS MPI

##
## MacOS 10.x on PowerPC builds
##
macosx.mesa:
	./configure MACOSX MESA FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF

macosx.opengl.x11:
	./configure MACOSX OPENGL FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF

# MacOS X 10.3 builds, no python etc
#macosx.opengl:
#	./configure MACOSX FLTKOPENGL FLTK IMD VRPN TK TCL SILENT LIBSBALL TDCONNEXION PTHREADS LIBTACHYON ACTC NETCDF
# TDCONNEXION remains off until the code is fully tested
macosx.opengl:
	./configure MACOSX FLTKOPENGL FLTK IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF


# Apple's Python framework can be used for 10.4 based PowerPC builds
#macosx.opengl.104:
#macosx.opengl:
#	./configure MACOSX FLTKOPENGL FLTK IMD TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF

##
## MacOS 10.x on x86 builds
##
#macosx.x86.opengl:
#	./configure MACOSXX86 FLTKOPENGL FLTK IMD VRPN TK TCL SILENT LIBSBALL TDCONNEXION PTHREADS LIBTACHYON ACTC NETCDF
# TDCONNEXION remains off until the code is fully tested, CUDA is on by default
# for 10.5.x builds
macosx.x86.opengl:
	./configure MACOSXX86 FLTKOPENGL FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF

macosx.x86.opengl.cuda:
	./configure MACOSXX86 FLTKOPENGL FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF CUDA

macosx.x86.opengl.nocuda:
	./configure MACOSXX86 FLTKOPENGL FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF

macosx.x86.text:
	./configure MACOSXX86 COLVARS IMD TCL SILENT PTHREADS LIBTACHYON ACTC NETCDF

macosx.x86.cuda.opengl:
	./configure MACOSXX86 FLTKOPENGL FLTK COLVARS IMD VRPN TK TCL SILENT LIBSBALL PTHREADS LIBTACHYON ACTC NETCDF CUDA

macosx.x86_64.opengl:
	./configure MACOSXX86_64 FLTKOPENGL FLTK COLVARS IMD TK TCL SILENT PTHREADS LIBTACHYON ACTC LP64 OPENCL CUDA

macosx.x86.64.text:
	./configure MACOSXX86_64 COLVARS IMD TCL SILENT PTHREADS LP64

##
## Solaris 9 biulds
##
solaris.mesa:
	./configure SOLARIS2 MESA FLTK TK COLVARS IMD SILENT TCL PTHREADS ACTC LIBTACHYON NETCDF

solaris.opengl:
	./configure SOLARIS2 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL TCL PTHREADS ACTC LIBTACHYON NETCDF

solaris.64.opengl:
	./configure SOLARIS2_64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD VRPN SILENT LIBSBALL TCL PTHREADS ACTC LIBTACHYON NETCDF LP64

solarisx86.opengl:
	./configure SOLARISX86 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD SILENT TCL PTHREADS ACTC

solarisx86.64.opengl:
	./configure SOLARISX86_64 OPENGL OPENGLPBUFFER FLTK TK COLVARS IMD SILENT TCL LIBTACHYON PTHREADS ACTC LP64

solarisx86.text:
	./configure SOLARISX86 COLVARS IMD SILENT TCL PTHREADS NETCDF

solarisx86.64.text:
	./configure SOLARISX86 COLVARS IMD SILENT TCL PTHREADS NETCDF LP64


##
## Tru64 4.x builds
##
tru64.mesa:
	./configure TRU64 MESA FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON LIBSBALL NETCDF

tru64.opengl:
	./configure TRU64 OPENGL FLTK TK IMD SILENT TCL PTHREADS LIBTACHYON LIBSBALL NETCDF


##
## Windows (32-bit)
##
win32.opengl:
	./configure WIN32 MSVC TCL 

##
## Windows (64-bit)
##
win64.opengl:
	./configure WIN64 MSVC TCL 


##
## Items below this point are not intended for use outside of UIUC, they
## are just here to make it easy for new VMD hackers to get a build tree
## going in our local environment. 
##
links :
	@echo "Making links to TCB-UIUC VMD libraries / headers etc..."
	@echo "Setting up links for link libraries..."
	@if [ -d lib ] ; then \
		mv lib lib.old; \
	fi;
	mkdir lib
	( 	cd lib; \
		MYCWD=`pwd`; \
		cd /Projects/vmd/vmd/lib; \
		for i in *;  \
		do  \
			ln -s /Projects/vmd/vmd/lib/$$i $$MYCWD/$$i; \
		done; \
	) 
	@echo "Setting up links for MSVC dynamic link libraries..."
	ln -s ../lib msvc/lib
	ln -s lib/msvclibs msvc/libs
	@echo "Setting up links for MSVC scripts..."
	ln -s lib/scripts msvc/scripts
	ln -s data/.vmdrc msvc/vmd.rc
	@echo "Setting up links for documentation pictures..."
	ln -s /Projects/vmd/vmd/pictures doc/pictures/localpics
	@echo "Setting up links for plugins..."
	ln -s /Projects/vmd/vmd/plugins  plugins

