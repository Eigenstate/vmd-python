Cross-compiling PLUMED under WIN32
==================================

These instructions are only necessary if you are unable to download
the pre-built Windows binaries.

First, download the MINGW build system for windows or your linux
distribution. Under Fedora, for example, packages are called
_mingw32-gcc-*_.

Plumed 2.0 (autoconf)
---------------------

It should be as easy as:

	./configure --host=i686-w64-mingw32  --disable-shared LDFLAGS="-static -s"
	make -j8
	ln -s src/lib/plumed plumed.exe



Plumed 1.3
----------

**Step 1.** Download PLUMED 1.3, go to the *driver* directory, and modify the Makefile adding the following lines (adapt to your system) 

	ifeq ($(arch),gfortran-mingw)
	    F90 = i686-w64-mingw32-gfortran -O3 -fno-second-underscore  
	    CC  = i686-w64-mingw32-gcc -O3 -DDRIVER 
	    CXX = i686-w64-mingw32-g++ -O3 -DDRIVER -DDEBUG
	    LINK = i686-w64-mingw32-gfortran -static
	    LIBS = 

	driver.exe: driver
		cp $< $@

	# Add missing drand48 on w32
	restraint_spath.o: restraint_spath.c
		$(CC) $(CFLAGS) '-Ddrand48(X)=((double)rand()/RAND_MAX)' -c  -o $@ $< 

	endif

**Step 2.** Issue  `make arch=gfortran-mingw` . 

