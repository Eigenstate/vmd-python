
from distutils.core import setup
from distutils.util import convert_path
from distutils.command.build import build as DistutilsBuild
from distutils.cmd import Command
from subprocess import check_call, check_output
from glob import glob
import platform
import os
import sys

packages = ['vmd']

###############################################################################

class VMDBuild(DistutilsBuild):

    def initialize_options(self):
        DistutilsBuild.initialize_options(self)

    #==========================================================================

    def finalize_options(self):
        DistutilsBuild.finalize_options(self)

    #==========================================================================

    def run(self):
        # Setup and run compilation script
        self.execute(self.compile, [], msg="Compiling VMD")
        # Run original build code
        DistutilsBuild.run(self)

    #==========================================================================

    def compile(self):
        # Determine target to build
        target = self.get_vmd_build_target()
        srcdir = convert_path(os.path.dirname(os.path.abspath(__file__)) + "/vmd")
        builddir = convert_path(os.path.abspath(self.build_lib + "/vmd"))
        pydir = convert_path(sys.executable.replace("bin/python", ""))

        self.set_environment_variables(pydir)

        # Execute the build
        cmd = [
            os.path.join(srcdir, "install.sh"),
            target,
            builddir,
            pydir,
        ]
        check_call(cmd, cwd=srcdir)

    #==========================================================================

    def _find_include_dir(self, incfile, pydir):
        """
        Finds the path containing an include file. Starts by searching
        $INCLUDE, then whatever system include paths gcc looks in.
        If it can't find the file, defaults to "$pydir/include"
        """

        # Look in directories specified by $INCLUDE
        searchdirs = [d for d in os.environ.get("INCLUDE", "").split(":")
                      if os.path.isdir(d)]
        # Also look in the directories gcc does
        try:
            out = check_output("echo | gcc -E -Wp,-v - 2>&1 | grep '^\s.*'",
                               shell=True)
            out = out.decode("utf-8").strip().split("\n")
            searchdirs.extend([d.strip() for d in out if os.path.isdir(d.strip())])
        except: pass

        # Find the actual file
        out = b""
        try:
            out = check_output(["find", "-H"]
                                + searchdirs
                                + ["-maxdepth", "1",
                                   "-name", incfile],
                               close_fds=True,
                               stderr=open(os.devnull, 'wb'))
        except: pass

        incdir = os.path.split(out.decode("utf-8").split("\n")[0])[0]
        if not glob(os.path.join(incdir, incfile)): # Glob allows fildcards
            incdir = os.path.join(pydir, "include", incfile)
            print("\nWARNING: Could not find include file '%s' in standard "
                  "include directories.\n Defaulting to: '%s'"
                  % (incfile, incdir))
        print("   INC: %s -> %s" % (incfile, incdir))
        return incdir

    #==========================================================================

    def _find_library_dir(self, libfile, pydir, fallback=True):
        """
        Finds the directory containing a library file. Starts by searching
        $LD_LIBRARY_PATH, then ld.so.conf system paths used by gcc.
        """

        # Look in directories specified by $LD_LIBRARY_PATH
        out = b""
        if "Darwin" in platform.system():
            libfile = "%s.dylib" % libfile
            searchdirs = [d for d in os.environ.get("DYLD_LIBRARY_PATH",
                                                    "").split(":")
                          if os.path.isdir(d)]
        else:
            libfile = "%s.so" % libfile
            searchdirs = [d for d in os.environ.get("LD_LIBRARY_PATH",
                                                    "").split(":")
                          if os.path.isdir(d)]
        try:
            out = check_output(["find", "-H"]
                               + searchdirs
                               + ["-maxdepth", "1",
                                  "-name", libfile],
                               close_fds=True,
                               stderr=open(os.devnull, 'wb'))
        except: pass

        libdir = os.path.split(out.decode("utf-8").split("\n")[0])[0]
        if glob(os.path.join(libdir, libfile)): # Glob allows wildcards
            print("   LIB: %s -> %s" % (libfile, libdir))
            return libdir

        # See if the linker can find it, alternatively
        # This only works on Linux
        if "Linux" in platform.system():
            try:
                out = check_output("ldconfig -p | grep %s$" % libfile, shell=True)
            except: pass
            libdir = os.path.split(out.decode("utf-8").split(" ")[-1])[0]

        # For OSX, use defaults + DYLD_FALLBACK_LIBRARY_PATH
        elif "Darwin" in platform.system():
            searchdirs = [os.path.join(os.environ.get("HOME", ""), "lib"),
                          os.path.join("usr", "local", "lib"),
                          os.path.join("usr", "lib")] \
                + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
            try:
                out = check_output(["find", "-H"]
                                   + [d for d in set(searchdirs) if os.path.isdir(d)]
                                   + ["-maxdepth", "1",
                                      "-name", libfile],
                                   close_fds=True,
                                   stderr=open(os.devnull, 'wb'))
            except: pass
            libdir = os.path.split(out.decode("utf-8").split("\n")[0])[0]
            if glob(os.path.join(libdir, libfile)):
                print("   LIB: %s -> %s" % (libfile, libdir))
                return libdir
        else:
            libdir = ""

        if not glob(os.path.join(libdir, libfile)):
            libdir = os.path.join(pydir, "lib")
            if not fallback:
                return None
            print("WARNING: Could not find library file '%s' in standard "
                  "library directories.\n Defaulting to: '%s'"
                  % (libfile, os.path.join(libdir, libfile)))
        print("   LIB: %s -> %s" % (libfile, libdir))
        return libdir

    #==========================================================================

    def set_environment_variables(self, pydir):
        print("Finding libraries...")
        osys = platform.system()
        if "Linux" in osys or "Windows" in osys:
            os.environ["LD_LIBRARY_PATH"] = "%s:%s" % (os.path.join(pydir, "lib"),
                                                       os.environ.get("LD_LIBRARY_PATH", ""))
        elif "Darwin" in osys:
            os.environ["DYLD_LIBRARY_PATH"] = "%s:%s" % (os.path.join(pydir, "lib"),
                                                         os.environ.get("DYLD_LIBRARY_PATH", ""))
        os.environ["INCLUDE"] = "%s:%s"  % (os.path.join(pydir, "include"),
                                               os.environ.get("INCLUDE", ""))

        # No reliable way to ask for actual available library, so try 8.5 first
        os.environ["TCL_LIBRARY_DIR"] = self._find_library_dir("libtcl8.5", pydir,
                                                               fallback=False)
        os.environ["TCLLDFLAGS"] = "-ltcl8.5"
        if os.environ["TCL_LIBRARY_DIR"] is None:
            print("  Newer libtcl8.6 used")
            os.environ["TCL_LIBRARY_DIR"] = self._find_library_dir("libtcl8.6", pydir)
            os.environ["TCLLDFLAGS"] = "-ltcl8.6"

        os.environ["TCL_INCLUDE_DIR"] = self._find_include_dir("tcl.h", pydir)
        os.environ["TCLLIB"] = "-L%s" % os.environ["TCL_LIBRARY_DIR"]
        os.environ["TCLINC"] = "-I%s" % os.environ["TCL_INCLUDE_DIR"]

        # Sqlite (for dmsplugin)
        os.environ["SQLITELIB"] = "-L%s" % self._find_library_dir("libsqlite3", pydir)
        os.environ["SQLITEINC"] = "-I%s" % self._find_include_dir("sqlite3.h", pydir)
        os.environ["SQLITELDFLAGS"] = "-lsqlite3"

        # Expat (for hoomdplugin)
        os.environ["EXPATLIB"] = "-L%s" % self._find_library_dir("libexpat", pydir)
        os.environ["EXPATINC"] = "-I%s" % self._find_include_dir("expat.h", pydir)
        os.environ["EXPATLDFLAGS"] = "-lexpat"

        # Netcdf (for netcdfplugin)
        os.environ["NETCDFLIB"] = "-L%s" % self._find_library_dir("libnetcdf", pydir)
        os.environ["NETCDFINC"] = "-I%s" % self._find_include_dir("netcdf.h", pydir)
        os.environ["NETCDFLDFLAGS"] = "-lnetcdf"
        os.environ["NETCDFDYNAMIC"] = "1"

        # Ask numpy where it is
        import numpy
        os.environ["NUMPY_INCLUDE_DIR"] = numpy.get_include()
        os.environ["NUMPY_LIBRARY_DIR"] = numpy.get_include().replace("include","lib")

        from distutils import sysconfig
        os.environ["PYTHON_LIBRARY_DIR"] = sysconfig.get_python_lib()
        os.environ["PYTHON_INCLUDE_DIR"] = sysconfig.get_python_inc()

        # Get python linker flag, as it may be -l3.6m or -l3.6 depending on malloc
        # this python was built against
        pylibname = "libpython%s*" % sysconfig.get_python_version()
        libs = glob(os.path.join(self._find_library_dir(pylibname, pydir), pylibname))
        libs = sorted(libs, key=lambda x: len(x))

        pythonldflag = "-l%s" % os.path.split(libs[-1])[-1][3:] # Remove "lib"
        if "Darwin" in osys:
            pythonldflag = pythonldflag.partition(".dylib")[0]
        else:
            pythonldflag = pythonldflag.partition(".so")[0]

        os.environ["VMDEXTRALIBS"] = " ".join([os.environ["SQLITELDFLAGS"],
                                               os.environ["EXPATLDFLAGS"],
                                               pythonldflag])

    #==========================================================================

    def get_vmd_build_target(self):
        osys = platform.system()
        mach = platform.machine()

        if "Linux" in osys:
            if "x86_64" in mach:
                target = "LINUXAMD64"
            else:
                target = "LINUX"
        elif "Darwin" in osys:
            if "x86_64" in mach:
                target = "MACOSXX86_64"
            elif "PowerPC" in mach:
                target = "MACOSX"
            else:
                target = "MACOSXX86"
        elif "Windows" in osys:
            if "64" in mach:
                target = "WIN64"
            else:
                target = "WIN32"
        else:
            raise ValueError("Unsupported os '%s' and machine '%s'" % (osys, mach))

        return target

###############################################################################

class VMDTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import subprocess, os
        errno = subprocess.call(["py.test", os.path.abspath(os.path.join("test",
                                                                         "test_vmd.py"))])
        raise SystemExit(errno)

###############################################################################

setup(name='vmd-python',
      version='2.0.5',
      description='Visual Molecular Dynamics Python module',
      author='Robin Betz',
      author_email='robin@robinbetz.com',
      url='http://github.com/Eigenstate/vmd-python',
      license='VMD License',
      zip_safe=False,
      extras_require={'hoomdplugin': ["expat"]},

      packages=['vmd'],
      package_data={'vmd' : ['libvmd.so']},
      cmdclass={
          'build': VMDBuild,
          'test': VMDTest,
      },
     )
