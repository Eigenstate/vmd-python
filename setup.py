
from setuptools import setup

from distutils.util import convert_path
from distutils.command.build import build as DistutilsBuild
from distutils.cmd import Command
from subprocess import check_call
from glob import glob
import platform
import os
import sys
import re

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
        builddir = convert_path(self.build_lib + "/vmd")
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

    def set_environment_variables(self, pydir):
        os.environ["LD_LIBRARY_PATH"] = "%s:%s" % (os.path.join(pydir, "lib"),
                                                   os.environ.get("LD_LIBRARY_PATH", ""))
        os.environ["NETCDFLIB"] = "-L%s" % os.path.join(pydir, "lib")
        os.environ["NETCDFINC"] = "-I%s" % os.path.join(pydir, "include")

        # Get netcdfldflags from settings file
        pattern = re.compile(r"Extra libraries:\s(.*)")
        with open(os.path.join(pydir, "lib", "libnetcdf.settings")) as fn:
            lines = "\n".join(fn.readlines())
            matches = re.findall(pattern, lines)
        os.environ["NETCDFLDFLAGS"] = "-lnetcdf %s" % matches[0] if len(matches) else ""
        os.environ["NETCDFDYNAMIC"] = "1"

        os.environ["TCLLIB"] = "-L%s" % os.path.join(pydir, "lib")
        os.environ["TCLINC"] = "-I%s" % os.path.join(pydir, "include")
        os.environ["TCLLDFLAGS"] = "-ltcl"

        os.environ["SQLITELIB"] = "-L%s" % os.path.join(pydir, "lib")
        os.environ["SQLITEINC"] = "-I%s" % os.path.join(pydir, "include")
        os.environ["SQLITELDFLAGS"] = "-lsqlite3"

        os.environ["EXPATLIB"] = "-L%s" % os.path.join(pydir, "lib")
        os.environ["EXPATINC"] = "-I%s" % os.path.join(pydir, "include")
        os.environ["EXPATLDFLAGS"] = "-lexpat"

        # Ask numpy where it is
        import numpy
        os.environ["NUMPY_INCLUDE_DIR"] = numpy.get_include()
        os.environ["NUMPY_LIBRARY_DIR"] = numpy.get_include().replace("include","lib")

        from distutils import sysconfig
        os.environ["TCL_LIBRARY_DIR"] = "%s" % os.path.join(pydir, "lib")
        os.environ["TCL_INCLUDE_DIR"] = "%s" % os.path.join(pydir, "include")
        os.environ["PYTHON_LIBRARY_DIR"] = sysconfig.get_python_lib()
        os.environ["PYTHON_INCLUDE_DIR"] = sysconfig.get_python_inc()

        # Get python linker flag
        libs = glob(os.path.join(pydir, "lib",
                                 "libpython%s*.so" % sysconfig.get_python_version()))
        libs = sorted(libs, key=lambda x: len(x))
        pythonldflag = os.path.split(libs[-1])[-1].replace("lib", "-l").replace(".so", "")
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
                target = "MACOSX86_64"
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
      version='2.0.0',
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
