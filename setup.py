
from setuptools import setup

from distutils.util import convert_path
from distutils.command.build import build as DistutilsBuild
from distutils.command.install import install as DistutilsInstall
from distutils.cmd import Command
from subprocess import check_call
import platform
import os
import site
import sys

packages = ['vmd']

###############################################################################

class VMDBuild(DistutilsBuild):
    def initialize_options(self):
        DistutilsBuild.initialize_options(self)

    def finalize_options(self):
        DistutilsBuild.finalize_options(self)

    def run(self):
        # Setup and run compilation script
        #self.build_lib = convert_path(os.path.abspath(self.build_lib) + "/vmd")
        self.execute(self.compile, [], msg="Compiling VMD")
        # Run original build code
        DistutilsBuild.run(self)
        

    def compile(self):
        # Determine target to build
        target = self.get_vmd_build_target()
        srcdir = convert_path(os.path.dirname(os.path.abspath(__file__)) + "/vmd")
        builddir = convert_path(os.path.abspath(self.build_lib) + "/vmd")
        #builddir = convert_path(os.path.abspath(self.build_lib))
        #if not os.path.isdir(builddir): os.makedirs(builddir)
        pydir = convert_path(sys.executable.replace("/bin/python",""))

        # Execute the build
        cmd = [
                srcdir + "/install.sh",
                target,
                builddir,
                pydir,
              ]
        check_call(cmd, cwd=srcdir)

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

#class VMDInstall(DistutilsInstall):
#    def initialize_options(self):
#        DistutilsInstall.initialize_options(self)
#
#    def finalize_options(self):
#        DistutilsInstall.finalize_options(self)
##        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))
#
#    def run(self):
#        # Check if we have built things
#        #print(convert_path(os.path.abspath(self.build_lib) + "/vmd.so"))
#        #quit()
#        #if not os.path.isfile(convert_path(os.path.abspath(self.build_lib) + "/vmd.so")):
#        #    self.run_command('build')
#
#        # Run original install code
#        DistutilsInstall.run(self)
#
###############################################################################

class VMDTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys, subprocess, os
        errno = subprocess.call([sys.executable, os.path.abspath('test/run_tests.py')])
        raise SystemExit(errno)

###############################################################################

setup(name='vmd',
      version='1.9.2a1',
      description='Visual Molecular Dynamics Python module',
      author='Robin Betz',
      author_email='robin@robinbetz.com',
      url='http://github.com/Eigenstate/vmd-python',
      license='VMD License',
      zip_safe=False,
      #setup_requires=['netcdf', 'numpy'],
      #install_requires=["netcdf"],
      extras_require = { 'hoomdplugin': ["expat"] },

      packages=['vmd'],
      package_data = { 'vmd' : ['vmd.so']},
      cmdclass={
          'build': VMDBuild,
          'test': VMDTest,
      },
)

