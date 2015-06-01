
from distutils.core import setup
from distutils.util import convert_path
from distutils.command.install import install as DistutilsInstall
from distutils.command.build import build as DistutilsBuild
from distutils.cmd import Command
from subprocess import check_call
import platform
import os
import site
import sys

packages = ['vmd-python']

###############################################################################

class VMDBuild(DistutilsBuild):
    def initialize_options(self):
        DistutilsBuild.initialize_options(self)

    def run(self):
        # Setup and run compilation script
        self.mkpath(self.build_lib)
        self.execute(self.compile, [], msg="Compiling VMD")
        
        # Run original build code
        DistutilsBuild.run(self)

    def compile(self):
        # Determine target to build
        target = self.get_vmd_build_target()
        srcdir = convert_path(os.path.dirname(os.path.abspath(__file__)))
        builddir = convert_path(os.path.abspath(self.build_lib))
        pydir = convert_path(sys.executable.replace("/bin/python",""))
        instdir = convert_path(site.getsitepackages()[0] + "/vmd")

        # Execute the build
        cmd = [
                srcdir + "/vmd/install.sh",
                target,
                builddir,
                pydir,
                instdir
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

class VMDInstall(DistutilsInstall):
    def initialize_options(self):
        DistutilsInstall.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        DistutilsInstall.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # Run original install code
        DistutilsInstall.run(self)

        # Copy all built files
        print("Copying %s to %s" % (self.build_lib, self.install_lib))
        instdir = convert_path(self.install_lib+ "/vmd")
        self.copy_tree(self.build_lib, instdir)


###############################################################################

class VMDTest(Command):
    description = "Runs VMD tests"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass

    def run(self):
        import tests

###############################################################################

setup(name='vmd-python',
      version='1.9.2',
      description='Visual Molecular Dynamics Python module',
      author='Robin Betz',
      author_email='robin@robinbetz.com',
      url='http://github.com/Eigenstate/vmd-python',
      license='VMD License',
      packages=packages,
      package_data = { 'vmd' : ['vmd.so']},
      cmdclass={
          'build': VMDBuild,
          'install': VMDInstall,
          'test': VMDTest,
      },
)


