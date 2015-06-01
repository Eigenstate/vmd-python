Summary: The VMD Molecular Visualization and Analysis Package
Vendor:  The Theoretical and Computational Biophysics Group (TCBG), University of Illinois at Urbana-Champaign
Name: vmd
Version: 1.8.3a28
Release: 8ak
BuildArch: i386
Exclusivearch: i386
URL: http://www.ks.uiuc.edu/Research/vmd/
Copyright: Free to use but Restricted
Group: Applications/Science
Source0: vmd-%{version}.bin.LINUX.opengl.tar.gz
Source1: vmd.desktop
Source2: vmd.xpm
Source3: vmd_16x16.png
Source4: vmd_22x22.png
Source5: vmd_32x32.png
Source6: vmd_48x48.png
BuildRoot: /var/tmp/%{name}-%{version}
# NOTE: the automatic requires/provides detection on many distribution 
# has problems, especially when the NVIDIA GLX libraries are installed.
# the followin needs to be updated whenever libraries on the build host
# are replaced with incompatible versions. AK 2004-11-01.
Autoreqprov: no
Requires: /bin/csh /bin/sh /usr/bin/env xterm
Requires: libc.so.6 libc.so.6(GLIBC_2.0) libc.so.6(GLIBC_2.1) libc.so.6(GLIBC_2.1.3) libc.so.6(GLIBC_2.2) libc.so.6(GLIBC_2.3) libdl.so.2 libdl.so.2(GLIBC_2.0) libdl.so.2(GLIBC_2.1) libgcc_s.so.1 libgcc_s.so.1(GCC_3.0) libgcc_s.so.1(GLIBC_2.0) libGL.so.1 libGLU.so.1 libm.so.6 libm.so.6(GLIBC_2.0) libpthread.so.0 libpthread.so.0(GLIBC_2.0) libpthread.so.0(GLIBC_2.1) libpthread.so.0(GLIBC_2.2) libpthread.so.0(GLIBC_2.3.2) libstdc++.so.5 libstdc++.so.5(CXXABI_1.2) libstdc++.so.5(GLIBCPP_3.2) libutil.so.1 libutil.so.1(GLIBC_2.0) libX11.so.6 libXft.so.2

# keep rpm from messing with the binaries.
%define __os_install_post true

%description
  VMD is designed for the visualization and analysis of biological
systems such as proteins, nucleic acids, lipid bilayer assemblies,
etc.  It may be used to view more general molecules, as VMD can read
standard Protein Data Bank (PDB) files and display the contained
structure.  VMD provides a wide variety of methods for rendering and
coloring a molecule: simple points and lines, CPK spheres and
cylinders, licorice bonds, backbone tubes and ribbons, and others.
VMD can be used to animate and analyze the trajectory of a molecular
dynamics (MD) simulation.  In particular, VMD can act as a graphical
front end for an external MD program by displaying and animating a
molecule undergoing simulation on a remote computer.

%prep
%setup

# fix up configure
mv configure configure.old
sed -e "s,^\(\$install_bin_dir\).*,\1=\"${RPM_BUILD_ROOT}%{_usr}/bin\";," \
    -e "s,^\(\$install_library_dir\).*,\1=\"${RPM_BUILD_ROOT}%{_libdir}/vmd\";," \
    -e "s,^\(\$def_babel_bin\).*,\1=\"/usr/local/bin/babel\";," \
        < configure.old > configure

# fix up the vmd script for those without a huge screen
mv bin/vmd bin/vmd.old
sed -e 's/setenv VMDSCRPOS.*/setenv VMDSCRPOS "300 100"/'               \
    -e 's/setenv VMDSCRSIZE.*/setenv VMDSCRSIZE "540 720"/'             \
        < bin/vmd.old > bin/vmd
chmod 0755 bin/vmd

%build
perl ./configure

%install

rm -rf ${RPM_BUILD_ROOT}
mkdir -p ${RPM_BUILD_ROOT}%{_usr}/bin
mkdir -p ${RPM_BUILD_ROOT}%{_libdir}/vmd
mkdir -p ${RPM_BUILD_ROOT}%{_usr}/share/applications
mkdir -p ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps
make -C src install

# support for desktop environments
cp %{SOURCE1} ${RPM_BUILD_ROOT}%{_usr}/share/applications/
cp %{SOURCE2} ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps/
cp %{SOURCE3} ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps/
cp %{SOURCE4} ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps/
cp %{SOURCE5} ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps/
cp %{SOURCE6} ${RPM_BUILD_ROOT}%{_usr}/share/pixmaps/
        
# remove potential traces from installation to temporary location
sed -e "s,${RPM_BUILD_ROOT},,g" ${RPM_BUILD_ROOT}%{_usr}/bin/vmd \
        > ${RPM_BUILD_ROOT}%{_usr}/bin/vmd.new 
mv ${RPM_BUILD_ROOT}%{_usr}/bin/vmd.new ${RPM_BUILD_ROOT}%{_usr}/bin/vmd

# fix permissions
chmod 0755 ${RPM_BUILD_ROOT}%{_usr}/bin/vmd
chmod -R a+rX,g-w,o-w ${RPM_BUILD_ROOT}/%{_libdir}/vmd \
        ${RPM_BUILD_ROOT}/%{_usr}/share


%clean
rm -rf ${RPM_BUILD_ROOT}

%post


%files
%defattr(-,root,root)
%doc LICENSE README Announcement proteins
%{_usr}/bin
%{_libdir}/vmd
%{_usr}/share/applications
%{_usr}/share/pixmaps

%changelog
* Mon Nov  1 2004 Axel Kohlmeyer <axel.kohlmeyer@theochem.ruhr-uni-bochum.de>
- more updates to integrate VMD with some desktop environments and installation in /usr

* Sun Oct 31 2004 Axel Kohlmeyer <axel.kohlmeyer@rub.de> - 1.8.3a28-7ak
- updated spec file for 1.8.3 release and packaging on SuSE 9.1

* Tue Jun 17 2003 Axel Kohlmeyer <axel.kohlmeyer@nexgo.de>
- upgrade to version 1.8.1

* Mon Aug 26 2002 Axel Kohlmeyer <axel.kohlmeyer@theochem.ruhr-uni-bochum.de>
- upgrade to version 1.8a21

* Tue Oct 16 2001 Axel Kohlmeyer <axel.kohlmeyer@ruhr-uni-bochum.de>
- created spec file from gopenmol template. uses precompiled binaries.





