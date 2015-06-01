Prefix: /usr/local
Summary: Molecular dynamics visualization software
Name: vmd
Version: 1.2
Release: 1
Icon: vmd.gif
Copyright: distributable
Source: ftp://ftp.ks.uiuc.edu/pub/mdscope/vmd/vmd-1.2b4.bin.LINUX.tar.gz
Group: Applications/Graphics
ExcludeArch: axp sparc

%description
This program allows the user to visualize molecular dynamics simulations,
using 3-D rendering via OpenGL or Mesa.

%prep
echo "Nothing to prepare"

%build
echo "No need to rebuild binary VMD distribution"

%install
echo "Make install me"

%files
/usr/local/bin/vmd
/usr/local/lib/vmd

%pre
echo ""
echo ""
echo ""
echo "VMD - Visual Molecular Dynamics"
echo "A. Dalke, W. Humphrey, S. Izrailev, J. Stone, J. Ulrich"
echo ""
echo "Copyright 1998, The Board of Trustees of the University of Illinois"
echo "All Rights Reserved"
echo ""
echo "http://www.ks.uiuc.edu/Research/vmd/"
echo ""
echo "Please send questions to vmd@ks.uiuc.edu"
echo ""
echo "VMD files will be installed in:"
echo "   $RPM_INSTALL_PREFIX/bin"
echo "   $RPM_INSTALL_PREFIX/lib/vmd"
echo ""
echo "Installing VMD 1.2 Beta 4....."

%post
mv $RPM_INSTALL_PREFIX/bin/vmd /tmp/vmd.orig
sed -e "s|/usr/local|$RPM_INSTALL_PREFIX|g" < /tmp/vmd.orig > $RPM_INSTALL_PREFIX/bin/vmd
rm /tmp/vmd.orig
chmod 555 $RPM_INSTALL_PREFIX/bin/vmd

echo "VMD installation complete!"
echo ""
echo "Displaying the VMD license agreement now:"
echo "       $RPM_INSTALL_PREFIX/lib/vmd/License.txt"
echo ""
echo "By using this software you agree to the terms of the VMD"
echo "license.  Please register your copy of VMD using the form"
echo "included at the end of the license."
echo ""

/bin/more $RPM_INSTALL_PREFIX/lib/vmd/License.txt

echo "" 

%preun 
echo "Uninstalling VMD"

%postun
echo "VMD uninstalled!"


