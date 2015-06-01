# Microsoft Developer Studio Project File - Name="winvmd" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

CFG=winvmd - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "winvmd.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "winvmd.mak" CFG="winvmd - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "winvmd - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "winvmd - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE "winvmd - Win32 Spaceball VRPN" (based on "Win32 (x86) Console Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "winvmd - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /c
# ADD CPP /nologo /G5 /MT /W3 /vmg /O2 /I "..\..\..\plugins\include" /I "..\..\lib\fltk\include" /I "..\..\lib\tcl\include" /I "..\..\lib\tk\include" /I "..\..\lib\tclx\include" /I "..\..\src" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D VMDOPENGL=1 /D VMDTCL=1 /D VMDTK=1 /D strcasecmp=strupcmp /D strncasecmp=strupncmp /D VMDGUI=1 /D VMDFORMS=1 /D VMDFLTK=1 /D VMDSURF=1 /D VMDMSMS=1 /D VMDIMD=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1 /D VMDWITHCARBS=1 /D VOLMAP_SELECTION_HACK=1 /FD /TP /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /fo"res\winvmd.res" /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 ..\..\lib\fltk\WIN32\fltk.lib res\winvmd.res ..\..\lib\tcl\lib_WIN32\tcl85.lib ..\..\lib\tk\lib_WIN32\tk85.lib opengl32.lib glu32.lib wsock32.lib comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib winmm.lib imm32.lib /nologo /version:1.4 /stack:0x800000 /subsystem:console /debug /machine:I386 /nodefaultlib:"libcd.lib" /out:"Release/vmd.exe" /warn:4
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /YX /FD /GZ /c
# ADD CPP /nologo /G5 /MTd /W2 /Gm /vmg /ZI /Od /I "..\..\..\plugins\include" /I "..\..\lib\fltk\include" /I "..\..\lib\tcl\include" /I "..\..\lib\tk\include" /I "..\..\lib\tclx\include" /I "..\..\src" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D VMDOPENGL=1 /D VMDTCL=1 /D VMDTK=1 /D strcasecmp=strupcmp /D strncasecmp=strupncmp /D VMDGUI=1 /D VMDFORMS=1 /D VMDFLTK=1 /D VMDSURF=1 /D VMDMSMS=1 /D VMDIMD=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1 /D VMDWITHCARBS=1 /D VOLMAP_SELECTION_HACK=1 /FR /FD /GZ /TP /c
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /fo"res\winvmd.res" /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ..\..\lib\fltk\WIN32\fltkd.lib res\winvmd.res ..\..\lib\tcl\lib_WIN32\tcl85.lib ..\..\lib\tk\lib_WIN32\tk85.lib opengl32.lib glu32.lib wsock32.lib comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib winmm.lib imm32.lib /nologo /version:1.4 /stack:0x800000 /subsystem:console /debug /debugtype:both /machine:I386 /nodefaultlib:"libcd.lib" /out:"Debug/vmd.exe" /pdbtype:sept /warn:4
# SUBTRACT LINK32 /pdb:none

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "winvmd___Win32_Spaceball_VRPN"
# PROP BASE Intermediate_Dir "winvmd___Win32_Spaceball_VRPN"
# PROP BASE Ignore_Export_Lib 0
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "winvmdfullrelease"
# PROP Intermediate_Dir "winvmdfullrelease"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /G5 /MT /W3 /vmg /O2 /I "..\..\lib\fltk\include" /I "..\..\lib\tcl\include" /I "..\..\lib\tk\include" /I "..\..\lib\tclx\include" /I "..\..\src" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D VMDOPENGL=1 /D VMDTCL=1 /D VMDTK=1 /D strcasecmp=strupcmp /D strncasecmp=strupncmp /D VMDGUI=1 /D VMDFORMS=1 /D VMDFLTK=1 /D VMDSURF=1 /D VMDIMD=1 /FD /TP /c
# ADD CPP /nologo /G5 /MT /W3 /vmg /O2 /I "c:\program files\spacetec\spaceware\sdk\inc" /I "..\..\lib\tachyon\include" /I "..\..\lib\vrpn\lib_WIN32\quat" /I "..\..\lib\vrpn\lib_WIN32\vrpn" /I "..\..\..\plugins\include" /I "..\..\lib\fltk\include" /I "..\..\lib\tcl\include" /I "..\..\lib\tk\include" /I "..\..\src" /D "NDEBUG" /D VMDSPACEWARE=1 /D VMDVRPN=1 /D VMDPOLYHEDRA=1 /D VMDWITHCARBS=1 /D VMDORBITALS=1 /D VMDQUICKSURF=1 /D "WIN32" /D "_CONSOLE" /D "_MBCS" /D VMDOPENGL=1 /D VMDTCL=1 /D VMDTK=1 /D strcasecmp=strupcmp /D strncasecmp=strupncmp /D VMDGUI=1 /D VMDFORMS=1 /D VMDFLTK=1 /D VMDSURF=1 /D VMDMSMS=1 /D VMDIMD=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1 /D VOLMAP_SELECTION_HACK=1 /D VMDTHREADS=1 /D WKFTHREADS=1 /D VMDLIBTACHYON=1 /FD /TP /c
# ADD BASE RSC /l 0x409 /fo"res\winvmd.res" /d "NDEBUG"
# ADD RSC /l 0x409 /fo"res\winvmd.res" /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 ..\..\lib\fltk\WIN32\fltk.lib res\winvmd.res ..\..\lib\tcl\lib_WIN32\tcl80.lib ..\..\lib\tk\lib_WIN32\tk80.lib opengl32.lib glu32.lib wsock32.lib kernel32.lib user32.lib gdi32.lib advapi32.lib winmm.lib /nologo /version:1.4 /stack:0x800000 /subsystem:console /machine:I386 /nodefaultlib:"libcd.lib" /warn:4
# SUBTRACT BASE LINK32 /pdb:none
# ADD LINK32 "c:\program files\spacetec\spaceware\sdk\lib\siapp\bin\win32i\release\mthread\siappmt.lib" ..\..\lib\vrpn\lib_WIN32\quat\pc_win32\Release\quat.lib ..\..\lib\vrpn\lib_WIN32\vrpn\pc_win32\Release\vrpn.lib ..\..\lib\tachyon\lib_WIN32\libtachyon.lib ..\..\lib\fltk\WIN32\fltk.lib res\winvmd.res ..\..\lib\tcl\lib_WIN32\tcl85.lib ..\..\lib\tk\lib_WIN32\tk85.lib opengl32.lib glu32.lib wsock32.lib comctl32.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib winmm.lib imm32.lib /nologo /version:1.4 /stack:0x800000 /subsystem:console /debug /debugtype:both /machine:I386 /nodefaultlib:"libcd.lib" /out:"winvmdfullrelease/vmd.exe" /warn:4
# SUBTRACT LINK32 /pdb:none

!ENDIF 

# Begin Target

# Name "winvmd - Win32 Release"
# Name "winvmd - Win32 Debug"
# Name "winvmd - Win32 Spaceball VRPN"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\src\Animation.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ArtDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\AtomColor.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\AtomLexer.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\AtomParser.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\AtomRep.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\AtomSel.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Axes.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\BaseMolecule.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Benchmark.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\BondSearch.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_animate.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_collab.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_color.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_display.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_imd.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_label.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_material.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_menu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_mobile.C
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_mol.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_mouse.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_parallel.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_plugin.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_render.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_spaceball.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_tool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_trans.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_user.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_util.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_vmdbench.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdAnimate.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdColor.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdDisplay.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdIMD.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdLabel.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMaterial.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMol.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdRender.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CmdTrans.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ColorFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ColorInfo.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CommandQueue.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CoorPluginData.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\CUDAAccel.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DispCmds.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Displayable.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DisplayFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DisplayRocker.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawForce.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolecule.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItem.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItem2.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemMSMS.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemOrbital.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemQuickSurf.C
# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemRibbons.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemRings.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemSurface.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItemVolume.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\DrawRingsUtils.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\FileChooserFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\FileRenderer.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\FileRenderList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\fitrms.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\FPS.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\frame_selector.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GelatoDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryAngle.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryAtom.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryBond.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryDihedral.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryMol.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GeometrySpring.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\GraphicsFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\hash.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Hershey.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ImageIO.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\imd.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\IMDMgr.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\IMDSim.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\IMDSimBlocking.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\IMDSimThread.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Inform.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\inthash.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\intstack.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Isosurface.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\JRegex.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\JString.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\LibTachyonDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MainFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MaterialFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MaterialList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Matrix4.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MayaDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MDFF.C
# End Source File
# Begin Source File

SOURCE=..\..\src\Measure.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MeasureCluster.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MeasurePBC.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MeasureRDF.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MeasureSurface.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MeasureSymmetry.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MobileButtons.C
# End Source File
# Begin Source File

SOURCE=..\..\src\MobileInterface.C
# End Source File
# Begin Source File

SOURCE=..\..\src\MobileTracker.C
# End Source File
# Begin Source File

SOURCE=..\..\src\MolBrowser.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Molecule.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeGraphics.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MolFilePlugin.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Mouse.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\msmpot.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\msmpot_compute.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\msmpot_cubic.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\msmpot_setup.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\MSMSInterface.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLCache.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLExtensions.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLRenderer.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLShader.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Orbital.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_Buttons.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_CmdTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_GrabTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_JoystickButtons.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_JoystickTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_PinchTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_PrintTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_RotateTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_SensorConfig.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_Tool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_Tracker.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_TugTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_UIVR.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNButtons.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNFeedback.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNTracker.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ParseTree.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\pcre.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PeriodicTable.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeAddBond.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeCenter.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeForce.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeMolLabel.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeMove.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeUser.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PlainTextInterp.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PluginMgr.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\POV3DisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\PSDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\QMData.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\QMTimestep.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\QuickSurf.C
# End Source File
# Begin Source File

SOURCE=..\..\src\R3dDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\RadianceDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\RayShadeDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\RenderFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\RenderManDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SaveTrajectoryFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Scene.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SelectionBuilder.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SnapshotDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Spaceball.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SpaceballButtons.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SpaceballTracker.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SpatialSearch.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SpringTool.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Stage.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\STLDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Stride.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Surf.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\SymbolTable.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TachyonDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\tcl_commands.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclCommands.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclGraphics.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclMDFF.C
# End Source File
# Begin Source File

SOURCE=..\..\src\TclMeasure.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclMolInfo.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclTextInterp.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclVec.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\TclVolMap.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Timestep.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\ToolFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\UIObject.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\UIText.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\utilities.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\vmd.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDApp.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDCollab.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\vmdconsole.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDDir.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDDisplayList.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\vmddlopen.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDFltkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\vmdmain.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDQuat.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\vmdsock.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDThreads.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDTitle.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VMDTkMenu.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VolCPotential.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VolMapCreate.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VolMapCreateILS.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VolumeTexture.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VolumetricData.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Vrml2DisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\VrmlDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\WavefrontDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Win32Joystick.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\Win32OpenGLDisplayDevice.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\win32vmdstart.c

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\WKFThreads.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\WKFUtils.C

!IF  "$(CFG)" == "winvmd - Win32 Release"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Debug"

# SUBTRACT CPP /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ELSEIF  "$(CFG)" == "winvmd - Win32 Spaceball VRPN"

# SUBTRACT CPP /D VMDORBITALS=1 /D VMDFORMS=1 /D VMDISOSURFACE=1 /D VMDFIELDLINES=1 /D VMDVOLUMETEXTURE=1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\src\X3DDisplayDevice.C
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=..\..\src\AnimateFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Animation.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ArtDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Atom.h
# End Source File
# Begin Source File

SOURCE=..\..\src\AtomColor.h
# End Source File
# Begin Source File

SOURCE=..\..\src\AtomParser.h
# End Source File
# Begin Source File

SOURCE=..\..\src\AtomRep.h
# End Source File
# Begin Source File

SOURCE=..\..\src\AtomSel.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Axes.h
# End Source File
# Begin Source File

SOURCE=..\..\src\BabelConvert.h
# End Source File
# Begin Source File

SOURCE=..\..\src\BaseMolecule.h
# End Source File
# Begin Source File

SOURCE=..\..\src\BondSearch.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_animate.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_bond.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_color.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_display.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_external.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_imd.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_label.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_material.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_menu.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_mol.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_mouse.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_render.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_tool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_trans.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_user.h
# End Source File
# Begin Source File

SOURCE=..\..\src\cmd_util.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdAnimate.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdColor.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdDisplay.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdExternal.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdIMD.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdLabel.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMaterial.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMenu.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMol.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdMouse.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdRender.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdTrans.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdUser.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CmdUtil.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ColorFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ColorList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ColorUser.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Command.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CommandQueue.h
# End Source File
# Begin Source File

SOURCE=..\..\src\config.h
# End Source File
# Begin Source File

SOURCE=..\..\src\CoorFileData.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DepthSortObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DispCmds.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Displayable.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DisplayFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DrawForce.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolecule.h
# End Source File
# Begin Source File

SOURCE=..\..\src\DrawMolItem.h
# End Source File
# Begin Source File

SOURCE=..\..\src\EditFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\FileRenderer.h
# End Source File
# Begin Source File

SOURCE=..\..\src\FileRenderList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\FilesFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\forms_ui.h
# End Source File
# Begin Source File

SOURCE=..\..\src\FormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Fragment.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Geometry.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryAngle.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryAtom.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryBond.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryDihedral.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GeometryMol.h
# End Source File
# Begin Source File

SOURCE=..\..\src\GraphicsFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Grid.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Gromacs.h
# End Source File
# Begin Source File

SOURCE=..\..\src\hash.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Hershey.h
# End Source File
# Begin Source File

SOURCE=..\..\src\imd.h
# End Source File
# Begin Source File

SOURCE=..\..\src\IMDMgr.h
# End Source File
# Begin Source File

SOURCE=..\..\src\IMDSim.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Inform.h
# End Source File
# Begin Source File

SOURCE=..\..\src\JRegex.h
# End Source File
# Begin Source File

SOURCE=..\..\src\JString.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MainFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MaterialFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MaterialList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MaterialUser.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Matrix4.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Measure.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MolAction.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Molecule.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeEDM.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeGraphics.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeGrasp.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeIMD.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MoleculeRaster3D.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MolFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Mouse.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MouseFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\MSMSInterface.h
# End Source File
# Begin Source File

SOURCE=..\..\src\NameList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\NormalScene.h
# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\OpenGLRenderer.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_Buttons.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_CaveButtons.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_CaveTracker.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_CmdTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_Feedback.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_FreeVRButtons.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_FreeVRTracker.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_GrabTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_JoystickButtons.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_JoystickTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_PinchTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_RotateTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_Sensor.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_SensorConfig.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_SMDTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_Tool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_Tracker.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_TrackerFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_TugTool.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_UIVR.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNButtons.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNConnection.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNFeedback.h
# End Source File
# Begin Source File

SOURCE=..\..\src\P_VRPNTracker.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ParseTree.h
# End Source File
# Begin Source File

SOURCE=..\..\src\pcre.h
# End Source File
# Begin Source File

SOURCE=..\..\src\pcreinternal.h
# End Source File
# Begin Source File

SOURCE=..\..\src\pcretables.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Pickable.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickMode.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeCenter.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeForce.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeMolLabel.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PickModeMove.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PlainTextInterp.h
# End Source File
# Begin Source File

SOURCE=..\..\src\POV3DisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\PSDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Quat.h
# End Source File
# Begin Source File

SOURCE=..\..\src\R3dDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\RadianceDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\RayShadeDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadDCD.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadEDM.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadMDLMol.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadPARM.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadPDB.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ReadPSF.h
# End Source File
# Begin Source File

SOURCE=..\..\src\RenderFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\RenderManDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Residue.h
# End Source File
# Begin Source File

SOURCE=..\..\src\ResizeArray.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Scene.h
# End Source File
# Begin Source File

SOURCE=..\..\src\SimFormsObj.h
# End Source File
# Begin Source File

SOURCE=..\..\src\SnapshotDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\SortableArray.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Stack.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Stage.h
# End Source File
# Begin Source File

SOURCE=..\..\src\STLDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Stride.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Surf.h
# End Source File
# Begin Source File

SOURCE=..\..\src\SymbolTable.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TachyonDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TclCommands.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TclMeasure.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TclTextInterp.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TclVec.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TextEvent.h
# End Source File
# Begin Source File

SOURCE=..\..\src\TextInterp.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Timestep.h
# End Source File
# Begin Source File

SOURCE=..\..\src\UIExternal.h
# End Source File
# Begin Source File

SOURCE=..\..\src\UIObject.h
# End Source File
# Begin Source File

SOURCE=..\..\src\UIText.h
# End Source File
# Begin Source File

SOURCE=..\..\src\utilities.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VMDApp.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VMDDir.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VMDDisplayList.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VMDEvent.h
# End Source File
# Begin Source File

SOURCE=..\..\src\vmdsock.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VMDTitle.h
# End Source File
# Begin Source File

SOURCE=..\..\src\VrmlDisplayDevice.h
# End Source File
# Begin Source File

SOURCE=..\..\src\Win32ftp.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
