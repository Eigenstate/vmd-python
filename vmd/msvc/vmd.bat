@echo off
rem mode co80,12
set LIB=%LIB%;libs;
set PATH=%PATH%;libs;
set VMDDIR=s:/vmd/msvc
set VMDDIRB=s:\vmd\msvc
set TCL_LIBRARY=%VMDDIR%/scripts/tcl
set TCLX_LIBRARY=%VMDDIR%/scripts/tclX
set DP_LIBRARY=%VMDDIR%/scripts/dp
set TRACKERDIR=%VMDDIR%
set VMDTMPDIR=c:\temp
set BABEL_DIR=%VMDDIRB%\babel
set VMDBABELBIN=%VMDDIRB%\babel\babel.exe
set SURF_BIN=%VMDDIR%/surf_WIN32.exe
set STRIDE_BIN=%VMDDIR%/stride_WIN32.exe
set COMSPEC=c:\winnt\system32\cmd.exe

rem "c:\program files\rational\purify\purifyw"  %VMDDIR%\winvmd\debug\winvmd.exe %1 %2 %3 %4 %5
rem  %VMDDIR%\winvmd\debug\winvmd.exe %1 %2 %3 %4 %5
  %VMDDIR%\winvmd\Release\winvmd.exe %1 %2 %3 %4 %5  
