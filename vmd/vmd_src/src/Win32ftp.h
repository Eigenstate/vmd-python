/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2019 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Win32ftp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Very simple Windows FTP client code used for past implementations
 *   of webpdb/mol pdbload
 ***************************************************************************/

#define FTP_FAILURE  -1
#define FTP_SUCCESS   0

vmd_ftpclient(const char * site, const char * remotefile, const char * localfile);
