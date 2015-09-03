/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2011 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: Win32ftp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $      $Date: 2010/12/16 04:08:52 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Very simple Windows FTP client code used for past implementations
 *   of webpdb/mol pdbload 
 ***************************************************************************/

#include <stdio.h>
#include <afxinet.h> 
#include <Win32ftp.h>

vmd_ftpclient(const char * site, 
            const char * remotefile, 
            const char * localfile) {
  CInternetSession S("Eagle FTP"); 
  CFtpConnection *f; 

  try { 
    f = S.GetFtpConnection(site,
                           "anonymous",
                           "anonymous@anonymous.org",
                           21,
                           FALSE); 
    f->SetCurrentDirectory("/"); 
    f->GetFile(remotefile, localfile,
               FALSE,
               FILE_ATTRIBUTE_NORMAL,
               FTP_TRANSFER_TYPE_BINARY,
               1);
    
    delete f; 
    S.Close(); 
  
    return FTP_SUCCESS;
  } 

  catch (CInternetException) { 
    printf("FTP Error!\n"); 
    return FTP_FAILURE;
  } 

  return FTP_SUCCESS;
} 

#if defined(FTPMAIN)
int main(int argc, char **argv) {
  int rc;
  printf("VMD Win32 FTP Client\n");
  if (argc != 4) {
    printf("usage: %s ftp.rcsb.org /pub/README README.txt\n", argv[0]);
    return FTP_FAILURE;
  }

  printf("%s:%s:%s\n",argv[1], argv[2], argv[3]);

  rc=vmd_ftpclient(argv[1], argv[2], argv[3]);   

  if (rc != FTP_SUCCESS) {
    printf("FTP failed.\n"); 
  }
  return rc;
}
#endif

