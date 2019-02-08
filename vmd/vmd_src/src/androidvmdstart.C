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
 *      $RCSfile: androidvmdstart.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $      $Date: 2019/01/17 21:21:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Android startup code
 ***************************************************************************/

// only compile this file if we're building on Android
#if defined(ANDROID)
#include "androidvmdstart.h"
#include "vmd.h"

//
// VMD is wrapped in a JNI shared object and called by Java...
//
#include <stdio.h>
#include <jni.h>

#if !defined(VMD_JNI_CLASSNAME)
#define VMD_JNI_CLASSNAME "edu/uiuc/VMD/VMD"
#endif
#if !defined(VMD_JNI_WRAPFUNC)
#define VMD_JNI_WRAPFUNC Java_edu_uiuc_VMD_VMD_nativeMain
#endif

#define NUM_METHODS 5
#define METH_logOutput 0
#define METH_getNumMsgs 1
#define METH_getPlatformValue 2
#define METH_getMsgBlock 3
#define METH_getMsgNonBlock 4

struct AndroidHandle {
  JNIEnv *env;
  jobject thiz;
  jmethodID meths[NUM_METHODS];
} global_AndHand;

// ----------------------------------------------------------------------
void logtojava(AndroidHandle &ah, const char *logstring)
{
  /* this actually logs a char*, logstring */
  (ah.env)->CallVoidMethod(ah.thiz,
                        ah.meths[METH_logOutput],
                       (ah.env)->NewStringUTF( logstring));
}

/* ------------------------------------------------------------ */
/* output is allocated elsewhere */
/* key: FilesDir    - gets internal phone memory space for the app. The zip
                      file from server is unzipped to this dir                */
/* key: VMDDIR      - gets internal phone memory space, where VMD files live
                      (currently exactly same as FilesDir                     */
/* key: ExternalStorageDir  - gets SDcard memory space (global, use with care)*/
char *getPlatformValue(AndroidHandle &ah, const char *key, char *output)
{

  /* NewStringUTF makes a java string from a char* */
  jobject j = (ah.env)->CallObjectMethod(ah.thiz,
                                  ah.meths[METH_getPlatformValue],
                                 (ah.env)->NewStringUTF(key));

  const char* str = (ah.env)->GetStringUTFChars((jstring) j, NULL);

  strcpy(output, str);

  (ah.env)->ReleaseStringUTFChars((jstring) j, str);
  return output;
}

/* ------------------------------------------------------------ */
/* Get string Message from Android frontend.  Blocks until message
 * is actually sent.
 * 'output' is allocated elsewhere.  
 * Returns pointer to 'output' as convenience.
 */
char *getMessage(AndroidHandle &ah, char *output)
{
  jobject j = (ah.env)->CallObjectMethod(ah.thiz,
                                  ah.meths[METH_getMsgBlock]);

  const char* str = (ah.env)->GetStringUTFChars((jstring) j, NULL);

  strcpy(output, str);

  (ah.env)->ReleaseStringUTFChars((jstring) j, str);
  return output;
}

/* ------------------------------------------------------------ */
/* Get string Message from Android frontend if one is ready.
 * 'output' is allocated elsewhere.  Returns 'output' if message
 * existed.  Returns 0 (and doesn't modify 'output') if no
 * message was waiting. 
 * XXX need think about how to use
 */
char *getMessageNonBlock(AndroidHandle &ah, char *output)
{
  jobject j = (ah.env)->CallObjectMethod(ah.thiz,
                                    ah.meths[METH_getMsgNonBlock]);

  if (!j) {
    return 0;
  }

  const char* str = (ah.env)->GetStringUTFChars((jstring) j, NULL);

  strcpy(output, str);

  (ah.env)->ReleaseStringUTFChars((jstring) j, str);
  return output;
}

/* ------------------------------------------------------------ */
/*  Get the number of messages that the Android frontend has queued
 *  up ready to be consumed by one of the getMessage fcts.
*/
jint getNumMessages(AndroidHandle &ah)
{
  return (ah.env)->CallIntMethod(ah.thiz,
                                ah.meths[METH_getNumMsgs]);
}

// ----------------------------------------------------------------------
/* Internally used fct called at startup to perform expensive tasks once.
 */
void cacheAndroidMethodIDs(AndroidHandle &ah)
{
  // cache method IDs. Could be done in fct, of course
  // these calls are expensive
  jclass clazz = (ah.env)->FindClass("edu/uiuc/VMD/VMD");
  ah.meths[METH_logOutput] = (ah.env)->GetMethodID(
                        clazz, "logOutput", "(Ljava/lang/String;)V");
  ah.meths[METH_getNumMsgs] = (ah.env)->GetMethodID(
                        clazz, "getNumMsgs", "()I");
  ah.meths[METH_getPlatformValue] = (ah.env)->GetMethodID(
                        clazz, "getPlatformValue",
                        "(Ljava/lang/String;)Ljava/lang/String;");
  ah.meths[METH_getMsgBlock] = (ah.env)->GetMethodID(
                        clazz, "getMsgBlock", "()Ljava/lang/String;");
  ah.meths[METH_getMsgNonBlock] = (ah.env)->GetMethodID(
                        clazz, "getMsgNonBlock", "()Ljava/lang/String;");
}

extern "C" {

//
// Wrapper function to hide use of the cached global state until
// until we make appropriate changes so that the JNI launcher has
// a mechanism to provide VMD with the JNI objects for use by calls
// back to android APIs.
//
void log_android(const char *prompt, const char * msg) {
  char logstring[2048];

  strncpy(logstring, prompt, sizeof(logstring)-2);
  strcat(logstring, msg);
  strcat(logstring, "\n");

  logtojava(global_AndHand, logstring);
}


//
// This is the main JNI wrapper function.
// Contains startup code, neverending loop, shutdown code, etc...
//
void VMD_JNI_WRAPFUNC(JNIEnv* env, jobject thiz) {
  char* rargv[10];
 
  global_AndHand.env = env;   // XXX this is a hack!
  global_AndHand.thiz = thiz; // XXX this is a hack!

  cacheAndroidMethodIDs(global_AndHand);    // XXX this caches into a hack!

  fprintf(stderr, "--stderr fprintf---------------------------------\n");
  printf("---regular printf----------------------------\n");
  fflush(stdout);
  log_android("", "--Log event ---------------------");

#if 1
  printf("VMD Android platform info:\n");
  printf("  sizeof(char): %d\n", sizeof(char));
  printf("  sizeof(int): %d\n", sizeof(int));
  printf("  sizeof(long): %d\n", sizeof(long));
  printf("  sizeof(void*): %d\n", sizeof(void*));
  fflush(stdout);
#endif

  char tmp[8192];
  const char * vmddir = NULL;

  // set to a worst-case guess until we have something better.
  vmddir = "/data/data/edu.uiuc.VMD/files/vmd";

#if 1
  // Query Android for app directories and files here...
  char androidappdatadir[8192];

  memset(androidappdatadir, 0, sizeof(androidappdatadir));
  getPlatformValue(global_AndHand, "FilesDir", androidappdatadir);

  if (strlen(androidappdatadir) > 0) {
//    log_android("ANDROID APP DIR: ", androidappdatadir);
    strcat(androidappdatadir, "/vmd");
    vmddir = androidappdatadir;
  }
#endif

  if (vmddir == NULL) {
    return; // fail/exit
  }

  if (!getenv("VMDDIR")) {
    setenv("VMDDIR", vmddir, 1);
  }

  if (!getenv("TCL_LIBRARY")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/tcl");
    setenv("TCL_LIBRARY", tmp, 1);
  }

  if (!getenv("TK_LIBRARY")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/tk");
    setenv("TK_LIBRARY", tmp, 1);
  }

  if (!getenv("PYTHONPATH")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/python");
    setenv("PYTHONPATH", tmp, 1);
  } else {
    strcpy(tmp, getenv("PYTHONPATH"));
    strcat(tmp, ":");
    strcat(tmp, vmddir);
    strcat(tmp, "/scripts/python");
    setenv("PYTHONPATH", tmp, 1);
  }

  if (!getenv("STRIDE_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_ANDROIDARMV7A)
    strcat(tmp, "/stride_ANDROIDARMV7A");
#else
#error unhandled compilation scenario
#endif
    setenv("STRIDE_BIN", tmp, 1);
  }

  if (!getenv("SURF_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_ANDROIDARMV7A)
    strcat(tmp, "/surf_ANDROIDARMV7A");
#else
#error unhandled compilation scenario
#endif
    setenv("SURF_BIN", tmp, 1);
  }

  if (!getenv("TACHYON_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_ANDROIDARMV7A)
    strcat(tmp, "/tachyon_ANDROIDARMV7A");
#else
#error unhandled compilation scenario
#endif
    setenv("TACHYON_BIN", tmp, 1);
  }

  rargv[0] = (char *) "VMD.so";
#if 1
  rargv[1] = (char *) "1e79";
#elif 1
  rargv[1] = (char *) "/data/data/edu.uiuc.VMD/files/alanin.pdb";
#else
  rargv[1] = (char *) "-h";
#endif
  rargv[2] = NULL;

  VMDmain(2, rargv); /* launch VMD... */

  log_android("", "--Log event ---------------------");
  fprintf(stderr, "--stderr fprintf---------------------------------\n");
  printf("---regular printf----------------------------\n");
  fflush(stdout);
}

} // extern "C"

#endif

