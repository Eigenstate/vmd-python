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
*      $RCSfile: OptiXDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.3 $         $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*   VMD interface for NVIDIA GPU hardware video encoding APIs
*
***************************************************************************/

#ifndef NVENCMGR_H
#define NVENCMGR_H

#include "nvEncodeAPI.h" // NVIDIA's NVENC API header

#define NVENCMGR_SUCCESS  0
#define NVENCMGR_ERROR   -1

typedef NVENCSTATUS (NVENCAPI* PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST *functionList);

class NVENCMgr {
  public:
    NVENCMgr(void);
    ~NVENCMgr(void);
    int init(void);
    int open_session(void);
    int create_inbufs(int bufcount);

  private:
    int enc_ready;
    void *nvenc_lib;
    PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance;
    NV_ENCODE_API_FUNCTION_LIST nvenc_fctns;
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS session_parms;
    void *nvenc_ctx;

    NV_ENC_PRESET_CONFIG preset_conf;
    NV_ENC_INITIALIZE_PARAMS init_parms;
    NV_ENC_CONFIG conf;
    GUID enc_preset;
    GUID codec;

    NV_ENC_CREATE_INPUT_BUFFER create_inbuf;
    int inbuf_count;

    NV_ENC_CREATE_BITSTREAM_BUFFER create_outbuf;
};

#endif
