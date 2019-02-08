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
*      $Revision: 1.5 $         $Date: 2019/01/17 21:21:00 $
*
***************************************************************************
* DESCRIPTION:
*   VMD interface for NVIDIA GPU hardware video encoding APIs
*
*   The implementation here could in principal use either high level
*   libraries like NvPipe or GRID, or lower level video encode APIs 
*   like NvEnc, depending on what hardware the target platform has and
*   whether we want to use codecs not supported by certain APIs.
*
*   NvPipe: 
*     https://github.com/NVIDIA/NvPipe/blob/master/README.md
*
***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NVENCMgr.h"
#include "Inform.h"
#include "vmddlopen.h"

#include "cuda.h"

#define DBG() printf("NVENCMgr) %s\n", __func__);

typedef NVENCSTATUS (NVENCAPI* PNVENCODEAPICREATEINSTANCE)(NV_ENCODE_API_FUNCTION_LIST *functionList);

class nvencfctns {
  PNVENCODEAPICREATEINSTANCE nvEncodeAPICreateInstance;
  NV_ENCODE_API_FUNCTION_LIST fctns; 
};


NVENCMgr::NVENCMgr(void) {
  DBG();

  nvenc_lib = NULL;
  enc_ready = 0;
  inbuf_count = 0;
  memset(&nvenc_fctns, 0, sizeof(nvenc_fctns));

  memset(&session_parms, 0, sizeof(session_parms));
  session_parms.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  session_parms.apiVersion = NVENCAPI_VERSION;

  memset(&preset_conf, 0, sizeof(preset_conf));
  preset_conf.version = NV_ENC_PRESET_CONFIG_VER;
  preset_conf.presetCfg.version = NV_ENC_CONFIG_VER;

  memset(&init_parms, 0, sizeof(init_parms));
  init_parms.version = NV_ENC_INITIALIZE_PARAMS_VER;

  memset(&conf, 0, sizeof(conf));
  conf.version = NV_ENC_CONFIG_VER;

  memset(&create_inbuf, 0, sizeof(create_inbuf));
  create_inbuf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;

  memset(&create_outbuf, 0, sizeof(create_outbuf));
  create_outbuf.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

  memset(&enc_preset, 0, sizeof(enc_preset));
}


NVENCMgr::~NVENCMgr(void) {
  DBG();

  if (nvenc_lib)
    vmddlclose(nvenc_lib);
}


int NVENCMgr::init(void) {
  DBG();
 
  nvenc_lib = vmddlopen("libnvidia-encode.so.1");
  if (!nvenc_lib) {
    msgInfo << "NVENCMgr) Failed to open NVENC hardware video encoder library." 
            << sendmsg;
    vmddlclose(nvenc_lib);
    return -1;
  } 

  nvenc_fctns.version = NV_ENCODE_API_FUNCTION_LIST_VER;

  nvEncodeAPICreateInstance = (PNVENCODEAPICREATEINSTANCE) vmddlsym(nvenc_lib, "NvEncodeAPICreateInstance");
  if (!nvEncodeAPICreateInstance) {
    msgInfo << "NVENCMgr) Failed to load NVENC hardware video encoder instance fctn." 
            << sendmsg;
    vmddlclose(nvenc_lib);
    return -1;
  }

  NVENCSTATUS rc;
  rc = nvEncodeAPICreateInstance(&nvenc_fctns);
  if (rc != NV_ENC_SUCCESS) {
    msgInfo << "NVENCMgr) Failed to load NVENC hardware video encoder instance fctn table." 
            << sendmsg;
  }

  return 0;
}


int NVENCMgr::open_session(void) {
  DBG();

  CUcontext cuctx = NULL;
  CUresult curc;
  CUcontext cuctxcurrent = NULL;
  int cudevcount = 0;
  CUdevice cudev = 0;

#if 0
  if ((curc = cuInit(0)) != CUDA_SUCCESS) {
    printf("NVENCMgr) failed to initialize CUDA driver API\n");
    return -1;
  }
#endif
 
  if ((curc = cuDeviceGetCount(&cudevcount)) != CUDA_SUCCESS) {
    printf("NVENCMgr) failed to query CUDA driver API device count\n");
    return -1;
  }

  if (cudevcount < 1) {
    printf("NVENCMgr) no CUDA devices found for NVENC video encoding\n");
    return -1;
  } else {
    printf("NVENCMgr) CUDA dev count: %d\n", cudevcount);
  }

  if ((curc = cuDeviceGet(&cudev, 0)) != CUDA_SUCCESS) {
    printf("NVENCMgr) Unable to bind CUDA device 0\n");
    return -1;
  }

  if ((curc = cuCtxCreate(&cuctx, 0, cudev)) != CUDA_SUCCESS) {
    printf("NVENCMgr) Unable create CUDA ctx\n");
    return -1;
  }

  if ((curc = cuCtxPopCurrent(&cuctxcurrent)) != CUDA_SUCCESS) {
    printf("NVENCMgr) Unable pop current CUDA ctx\n");
    return -1;
  }
//  curc = cuCtxGetCurrent(&cuctx);


  session_parms.device = cuctx;
  session_parms.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  // XXX client keys are not presently required
  // session_parms.clientKeyPtr = &NV_CLIENT_KEY;

//  printf("NVENCMgr) Creating NVENC hardware encoder session...\n");

  NVENCSTATUS encstat;
  encstat = nvenc_fctns.nvEncOpenEncodeSessionEx(&session_parms, &nvenc_ctx);
  if (encstat != NV_ENC_SUCCESS) {
    printf("NVENCMgr) nvEncOpenEncodeSessionEx() returned an error!\n");
    return -1;
  }

  // setup encoder presets
  enc_preset = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;
  // enc_preset = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;
  // enc_preset = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
  // enc_preset = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID;
  // enc_preset = NV_ENC_PRESET_LOSSLESS_HP_GUID;
  // enc_preset = NV_ENC_PRESET_HQ_GUID;
  // enc_preset = NV_ENC_PRESET_DEFAULT_GUID;

#if 1
  codec = NV_ENC_CODEC_H264_GUID;
#else
  codec = NV_ENC_CODEC_HEVC_GUID;
#endif

  printf("NVENCMgr) establishing NVENC hardware encoder preset config\n");
  encstat = nvenc_fctns.nvEncGetEncodePresetConfig(nvenc_ctx, codec, 
                                                   enc_preset, &preset_conf);
  if (encstat != NV_ENC_SUCCESS) {
    printf("NVENCMgr) nvEncGetEncodePresetConfig() returned an error!\n");
    return -1;
  }

  init_parms.version = NV_ENC_INITIALIZE_PARAMS_VER;
  init_parms.encodeWidth = 1920;
  init_parms.encodeHeight = 1080;
  init_parms.darWidth = 1920;
  init_parms.darHeight = 1080;
  init_parms.maxEncodeWidth = 1920;
  init_parms.maxEncodeHeight = 1080;
  init_parms.frameRateNum = 1;
  init_parms.frameRateDen = 30;
  init_parms.enableEncodeAsync = 0;
  init_parms.enablePTD = 1;

  init_parms.encodeGUID = codec;
  init_parms.presetGUID = enc_preset;
  init_parms.encodeConfig = &preset_conf.presetCfg;

//  printf("NVENCMgr) Initializing NVENC hardware encoder ...\n");
  encstat = nvenc_fctns.nvEncInitializeEncoder(nvenc_ctx, &init_parms);
  if (encstat != NV_ENC_SUCCESS) {
    printf("NVENCMgr) nvEncInitializeEncoder() returned an error!\n");
    return -1;
  }

#if 0
  // setting refs to zero allows hardware encoder to decide
  conf.encodeCodecConfig.h264Config.maxNumRefFrames = 0;
  conf.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = 0;

  // infinite GOP == 0, I-only == 1?
  conf.gopLength = 30;
  if (conf.gopLength > 0) {
    conf.frameIntervalP = 3; // 0=I only, 1=IP, 2=IBP, 3=IBBP, ...
  } else {
    conf.frameIntervalP = 0; // 0=I only, 1=IP, 2=IBP, 3=IBBP, ...
    conf.gopLength = 1;
  }

  conf.encodeCodecConfig.h264Config.idrPeriod = conf.gopLength;
  // H.264 only
  conf.encodeCodecConfig.h264Config.hierarchicalPFrames = 1;
  conf.encodeCodecConfig.h264Config.hierarchicalBFrames = 1;

  conf.rcParams.averageBitRate = 15000000; // 15Mbps
  conf.rcParams.maxBitRate     = 20000000; // 20Mbps
#endif

  // flag encoder as ready
  enc_ready = 1;
  printf("NVENCMgr) NVENC hardware encoder ready.\n");

  return 0;
}


int NVENCMgr::create_inbufs(int bufcount) {
  DBG();
  NVENCSTATUS encstat;

  if (!enc_ready) {
    printf("NVENCMgr) Error creating input buffers, encoder not ready!\n");
    return -1;
  }

  // must be multiples of 32
  create_inbuf.width = 1920;
  create_inbuf.height= 1080;
  create_inbuf.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
  create_inbuf.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;

  inbuf_count = bufcount;
  int i;
  for (i=0; i<inbuf_count; i++) {
    encstat = nvenc_fctns.nvEncCreateInputBuffer(nvenc_ctx, &create_inbuf); 
    if (encstat != NV_ENC_SUCCESS) {
      printf("NVENCMgr) Failed to create input buffers!\n");
      return -1;
    }
  } 

  return 0;
}

