////////////////////////////////////////////////////////////////////
//                                                                //
// routines for loading / storing contiguous data                 //
// assumes reg_len is a power of 2                                //
//                                                                //
// based on code originally written by Jeremy Appleyard           //
// modified by Mike Giles and Istvan Reguly                       //
//                                                                //
////////////////////////////////////////////////////////////////////

template <int reg_len, int WARP_SIZE, typename REAL>
__device__ void loadDataIntoRegisters_contig(int tid, int len,
                                             REAL* regArray,
                                    volatile REAL* smem, 
                                       const REAL* devArray, 
                                             REAL  blank) {
  for (int i=0; i<reg_len; i++) {
    int gmemIdx = tid + i*WARP_SIZE;
    if (gmemIdx < len) regArray[i] = devArray[gmemIdx];
    else               regArray[i] = blank;
  }
            
  for (int i=0; i<reg_len; i++) {
    int smemIdx = tid + i*(WARP_SIZE+1);
    smem[smemIdx] = regArray[i];
  }
   
  for (int i=0; i<reg_len; i++) {
    regArray[i] = smem[i + (tid*reg_len*(WARP_SIZE+1))/WARP_SIZE];
  }
}


template <int reg_len, int WARP_SIZE, typename REAL>
__device__ void storeDataFromRegisters_contig(int tid, int len,
                                              REAL* regArray,
                                     volatile REAL* smem, 
                                              REAL* devArray) {
  for (int i=0; i<reg_len; i++) {
    smem[i + (tid*reg_len*(WARP_SIZE+1))/WARP_SIZE] = regArray[i];
  }

  for (int i=0; i<reg_len; i++) {
    int smemIdx = tid + i*(WARP_SIZE+1);
    regArray[i] = smem[smemIdx];
  }
   
  for (int i=0; i<reg_len; i++) {
    int gmemIdx = tid + i*WARP_SIZE;
    if (gmemIdx < len) devArray[gmemIdx] = regArray[i];
  }
}


template <int reg_len, int WARP_SIZE, typename REAL>
__device__ void incDataFromRegisters_contig(int tid, int len,
                                              REAL* regArray,
                                     volatile REAL* smem, 
                                              REAL* devArray) {
  for (int i=0; i<reg_len; i++) {
    smem[i + (tid*reg_len*(WARP_SIZE+1))/WARP_SIZE] = regArray[i];
  }

  for (int i=0; i<reg_len; i++) {
    int gmemIdx = tid + i*WARP_SIZE;
    if (gmemIdx < len) regArray[i] = devArray[gmemIdx];
  }

  for (int i=0; i<reg_len; i++) {
    int smemIdx = tid + i*(WARP_SIZE+1);
      regArray[i] += smem[smemIdx];
  }
   
  for (int i=0; i<reg_len; i++) {
    int gmemIdx = tid + i*WARP_SIZE;
    if (gmemIdx < len) devArray[gmemIdx] = regArray[i];
  }
}
