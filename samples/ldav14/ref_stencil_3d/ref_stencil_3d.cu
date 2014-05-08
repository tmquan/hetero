////////////////////////////////////////////////////////////////////
//                                                                //
// standard headers plus new one defining tridiagonal solvers     //
//                                                                //
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "trid.h"
#include "utilities.h"


#define  COLS 16

////////////////////////////////////////////////////////////////////
//                                                                //
// error-checking utility                                         //
//                                                                //
////////////////////////////////////////////////////////////////////

#define cudaSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)

inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    printf("%s(%i) : cudaSafeCall() Runtime API error : %d %s.\n",
           file, line, err, cudaGetErrorString(err) );
    exit(-1);
  }
}

////////////////////////////////////////////////////////////////////
//                                                                //
// explicit Black-Scholes finite difference kernels               //
//                                                                //
////////////////////////////////////////////////////////////////////

//
// linear extrapolation b.c.
//

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_bc1(int NX, int NY, int NZ, REAL *u1) {

  int t, i, j, k, indg, IOFF, JOFF, KOFF;
  t = threadIdx.x + blockIdx.x*blockDim.x;

  IOFF =  1;
  JOFF =  NX+pad_total;
  KOFF = (NX+pad_total)*(NY+2);

  if (t<NX*NY) {
    i = t%NX;
    j = t/NX;
    k = NZ;
    indg = (i+pad_left) + (j+1)*JOFF + (k+1)*KOFF;
    u1[indg] = 2.0f*u1[indg-KOFF] - u1[indg-2*KOFF];
  }
  else if (t<NX*NY + NY*NZ) {
    t = t - NX*NY;
    j = t%NY;
    k = t/NY;
    i = NX;
    indg = (i+pad_left) + (j+1)*JOFF + (k+1)*KOFF;
    u1[indg] = 2.0f*u1[indg-IOFF] - u1[indg-2*IOFF];
  }
  else if (t<NX*NY + NY*NZ + NZ*NX) {
    t = t - NX*NY - NY*NZ;
    k = t%NZ;
    i = t/NZ;
    j = NY;
    indg = (i+pad_left) + (j+1)*JOFF + (k+1)*KOFF;
    u1[indg] = 2.0f*u1[indg-JOFF] - u1[indg-2*JOFF];
  }
}

//
// explicit solvers
//

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_explicit1(int NX, int NY, int NZ, REAL dS,
                             REAL c1_1,  REAL c1_2,  REAL c1_3,
                             REAL c2_1,  REAL c2_2,  REAL c2_3,
                             REAL c3, REAL c12, REAL c13, REAL c23,
                       const REAL* __restrict__ u1,  REAL* __restrict__ u2) {

  REAL S1, S2, S3, t12, t13, t23;
  int   i, j, k, indg, active, IOFF, JOFF, KOFF;

  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = (i+pad_left) + (j+1)*(NX+pad_total) + (NX+pad_total)*(NY+2);

  IOFF =  1;
  JOFF =  NX+pad_total;
  KOFF = (NX+pad_total)*(NY+2);

  active = (i<NX) && (j<NY);

  if (active) {
    for (k=0; k<NZ; k++) {
      S1 = ((REAL) i)*dS;
      S2 = ((REAL) j)*dS;
      S3 = ((REAL) k)*dS;

      t12 = c12*S1*S2;
      t13 = c13*S1*S3;
      t23 = c23*S2*S3;

      u2[indg] = t23                               * u1[indg-KOFF-JOFF]
              +  t13                               * u1[indg-KOFF-IOFF]
              + (c1_3*S3*S3 - c2_3*S3 - t13 - t23) * u1[indg-KOFF]
              +  t12                               * u1[indg-JOFF-IOFF]
              + (c1_2*S2*S2 - c2_2*S2 - t12 - t23) * u1[indg-JOFF]
              + (c1_1*S1*S1 - c2_1*S1 - t12 - t13) * u1[indg-IOFF]
              + (1.0f - c3 - 2.0f*( c1_1*S1*S1 + c1_2*S2*S2 + c1_3*S3*S3
                             - t12 - t13 - t23 ) ) * u1[indg]
              + (c1_1*S1*S1 + c2_1*S1 - t12 - t13) * u1[indg+IOFF]
              + (c1_2*S2*S2 + c2_2*S2 - t12 - t23) * u1[indg+JOFF]
              +  t12                               * u1[indg+JOFF+IOFF]
              + (c1_3*S3*S3 + c2_3*S3 - t13 - t23) * u1[indg+KOFF]
              +  t13                               * u1[indg+KOFF+IOFF]
              +  t23                               * u1[indg+KOFF+JOFF];
      indg += KOFF;
    }
  }
}


template <int pad_left, int pad_total, typename REAL>
__global__ void BS_explicit2(int NX, int NY, int NZ, REAL dS,
                             REAL c1_1,  REAL c1_2,  REAL c1_3,
                             REAL c2_1,  REAL c2_2,  REAL c2_3,
                             REAL c3, REAL c12, REAL c13, REAL c23,
                       const REAL* __restrict__ u1,  REAL* __restrict__ u2) {

  REAL S1, S2, S3, t12, t13, t23;
  REAL u1_mm, u1_om, u1_mo, u1_m, u1_oo, u1_po, u1_op, u1_pp, u;
  int   i, j, k, indg, active, IOFF, JOFF, KOFF;

  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = (i+pad_left) + (j+1)*(NX+pad_total) + (NX+pad_total)*(NY+2);

  IOFF =  1;
  JOFF =  NX+pad_total;
  KOFF = (NX+pad_total)*(NY+2);

  active = (i<NX) && (j<NY);

  if (active) {
    u1_om = u1[indg-KOFF-JOFF];
    u1_mo = u1[indg-KOFF-IOFF];
    u1_m  = u1[indg-KOFF];
    u1_oo = u1[indg];
    u1_po = u1[indg+IOFF];
    u1_op = u1[indg+JOFF];

    for (k=0; k<NZ; k++) {
      S1 = ((REAL) i)*dS;
      S2 = ((REAL) j)*dS;
      S3 = ((REAL) k)*dS;

      t12 = c12*S1*S2;
      t13 = c13*S1*S3;
      t23 = c23*S2*S3;

      u =        t23                               * u1_om
              +  t13                               * u1_mo
              + (c1_3*S3*S3 - c2_3*S3 - t13 - t23) * u1_m;

      u1_mm = u1[indg-JOFF-IOFF];
      u1_om = u1[indg-JOFF];
      u1_mo = u1[indg-IOFF];
      u1_pp = u1[indg+IOFF+JOFF];

      u = u   +  t12                               * u1_mm
              + (c1_2*S2*S2 - c2_2*S2 - t12 - t23) * u1_om
              + (c1_1*S1*S1 - c2_1*S1 - t12 - t13) * u1_mo
              + (1.0f - c3 - 2.0f*( c1_1*S1*S1 + c1_2*S2*S2 + c1_3*S3*S3
                             - t12 - t13 - t23 ) ) * u1_oo
              + (c1_1*S1*S1 + c2_1*S1 - t12 - t13) * u1_po
              + (c1_2*S2*S2 + c2_2*S2 - t12 - t23) * u1_op
              +  t12                               * u1_pp;

      indg += KOFF;
      u1_m  = u1_oo;
      u1_oo = u1[indg];
      u1_po = u1[indg+IOFF];
      u1_op = u1[indg+JOFF];

      u = u   + (c1_3*S3*S3 + c2_3*S3 - t13 - t23) * u1_oo
              +  t13                               * u1_po
              +  t23                               * u1_op;

      u2[indg-KOFF] = u;
    }
  }
}



template <int pad_left, int pad_total, typename REAL, typename REAL2>
__launch_bounds__(256, 3) // (max 256 threads per block, min 3 blocks per SMX)
__global__ void BS_explicit3(int NX, int NY, int NZ, REAL dS,
                             REAL c1_1,  REAL c1_2,  REAL c1_3,
                             REAL c2_1,  REAL c2_2,  REAL c2_3,
                             REAL c3, REAL c12, REAL c13, REAL c23,
                       const REAL2 * __restrict__ u1,
                             REAL2 * __restrict__ u2) {

  REAL  S1m, S1p, S2, S3, t12m, t12p, t13m, t13p, t23;
  int   i, j, k, indg, active, JOFF, KOFF;

  REAL2 u1_mm, u1_om, u1_pm, u1_mp, u1_op, u1_pp, u;
  REAL  u1_om_w, u1_mm_w, u1_pm_z, u1_op_z;

  i    = threadIdx.x - 1 + blockIdx.x*(blockDim.x-2);
  j    = threadIdx.y     + blockIdx.y*blockDim.y;

  JOFF = (NX+pad_total)/2;
  KOFF = JOFF*(NY+2);

  indg  = i + pad_left/2 + (j+1)*JOFF;

  active = (i<=NX/2) && (j<NY);

  if (active) {
    u1_mm = u1[indg-JOFF];
    u1_om = u1[indg     ];
    u1_pm = u1[indg+JOFF];
    indg += KOFF;
    u1_mp = u1[indg-JOFF];
    u1_op = u1[indg     ];
    u1_pp = u1[indg+JOFF];

    u1_om_w = __shfl_up  (u1_om.y,1);
    u1_op_z = __shfl_down(u1_op.x,1);

    for (k=0; k<NZ; k++) {
      S1m = ((REAL) (2*i  ))*dS;
      S1p = ((REAL) (2*i+1))*dS;
      S2  = ((REAL) j)*dS;
      S3  = ((REAL) k)*dS;

      t12m = c12*S2*S1m;
      t12p = c12*S2*S1p;
      t13m = c13*S3*S1m;
      t13p = c13*S3*S1p;
      t23  = c23*S2*S3;

      u.x =
             t23                                * u1_mm.x
          +  t13m                               * u1_om_w
          + (c1_3*S3*S3 - c2_3*S3 - t13m - t23) * u1_om.x;
      u.y =
             t23                                * u1_mm.y
          +  t13p                               * u1_om.x
          + (c1_3*S3*S3 - c2_3*S3 - t13p - t23) * u1_om.y;

      u1_mm = u1_mp;
      u1_om = u1_op;
      u1_pm = u1_pp;

      u1_mm_w = __shfl_up  (u1_mm.y,1);
//    u1_mm_z = __shfl_down(u1_mm.x,1);
      u1_om_w = __shfl_up  (u1_om.y,1);
//    u1_om_z = __shfl_down(u1_om.x,1);  == u1_op_z
//    u1_pm_w = __shfl_up  (u1_pm.y,1);
      u1_pm_z = __shfl_down(u1_pm.x,1);

      u.x = u.x
          +  t12m                                   * u1_mm_w
          + (c1_2*S2*S2   - c2_2*S2  - t12m - t23 ) * u1_mm.x
          + (c1_1*S1m*S1m - c2_1*S1m - t12m - t13m) * u1_om_w
          + (1.0f - c3 - 2.0f*( c1_1*S1m*S1m + c1_2*S2*S2 + c1_3*S3*S3
                            - t12m - t13m - t23 ) ) * u1_om.x
          + (c1_1*S1m*S1m + c2_1*S1m - t12m - t13m) * u1_om.y
          + (c1_2*S2*S2   + c2_2*S2  - t12m - t23 ) * u1_pm.x
          +  t12m                                   * u1_pm.y;
      u.y = u.y
          +  t12p                                   * u1_mm.x
          + (c1_2*S2*S2   - c2_2*S2  - t12p - t23 ) * u1_mm.y
          + (c1_1*S1p*S1p - c2_1*S1p - t12p - t13p) * u1_om.x
          + (1.0f - c3 - 2.0f*( c1_1*S1p*S1p + c1_2*S2*S2 + c1_3*S3*S3
                            - t12p - t13p - t23 ) ) * u1_om.y
          + (c1_1*S1p*S1p + c2_1*S1p - t12p - t13p) * u1_op_z
          + (c1_2*S2*S2   + c2_2*S2  - t12p - t23 ) * u1_pm.y
          +  t12p                                   * u1_pm_z;

      indg += KOFF;
      u1_mp = u1[indg-JOFF];
      u1_op = u1[indg     ];
      u1_pp = u1[indg+JOFF];

      u1_op_z = __shfl_down(u1_op.x,1);

      u.x = u.x
          + (c1_3*S3*S3 + c2_3*S3 - t13m - t23) * u1_op.x
          +  t13m                               * u1_op.y
          +  t23                                * u1_pp.x;
      u.y = u.y
          + (c1_3*S3*S3 + c2_3*S3 - t13p - t23) * u1_op.y
          +  t13p                               * u1_op_z
          +  t23                                * u1_pp.y;

      if (threadIdx.x>0 && threadIdx.x<blockDim.x-1 && i<NX/2)
        u2[indg-KOFF] = u;
    }
  }
}


////////////////////////////////////////////////////////////////////
//                                                                //
// implicit Black-Scholes finite difference kernels               //
//                                                                //
////////////////////////////////////////////////////////////////////

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_implicit2_rhs(int NX, int NY, int NZ, REAL dS,
                                 REAL c1_1,  REAL c1_2,  REAL c1_3,
                                 REAL c2_1,  REAL c2_2,  REAL c2_3,
                                 REAL c3, REAL c12, REAL c13, REAL c23,
                           const REAL* __restrict__ u1,
                                 REAL* __restrict__ u2) {

  REAL S1, S2, S3, t12, t13, t23;
  REAL u1_mm, u1_om, u1_mo, u1_m, u1_oo, u1_po, u1_op, u1_pp, u;
  int   i, j, k, indg, active, IOFF, JOFF, KOFF;

  i    = threadIdx.x + blockIdx.x*blockDim.x;
  j    = threadIdx.y + blockIdx.y*blockDim.y;
  indg = (i+pad_left) + (j+1)*(NX+pad_total) + (NX+pad_total)*(NY+2);

  IOFF =  1;
  JOFF =  NX+pad_total;
  KOFF = (NX+pad_total)*(NY+2);

  active = (i<NX) && (j<NY);

  if (active) {
    u1_om = u1[indg-KOFF-JOFF];
    u1_mo = u1[indg-KOFF-IOFF];
    u1_m  = u1[indg-KOFF];
    u1_oo = u1[indg];
    u1_po = u1[indg+IOFF];
    u1_op = u1[indg+JOFF];

    for (k=0; k<NZ; k++) {
      S1 = ((REAL) i)*dS;
      S2 = ((REAL) j)*dS;
      S3 = ((REAL) k)*dS;

      t12 = c12*S1*S2;
      t13 = c13*S1*S3;
      t23 = c23*S2*S3;

      u =        t23                               * u1_om
              +  t13                               * u1_mo
              + (c1_3*S3*S3 - c2_3*S3 - t13 - t23) * u1_m;

      u1_mm = u1[indg-JOFF-IOFF];
      u1_om = u1[indg-JOFF];
      u1_mo = u1[indg-IOFF];
      u1_pp = u1[indg+IOFF+JOFF];

      u = u   +  t12                               * u1_mm
              + (c1_2*S2*S2 - c2_2*S2 - t12 - t23) * u1_om
              + (c1_1*S1*S1 - c2_1*S1 - t12 - t13) * u1_mo
              + (     - c3 - 2.0f*( c1_1*S1*S1 + c1_2*S2*S2 + c1_3*S3*S3
                             - t12 - t13 - t23 ) ) * u1_oo
              + (c1_1*S1*S1 + c2_1*S1 - t12 - t13) * u1_po
              + (c1_2*S2*S2 + c2_2*S2 - t12 - t23) * u1_op
              +  t12                               * u1_pp;

      indg += KOFF;
      u1_m  = u1_oo;
      u1_oo = u1[indg];
      u1_po = u1[indg+IOFF];
      u1_op = u1[indg+JOFF];

      u = u   + (c1_3*S3*S3 + c2_3*S3 - t13 - t23) * u1_oo
              +  t13                               * u1_po
              +  t23                               * u1_op;

      u2[indg-KOFF] = u;
    }
  }
}

//
// solves tridiagonal equations in x-direction, and increments solution
//

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_implicit2_x(int NX, int NY, int NZ, REAL dS,
                               REAL c1, REAL c2, REAL c3,
                               REAL* __restrict__ u,
                         const REAL* __restrict__ rhs ) {
  volatile __shared__ REAL smem[(256+8)*4];

  REAL S, lambda, gamma, a[8], b[8], c[8], d[8];
  int  j, k, tid;

  tid  = threadIdx.x;
  j    = threadIdx.y;
  k    = blockIdx.x;

  rhs  = rhs + pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);
  u    = u   + pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);

  for ( ; j<NY; j=j+4) {
    for (int i=0; i<8; i++) {
      S      = (8*tid+i) * dS;
      lambda = c1*S*S;
      gamma  = c2*S;

      a[i] =              - ( lambda - gamma );
      b[i] = 1.0f + c3 + 2.0f*lambda;
      c[i] =              - ( lambda + gamma );
    }

    if (tid==31) {
      a[7] =           + 2.0f*gamma;
      b[7] = 1.0f + c3 - 2.0f*gamma;
      c[7] = 0.0f;
    }

    int off = threadIdx.y*(256+8);
    loadDataIntoRegisters_contig<8,32>(tid,256,d,smem+off,rhs,(REAL)0.0);

    trid_warp<8>(a,b,c,d);

    incDataFromRegisters_contig<8,32>(tid,256,d,smem+off,u);

    rhs = rhs + 4*(NX+pad_total);  // increment pointers for next line
    u   = u   + 4*(NX+pad_total);

  }
}

//
// solves tridiagonal equations in y-direction
//

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_implicit2_y(int NX, int NY, int NZ, REAL dS,
                               REAL c1, REAL c2, REAL c3,
                               REAL* __restrict__ u     ) {

  __shared__ REAL s1[33*COLS], s2[33*COLS];

  REAL S, lambda, gamma, a[8], b[8], c[8], d[8];
  int  i, j, k, tid, ind1, ind2;

  tid   = threadIdx.x + threadIdx.y*COLS;
  ind1  = tid + (tid/32);
  ind2  = (tid/32) + (tid%32)*COLS;
  ind2 += ind2 / 32;

  i =   threadIdx.x;
  j = 8*threadIdx.y;
  k =   blockIdx.x;

  u = u + i + pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);

  for (i=threadIdx.x; i<NX; i=i+COLS) {
    for (int n=0; n<8; n++) {
      S      = (j+n) * dS;
      lambda = c1*S*S;
      gamma  = c2*S;

      a[n] =              - ( lambda - gamma );
      b[n] = 1.0f      + 2.0f*lambda;
      c[n] =              - ( lambda + gamma );

      d[n] = u[n*(NX+pad_total)];
    }

    if (threadIdx.y==31) {
      a[7] =           + 2.0f*gamma;
      b[7] = 1.0f      - 2.0f*gamma;
      c[7] = 0.0f;
    }

    trid_warp_part1<8>(a,b,c,d);

    s1[ind1] = a[0]; s2[ind1] = a[7];
    __syncthreads();
    a[0] = s1[ind2]; a[7] = s2[ind2];
    __syncthreads();
    s1[ind1] = c[0]; s2[ind1] = c[7];
    __syncthreads();
    c[0] = s1[ind2]; c[7] = s2[ind2];
    __syncthreads();
    s1[ind1] = d[0]; s2[ind1] = d[7];
    __syncthreads();
    d[0] = s1[ind2]; d[7] = s2[ind2];

    trid2_warp(a[0],c[0],d[0],a[7],c[7],d[7]);

    s1[ind2] = d[0]; s2[ind2] = d[7];
    __syncthreads();
    d[0] = s1[ind1]; d[7] = s2[ind1];

    for (int n=1; n<7; n++) d[n] = d[n] - a[n]*d[0] - c[n]*d[7];

    for (int n=0; n<8; n++) u[n*(NX+pad_total)] = d[n];

    u = u + COLS;  // increment pointers for next lines
  }
}

//
// similar to BS_implicit2_y but solving in z-direction
//

template <int pad_left, int pad_total, typename REAL>
__global__ void BS_implicit2_z(int NX, int NY, int NZ, REAL dS,
                               REAL c1, REAL c2, REAL c3,
                               REAL* __restrict__ u     ) {

  __shared__ REAL s1[33*COLS], s2[33*COLS];

  REAL S, lambda, gamma, a[8], b[8], c[8], d[8];
  int  i, j, k, tid, ind1, ind2;

  tid   = threadIdx.x + threadIdx.y*COLS;
  ind1  = tid + (tid/32);
  ind2  = (tid/32) + (tid%32)*COLS;
  ind2 += ind2 / 32;

  i =   threadIdx.x;
  j =   blockIdx.x;  // swapping j, k in these two lines
  k = 8*threadIdx.y; // is one difference from implicit2_y

  u = u + i + pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);

  for (i=threadIdx.x; i<NX; i=i+COLS) {
    for (int n=0; n<8; n++) {
      S      = (k+n) * dS;      // changing j to k here is another
      lambda = c1*S*S;
      gamma  = c2*S;

      a[n] =              - ( lambda - gamma );
      b[n] = 1.0f      + 2.0f*lambda;
      c[n] =              - ( lambda + gamma );

      d[n] = u[n*(NX+pad_total)*(NY+2)];   // and a different offset here ...
    }

    if (threadIdx.y==31) {
      a[7] =           + 2.0f*gamma;
      b[7] = 1.0f      - 2.0f*gamma;
      c[7] = 0.0f;
    }

    trid_warp_part1<8>(a,b,c,d);

    s1[ind1] = a[0]; s2[ind1] = a[7];
    __syncthreads();
    a[0] = s1[ind2]; a[7] = s2[ind2];
    __syncthreads();
    s1[ind1] = c[0]; s2[ind1] = c[7];
    __syncthreads();
    c[0] = s1[ind2]; c[7] = s2[ind2];
    __syncthreads();
    s1[ind1] = d[0]; s2[ind1] = d[7];
    __syncthreads();
    d[0] = s1[ind2]; d[7] = s2[ind2];

    trid2_warp(a[0],c[0],d[0],a[7],c[7],d[7]);

    s1[ind2] = d[0]; s2[ind2] = d[7];
    __syncthreads();
    d[0] = s1[ind1]; d[7] = s2[ind1];

    for (int n=1; n<7; n++) d[n] = d[n] - a[n]*d[0] - c[n]*d[7];

    for (int n=0; n<8; n++) u[n*(NX+pad_total)*(NY+2)] = d[n]; // ... and here

    u = u + COLS;  // increment pointers for next lines
  }
}



////////////////////////////////////////////////////////////////////
//                                                                //
// main code to test all solvers for single & double precision    //
//                                                                //
////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  int    NX=256, NY=256, NZ=256, N, imid;
  float  *u_h, *u1_d, *u2_d, *foo_d;
  double *U_h, *U1_d, *U2_d, *Foo_d, val, err;
  int pad_left, pad_total;

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory for arrays

  int prod = (NX+32)*(NY+2)*(NZ+2)+2;
  u_h = (float *)malloc(prod*sizeof(float));
  // U_h = (double *)malloc(prod*sizeof(double));
  cudaSafeCall(cudaMalloc((void **)&u1_d, (prod+1)*sizeof(float)));
  // cudaSafeCall(cudaMalloc((void **)&U1_d, (prod+1)*sizeof(double)));
  cudaSafeCall(cudaMalloc((void **)&u2_d, (prod+1)*sizeof(float)));
  // cudaSafeCall(cudaMalloc((void **)&U2_d, (prod+1)*sizeof(double)));

  // execute kernels

  for (int prec=0; prec<1; prec++) {
    if (prec==0) {
      printf("\nsingle precision performance tests \n");
      cudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
    }
    else {
      printf("\ndouble precision performance tests \n");
      cudaSafeCall(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    }

    printf("---------------------------------- \n");
    printf(" method    exec time   GFinsts    GFlops       value at strike \n");

    for (int pass=0; pass<4; pass++) {
      pad_left  = 32;
      pad_total = 32;

      if (pass<3) {
        N = 500;
        cudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
      }
      else {
        N = 100;
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_x<32,32,double>, cudaFuncCachePreferShared));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_y<32,32,double>, cudaFuncCachePreferL1));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_z<32,32,double>, cudaFuncCachePreferL1));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_rhs<32,32,double>, cudaFuncCachePreferL1));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_x<32,32,float>, cudaFuncCachePreferShared));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_y<32,32,float>, cudaFuncCachePreferL1));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_z<32,32,float>, cudaFuncCachePreferL1));
        cudaSafeCall(cudaFuncSetCacheConfig(BS_implicit2_rhs<32,32,float>, cudaFuncCachePreferL1));
      }

      double Smax=200.0, K=100.0, r=0.05, sigma=0.2, T=0.05;
      double dS = Smax / 255.0;
      double dt = T / ( (double) N);
      double C1 = 0.5*dt*sigma*sigma / (dS*dS);
      double C2 = 0.5*dt*r / dS;
      double C3 = r*dt;

      float c1=C1, c2=C2, c3=C3, ds=dS;

  // initialise array (call on minimum of 3 assets) and copy over

      for (int i=-1; i<NX; i++) {
        for (int j=-1; j<NY; j++) {
          for (int k=-1; k<NZ; k++) {
             int indg = (i+pad_left) + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);
//             U_h[indg] = fmax(0.0, fmin(i*dS, fmin(j*dS,k*dS)) - K);
             U_h[indg] = fmax(0.0, i*dS-K);
             u_h[indg] = U_h[indg];
          }
        }
      }

      if (prec==0) {
        cudaSafeCall(cudaMemcpy(u1_d,u_h, prod*sizeof(float) ,cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(u2_d,u_h, prod*sizeof(float) ,cudaMemcpyHostToDevice));
      }
      else {
        cudaSafeCall(cudaMemcpy(U1_d,U_h, prod*sizeof(double),cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(U2_d,U_h, prod*sizeof(double),cudaMemcpyHostToDevice));
      }

  // now do main computation

      int  BLOCK_X = 64;
      int  BLOCK_Y = 4;

      int  bc_threads = BLOCK_X*BLOCK_Y;
      int  bc_blocks  = 1 + (NX*NY + NY*NZ + NZ*NX - 1) / bc_threads;

      int  bx = 1 + (NX-1)/BLOCK_X;
      int  by = 1 + (NY-1)/BLOCK_Y;

      if (pass==2) {
        BLOCK_X = 32;
        BLOCK_Y = 8;
        bx = 1 + (NX/2-1)/(BLOCK_X-2);
        by = 1 + (NY-1)/BLOCK_Y;
      }

      dim3 threads(BLOCK_X,BLOCK_Y);
      dim3 blocks(bx,by);

      cudaEventRecord(start);

      for (int n=1; n<=N; n++) {
        if (prec==0) {
          BS_bc1<32,32><<<bc_blocks, bc_threads>>>(NX,NY,NZ, u1_d);
          if (pass==0)
            BS_explicit1<32,32><<<blocks, threads>>>(NX,NY,NZ, ds, c1,c1,c1, c2,c2,c2,
                                                c3, 0.0f,0.0f,0.0f, u1_d, u2_d);
          else if (pass==1)
            BS_explicit2<32,32><<<blocks, threads>>>(NX,NY,NZ, ds, c1,c1,c1, c2,c2,c2,
                                                c3, 0.0f,0.0f,0.0f, u1_d, u2_d);
          else if (pass==2)
            BS_explicit3<32,32><<<blocks, threads>>>(NX,NY,NZ, ds, c1,c1,c1, c2,c2,c2,
                         c3, 0.0f,0.0f,0.0f, (float2*)(u1_d), (float2*)(u2_d));
          else if (pass==3) {
            BS_implicit2_rhs<32,32><<<blocks, threads>>>(NX,NY,NZ, ds, c1,c1,c1, c2,c2,c2,
                                                   c3, 0.0f,0.0f,0.0f, u1_d, u2_d);
            BS_implicit2_y<32,32><<<NZ, dim3(COLS,32)>>>(NX,NY,NZ, ds, c1,c2,c3,    u2_d);
            BS_implicit2_z<32,32><<<NY, dim3(COLS,32)>>>(NX,NY,NZ, ds, c1,c2,c3,    u2_d);
            BS_implicit2_x<32,32><<<NZ, dim3(32,4)>>>(NX,NY,NZ, ds, c1,c2,c3, u1_d, u2_d);
          }
          if (pass<3) {foo_d=u1_d; u1_d=u2_d; u2_d=foo_d;}   // swap u1, u2 pointers
        }
        else {
          BS_bc1<32,32><<<bc_blocks, bc_threads>>>(NX,NY,NZ, U1_d);
          if (pass==0)
            BS_explicit1<32,32><<<blocks, threads>>>(NX,NY,NZ, dS, C1,C1,C1, C2,C2,C2,
                                                   C3, 0.0,0.0,0.0, U1_d, U2_d);
          else if (pass==1)
            BS_explicit2<32,32><<<blocks, threads>>>(NX,NY,NZ, dS, C1,C1,C1, C2,C2,C2,
                                                   C3, 0.0,0.0,0.0, U1_d, U2_d);
          else if (pass==2)
            BS_explicit3<32,32><<<blocks, threads>>>(NX,NY,NZ, dS, C1,C1,C1, C2,C2,C2,
                          C3, 0.0,0.0,0.0, (double2*)(U1_d), (double2*)(U2_d));
          else if (pass==3) {
            BS_implicit2_rhs<32,32><<<blocks, threads>>>(NX,NY,NZ, dS, C1,C1,C1, C2,C2,C2,
                                                      C3, 0.0,0.0,0.0, U1_d, U2_d);
            BS_implicit2_y<32,32><<<NZ, dim3(COLS,32)>>>(NX,NY,NZ, dS, C1,C2,C3,    U2_d);
            BS_implicit2_z<32,32><<<NY, dim3(COLS,32)>>>(NX,NY,NZ, dS, C1,C2,C3,    U2_d);
            BS_implicit2_x<32,32><<<NZ, dim3(32,4)>>>(NX,NY,NZ, dS, C1,C2,C3, U1_d, U2_d);
          }
          if (pass<3) {Foo_d=U1_d; U1_d=U2_d; U2_d=Foo_d;}   // swap U1, U2 pointers
        }
      }

      cudaSafeCall(cudaEventRecord(stop));
      cudaSafeCall(cudaEventSynchronize(stop));
      cudaSafeCall(cudaEventElapsedTime(&milli, start, stop));

//      imid = (NX/2+1) + (NY/2+1)*(NX+2) + (NZ/2+1)*(NX+2)*(NY+2);
      imid = (NX/2+pad_left) + (NY/2+1)*(NX+pad_total) + (NZ/2+1)*(NX+pad_total)*(NY+2);

      if (prec==0) {
        cudaSafeCall(cudaMemcpy(u_h,u1_d,prod*sizeof(float), cudaMemcpyDeviceToHost));

        for (int i=0; i<NX; i++) {
          val = u_h[i+pad_left+(NX+pad_total)+(NX+pad_total)*(NY+2)];
          err = 0.0;
          for (int j=0; j<NY; j++) {
            for (int k=0; k<NZ; k++) {
              int ind = i+pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);
              err = fmax(err,fabs(val-u_h[ind]));
//              if (i==NX/2 && k==NX/2) printf(" %d  %f \n",j,u_h[ind]-u_h[imid]);
            }
          }
          if (err > 1e-2) printf(" %d  %f \n",i,err);
        }

        val = u_h[imid];
      }
      else {
        cudaSafeCall(cudaMemcpy(U_h,U1_d,prod*sizeof(double), cudaMemcpyDeviceToHost));

        for (int i=0; i<NX; i++) {
          val = u_h[i+pad_left+(NX+pad_total)+(NX+pad_total)*(NY+2)];
          err = 0.0;
          for (int j=0; j<NY; j++) {
            for (int k=0; k<NZ; k++) {
              int ind = i+pad_left + (j+1)*(NX+pad_total) + (k+1)*(NX+pad_total)*(NY+2);
              err = fmax(err,fabs(val-u_h[ind]));
            }
          }
          if (err > 1e-8) printf(" %d  %f \n",i,err);
        }

        val = U_h[imid];
      }

      if (pass<3)
        printf("explicit%d %9.0f %38.6f \n",pass+1,milli,val);
      else
        printf("implicit%d %9.0f %38.6f \n",pass-1,milli,val);
    }
  }

// CUDA exit -- needed to flush printf write buffer

  cudaSafeCall(cudaThreadSynchronize());
  cudaSafeCall(cudaDeviceReset());
  return 0;
}
