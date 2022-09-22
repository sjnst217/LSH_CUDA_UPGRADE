
#include "LSH_CUDA_UPGRADE.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROTL(x,r)   ((x) << (r)) | ((x) >> (32-r)) 
#define ROTR(x,r)   ((x) >> (r)) | ((x) << (32-r)) 

__device__ static const UINT g_StepConstants[208] = {
   0x917caf90, 0x6c1b10a2, 0x6f352943, 0xcf778243, 0x2ceb7472, 0x29e96ff2, 0x8a9ba428, 0x2eeb2642,
   0x0e2c4021, 0x872bb30e, 0xa45e6cb2, 0x46f9c612, 0x185fe69e, 0x1359621b, 0x263fccb2, 0x1a116870,
   0x3a6c612f, 0xb2dec195, 0x02cb1f56, 0x40bfd858, 0x784684b6, 0x6cbb7d2e, 0x660c7ed8, 0x2b79d88a,
   0xa6cd9069, 0x91a05747, 0xcdea7558, 0x00983098, 0xbecb3b2e, 0x2838ab9a, 0x728b573e, 0xa55262b5,
   0x745dfa0f, 0x31f79ed8, 0xb85fce25, 0x98c8c898, 0x8a0669ec, 0x60e445c2, 0xfde295b0, 0xf7b5185a,
   0xd2580983, 0x29967709, 0x182df3dd, 0x61916130, 0x90705676, 0x452a0822, 0xe07846ad, 0xaccd7351,
   0x2a618d55, 0xc00d8032, 0x4621d0f5, 0xf2f29191, 0x00c6cd06, 0x6f322a67, 0x58bef48d, 0x7a40c4fd,
   0x8beee27f, 0xcd8db2f2, 0x67f2c63b, 0xe5842383, 0xc793d306, 0xa15c91d6, 0x17b381e5, 0xbb05c277,
   0x7ad1620a, 0x5b40a5bf, 0x5ab901a2, 0x69a7a768, 0x5b66d9cd, 0xfdee6877, 0xcb3566fc, 0xc0c83a32,
   0x4c336c84, 0x9be6651a, 0x13baa3fc, 0x114f0fd1, 0xc240a728, 0xec56e074, 0x009c63c7, 0x89026cf2,
   0x7f9ff0d0, 0x824b7fb5, 0xce5ea00f, 0x605ee0e2, 0x02e7cfea, 0x43375560, 0x9d002ac7, 0x8b6f5f7b,
   0x1f90c14f, 0xcdcb3537, 0x2cfeafdd, 0xbf3fc342, 0xeab7b9ec, 0x7a8cb5a3, 0x9d2af264, 0xfacedb06,
   0xb052106e, 0x99006d04, 0x2bae8d09, 0xff030601, 0xa271a6d6, 0x0742591d, 0xc81d5701, 0xc9a9e200,
   0x02627f1e, 0x996d719d, 0xda3b9634, 0x02090800, 0x14187d78, 0x499b7624, 0xe57458c9, 0x738be2c9,
   0x64e19d20, 0x06df0f36, 0x15d1cb0e, 0x0b110802, 0x2c95f58c, 0xe5119a6d, 0x59cd22ae, 0xff6eac3c,
   0x467ebd84, 0xe5ee453c, 0xe79cd923, 0x1c190a0d, 0xc28b81b8, 0xf6ac0852, 0x26efd107, 0x6e1ae93b,
   0xc53c41ca, 0xd4338221, 0x8475fd0a, 0x35231729, 0x4e0d3a7a, 0xa2b45b48, 0x16c0d82d, 0x890424a9,
   0x017e0c8f, 0x07b5a3f5, 0xfa73078e, 0x583a405e, 0x5b47b4c8, 0x570fa3ea, 0xd7990543, 0x8d28ce32,
   0x7f8a9b90, 0xbd5998fc, 0x6d7a9688, 0x927a9eb6, 0xa2fc7d23, 0x66b38e41, 0x709e491a, 0xb5f700bf,
   0x0a262c0f, 0x16f295b9, 0xe8111ef5, 0x0d195548, 0x9f79a0c5, 0x1a41cfa7, 0x0ee7638a, 0xacf7c074,
   0x30523b19, 0x09884ecf, 0xf93014dd, 0x266e9d55, 0x191a6664, 0x5c1176c1, 0xf64aed98, 0xa4b83520,
   0x828d5449, 0x91d71dd8, 0x2944f2d6, 0x950bf27b, 0x3380ca7d, 0x6d88381d, 0x4138868e, 0x5ced55c4,
   0x0fe19dcb, 0x68f4f669, 0x6e37c8ff, 0xa0fe6e10, 0xb44b47b0, 0xf5c0558a, 0x79bf14cf, 0x4a431a20,
   0xf17f68da, 0x5deb5fd1, 0xa600c86d, 0x9f6c7eb0, 0xff92f864, 0xb615e07f, 0x38d3e448, 0x8d5d3a6a,
   0x70e843cb, 0x494b312e, 0xa6c93613, 0x0beb2f4f, 0x928b5d63, 0xcbf66035, 0x0cb82c80, 0xea97a4f7,
   0x592c0f3b, 0x947c5f77, 0x6fff49b9, 0xf71a7e5a, 0x1de8c0f5, 0xc2569600, 0xc4e4ac8c, 0x823c9ce1
};

__device__ static inline void load_msg_blk(LSH_internal* i_state, const UINT* msgblk)
{
    i_state->submsg_e_l[0] = (msgblk[0]);
    i_state->submsg_e_l[1] = (msgblk[1]);
    i_state->submsg_e_l[2] = (msgblk[2]);
    i_state->submsg_e_l[3] = (msgblk[3]);
    i_state->submsg_e_l[4] = (msgblk[4]);
    i_state->submsg_e_l[5] = (msgblk[5]);
    i_state->submsg_e_l[6] = (msgblk[6]);
    i_state->submsg_e_l[7] = (msgblk[7]);
    i_state->submsg_e_r[0] = (msgblk[8]);
    i_state->submsg_e_r[1] = (msgblk[9]);
    i_state->submsg_e_r[2] = (msgblk[10]);
    i_state->submsg_e_r[3] = (msgblk[11]);
    i_state->submsg_e_r[4] = (msgblk[12]);
    i_state->submsg_e_r[5] = (msgblk[13]);
    i_state->submsg_e_r[6] = (msgblk[14]);
    i_state->submsg_e_r[7] = (msgblk[15]);
    i_state->submsg_o_l[0] = (msgblk[16]);
    i_state->submsg_o_l[1] = (msgblk[17]);
    i_state->submsg_o_l[2] = (msgblk[18]);
    i_state->submsg_o_l[3] = (msgblk[19]);
    i_state->submsg_o_l[4] = (msgblk[20]);
    i_state->submsg_o_l[5] = (msgblk[21]);
    i_state->submsg_o_l[6] = (msgblk[22]);
    i_state->submsg_o_l[7] = (msgblk[23]);
    i_state->submsg_o_r[0] = (msgblk[24]);
    i_state->submsg_o_r[1] = (msgblk[25]);
    i_state->submsg_o_r[2] = (msgblk[26]);
    i_state->submsg_o_r[3] = (msgblk[27]);
    i_state->submsg_o_r[4] = (msgblk[28]);
    i_state->submsg_o_r[5] = (msgblk[29]);
    i_state->submsg_o_r[6] = (msgblk[30]);
    i_state->submsg_o_r[7] = (msgblk[31]);
}

__device__ static void msg_exp_even(LSH_internal* i_state)
{
    UINT temp;
    temp = i_state->submsg_e_l[0];
    i_state->submsg_e_l[0] = i_state->submsg_o_l[0] + i_state->submsg_e_l[3];
    i_state->submsg_e_l[3] = i_state->submsg_o_l[3] + i_state->submsg_e_l[1];
    i_state->submsg_e_l[1] = i_state->submsg_o_l[1] + i_state->submsg_e_l[2];
    i_state->submsg_e_l[2] = i_state->submsg_o_l[2] + temp;
    temp = i_state->submsg_e_l[4];
    i_state->submsg_e_l[4] = i_state->submsg_o_l[4] + i_state->submsg_e_l[7];
    i_state->submsg_e_l[7] = i_state->submsg_o_l[7] + i_state->submsg_e_l[6];
    i_state->submsg_e_l[6] = i_state->submsg_o_l[6] + i_state->submsg_e_l[5];
    i_state->submsg_e_l[5] = i_state->submsg_o_l[5] + temp;
    temp = i_state->submsg_e_r[0];
    i_state->submsg_e_r[0] = i_state->submsg_o_r[0] + i_state->submsg_e_r[3];
    i_state->submsg_e_r[3] = i_state->submsg_o_r[3] + i_state->submsg_e_r[1];
    i_state->submsg_e_r[1] = i_state->submsg_o_r[1] + i_state->submsg_e_r[2];
    i_state->submsg_e_r[2] = i_state->submsg_o_r[2] + temp;
    temp = i_state->submsg_e_r[4];
    i_state->submsg_e_r[4] = i_state->submsg_o_r[4] + i_state->submsg_e_r[7];
    i_state->submsg_e_r[7] = i_state->submsg_o_r[7] + i_state->submsg_e_r[6];
    i_state->submsg_e_r[6] = i_state->submsg_o_r[6] + i_state->submsg_e_r[5];
    i_state->submsg_e_r[5] = i_state->submsg_o_r[5] + temp;
}

__device__ static void msg_exp_odd(LSH_internal* i_state)
{
    UINT temp;
    temp = i_state->submsg_o_l[0];
    i_state->submsg_o_l[0] = i_state->submsg_e_l[0] + i_state->submsg_o_l[3];
    i_state->submsg_o_l[3] = i_state->submsg_e_l[3] + i_state->submsg_o_l[1];
    i_state->submsg_o_l[1] = i_state->submsg_e_l[1] + i_state->submsg_o_l[2];
    i_state->submsg_o_l[2] = i_state->submsg_e_l[2] + temp;
    temp = i_state->submsg_o_l[4];
    i_state->submsg_o_l[4] = i_state->submsg_e_l[4] + i_state->submsg_o_l[7];
    i_state->submsg_o_l[7] = i_state->submsg_e_l[7] + i_state->submsg_o_l[6];
    i_state->submsg_o_l[6] = i_state->submsg_e_l[6] + i_state->submsg_o_l[5];
    i_state->submsg_o_l[5] = i_state->submsg_e_l[5] + temp;
    temp = i_state->submsg_o_r[0];
    i_state->submsg_o_r[0] = i_state->submsg_e_r[0] + i_state->submsg_o_r[3];
    i_state->submsg_o_r[3] = i_state->submsg_e_r[3] + i_state->submsg_o_r[1];
    i_state->submsg_o_r[1] = i_state->submsg_e_r[1] + i_state->submsg_o_r[2];
    i_state->submsg_o_r[2] = i_state->submsg_e_r[2] + temp;
    temp = i_state->submsg_o_r[4];
    i_state->submsg_o_r[4] = i_state->submsg_e_r[4] + i_state->submsg_o_r[7];
    i_state->submsg_o_r[7] = i_state->submsg_e_r[7] + i_state->submsg_o_r[6];
    i_state->submsg_o_r[6] = i_state->submsg_e_r[6] + i_state->submsg_o_r[5];
    i_state->submsg_o_r[5] = i_state->submsg_e_r[5] + temp;
}

__device__ static inline void load_sc(const UINT** p_const_v, UINT i)
{
    *p_const_v = &g_StepConstants[i];
}

__device__ static void msg_add_even(UINT* cv_l, UINT* cv_r, LSH_internal* i_state)
{
    cv_l[0] ^= i_state->submsg_e_l[0]; cv_l[1] ^= i_state->submsg_e_l[1]; cv_l[2] ^= i_state->submsg_e_l[2]; cv_l[3] ^= i_state->submsg_e_l[3];
    cv_l[4] ^= i_state->submsg_e_l[4]; cv_l[5] ^= i_state->submsg_e_l[5]; cv_l[6] ^= i_state->submsg_e_l[6]; cv_l[7] ^= i_state->submsg_e_l[7];
    cv_r[0] ^= i_state->submsg_e_r[0]; cv_r[1] ^= i_state->submsg_e_r[1]; cv_r[2] ^= i_state->submsg_e_r[2]; cv_r[3] ^= i_state->submsg_e_r[3];
    cv_r[4] ^= i_state->submsg_e_r[4]; cv_r[5] ^= i_state->submsg_e_r[5]; cv_r[6] ^= i_state->submsg_e_r[6]; cv_r[7] ^= i_state->submsg_e_r[7];
}
__device__ static void msg_add_odd(UINT* cv_l, UINT* cv_r, LSH_internal* i_state)
{
    cv_l[0] ^= i_state->submsg_o_l[0]; cv_l[1] ^= i_state->submsg_o_l[1]; cv_l[2] ^= i_state->submsg_o_l[2]; cv_l[3] ^= i_state->submsg_o_l[3];
    cv_l[4] ^= i_state->submsg_o_l[4]; cv_l[5] ^= i_state->submsg_o_l[5]; cv_l[6] ^= i_state->submsg_o_l[6]; cv_l[7] ^= i_state->submsg_o_l[7];
    cv_r[0] ^= i_state->submsg_o_r[0]; cv_r[1] ^= i_state->submsg_o_r[1]; cv_r[2] ^= i_state->submsg_o_r[2]; cv_r[3] ^= i_state->submsg_o_r[3];
    cv_r[4] ^= i_state->submsg_o_r[4]; cv_r[5] ^= i_state->submsg_o_r[5]; cv_r[6] ^= i_state->submsg_o_r[6]; cv_r[7] ^= i_state->submsg_o_r[7];
}

__device__ static void add_blk(UINT* cv_l, const UINT* cv_r)
{
    cv_l[0] += cv_r[0];
    cv_l[1] += cv_r[1];
    cv_l[2] += cv_r[2];
    cv_l[3] += cv_r[3];
    cv_l[4] += cv_r[4];
    cv_l[5] += cv_r[5];
    cv_l[6] += cv_r[6];
    cv_l[7] += cv_r[7];
}

__device__ static void rotate_blk(UINT cv[8], const int rot_value)
{
    cv[0] = ROTL(cv[0], rot_value);
    cv[1] = ROTL(cv[1], rot_value);
    cv[2] = ROTL(cv[2], rot_value);
    cv[3] = ROTL(cv[3], rot_value);
    cv[4] = ROTL(cv[4], rot_value);
    cv[5] = ROTL(cv[5], rot_value);
    cv[6] = ROTL(cv[6], rot_value);
    cv[7] = ROTL(cv[7], rot_value);
}

__device__ static void  xor_with_const(UINT* cv_l, const UINT* const_v)
{
    cv_l[0] ^= const_v[0];
    cv_l[1] ^= const_v[1];
    cv_l[2] ^= const_v[2];
    cv_l[3] ^= const_v[3];
    cv_l[4] ^= const_v[4];
    cv_l[5] ^= const_v[5];
    cv_l[6] ^= const_v[6];
    cv_l[7] ^= const_v[7];
}

__device__ static void rotate_msg_gamma(UINT* cv_r)
{
    cv_r[1] = ROTL(cv_r[1], 8);
    cv_r[2] = ROTL(cv_r[2], 16);
    cv_r[3] = ROTL(cv_r[3], 24);
    cv_r[4] = ROTL(cv_r[4], 24);
    cv_r[5] = ROTL(cv_r[5], 16);
    cv_r[6] = ROTL(cv_r[6], 8);
}

__device__ static void word_perm(UINT* cv_l, UINT* cv_r)
{
    UINT temp;
    temp = cv_l[0];
    cv_l[0] = cv_l[6];
    cv_l[6] = cv_r[6];
    cv_r[6] = cv_r[2];
    cv_r[2] = cv_l[1];
    cv_l[1] = cv_l[4];
    cv_l[4] = cv_r[4];
    cv_r[4] = cv_r[0];
    cv_r[0] = cv_l[2];
    cv_l[2] = cv_l[5];
    cv_l[5] = cv_r[7];
    cv_r[7] = cv_r[1];
    cv_r[1] = temp;
    temp = cv_l[3];
    cv_l[3] = cv_l[7];
    cv_l[7] = cv_r[5];
    cv_r[5] = cv_r[3];
    cv_r[3] = temp;
};

__device__ static void mix(UINT* cv_l, UINT* cv_r, const UINT* const_v, const int rot_alpha, const int rot_beta)
{
    add_blk(cv_l, cv_r);
    rotate_blk(cv_l, rot_alpha);
    xor_with_const(cv_l, const_v);
    add_blk(cv_r, cv_l);
    rotate_blk(cv_r, rot_beta);
    add_blk(cv_l, cv_r);
    rotate_msg_gamma(cv_r);
}

__device__ static void initial_vector(LSH_Info* ctx)
{
    ctx->uChainVar_left[0] = 0x46a10f1f;
    ctx->uChainVar_left[1] = 0xfddce486;
    ctx->uChainVar_left[2] = 0xb41443a8;
    ctx->uChainVar_left[3] = 0x198e6b9d;
    ctx->uChainVar_left[4] = 0x3304388d;
    ctx->uChainVar_left[5] = 0xb0f5a3c7;
    ctx->uChainVar_left[6] = 0xb36061c4;
    ctx->uChainVar_left[7] = 0x7adbd553;
    ctx->uChainVar_right[0] = 0x105d5378;
    ctx->uChainVar_right[1] = 0x2f74de54;
    ctx->uChainVar_right[2] = 0x5c2f2d95;
    ctx->uChainVar_right[3] = 0xf2553fbe;
    ctx->uChainVar_right[4] = 0x8051357a;
    ctx->uChainVar_right[5] = 0x138668c8;
    ctx->uChainVar_right[6] = 0x47aa4484;
    ctx->uChainVar_right[7] = 0xe01afb41;
}

__device__ static void final(LSH_Info* Info)
{
    UINT i;
    for (i = 0; i < 8; i++)
    {
        Info->uChainVar_left[i] = Info->uChainVar_left[i] ^ Info->uChainVar_right[i];
    }
}

__device__ void LSH_Compress(LSH_Info* ctx, UINT_PTR sv_pt)
{
    UINT i;
    LSH_internal i_state[1];

    const UINT* const_v = NULL;
    UINT* cv_l = ctx->uChainVar_left;
    UINT* cv_r = ctx->uChainVar_right;

    load_msg_blk(i_state, sv_pt);

    msg_add_even(cv_l, cv_r, i_state);
    load_sc(&const_v, 0);
    mix(cv_l, cv_r, const_v, 29, 1);
    word_perm(cv_l, cv_r);

    msg_add_odd(cv_l, cv_r, i_state);
    load_sc(&const_v, 8);
    mix(cv_l, cv_r, const_v, 5, 17);
    word_perm(cv_l, cv_r);

    for (i = 1; i < 26 / 2; i++)
    {
        msg_exp_even(i_state);
        msg_add_even(cv_l, cv_r, i_state);
        load_sc(&const_v, 16 * i);
        mix(cv_l, cv_r, const_v, 29, 1);
        word_perm(cv_l, cv_r);

        msg_exp_odd(i_state);
        msg_add_odd(cv_l, cv_r, i_state);
        load_sc(&const_v, 16 * i + 8);
        mix(cv_l, cv_r, const_v, 5, 17);
        word_perm(cv_l, cv_r);
    }

    msg_exp_even(i_state);
    msg_add_even(cv_l, cv_r, i_state);
}


__device__ void LSH_Init(LSH_Info* Info)
{
    Info->remain_byte_len = 0;

    initial_vector(Info);

    return;
}

__device__ void LSH_update(LSH_Info* Info, const BYTE* pt, UINT pt_byte_len)
{
    UINT i = 0, t = 0;

    UINT remain_pt_byte;
    UINT pt_len = pt_byte_len;

    BYTE TEST_SV_PT[TEST_PT_SIZE] = { 0 };

    if (pt_byte_len == 0)
    {
        return;
    }

    for (int i = 0; i < TEST_PT_SIZE; i++)
    {
        TEST_SV_PT[i] = pt[i * blockDim.x * gridDim.x];
    }

    remain_pt_byte = Info->remain_byte_len;

    if (pt_len + remain_pt_byte < LSH_BLOCK_LEN)
    {
        memcpy((UCHAR_PTR)Info->sv_last_pt + remain_pt_byte, TEST_SV_PT, pt_len);
        Info->remain_byte_len += (UINT)pt_byte_len;
        return;
    }

    while (pt_len + remain_pt_byte >= LSH_BLOCK_LEN)
    {
        memcpy((UCHAR_PTR)(Info->sv_pt), TEST_SV_PT + i * LSH_BLOCK_LEN, (int)LSH_BLOCK_LEN);
        LSH_Compress(Info, (UINT_PTR)Info->sv_pt);

        i++;
        pt_len -= (LSH_BLOCK_LEN - remain_pt_byte);
        remain_pt_byte = 0;
    }

    memcpy((UCHAR_PTR)Info->sv_last_pt, TEST_SV_PT + i * LSH_BLOCK_LEN, pt_len);
    Info->remain_byte_len = (UINT)pt_len;

    return;
}

__device__ void LSH_final(LSH_Info* Info, BYTE* sv_hashval)
{
    UINT remain_pt_byte;

    remain_pt_byte = Info->remain_byte_len;

    Info->sv_last_pt[remain_pt_byte] = 0x80;

    memset(Info->sv_last_pt + remain_pt_byte + 1, 0, LSH_BLOCK_LEN - remain_pt_byte - 1);

    LSH_Compress(Info, (UINT_PTR)Info->sv_last_pt);

    final(Info);

    memcpy(sv_hashval, Info->uChainVar_left, sizeof(BYTE) * 32);

    memset(Info, 0, sizeof(LSH_Info));

    return;
}

__global__ void make_hash_val(LSH_Info* Info, BYTE* pt, BYTE* sv_hashval)
{
    int tid;
    tid = threadIdx.x + blockIdx.x * blockDim.x;

    LSH_Info us_Info[1];
    memcpy(us_Info, Info, sizeof(LSH_Info));

    LSH_Init(us_Info);
    LSH_update(us_Info, pt + tid, TEST_PT_SIZE);
    LSH_final(us_Info, sv_hashval + (tid * LSH_HASH_LEN));
}

void test_LSH_GPU(ULL Blocksize, ULL Threadsize)
{
    LSH_Info* info = NULL;
    BYTE* test_pt = NULL;
    BYTE* sv_hashval = NULL;
    BYTE* us_cpu_pt = NULL;

    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;

    info = (LSH_Info*)malloc(sizeof(LSH_Info));
    test_pt = (BYTE*)malloc(sizeof(BYTE) * TEST_PT_SIZE * Blocksize * Threadsize);
    sv_hashval = (BYTE*)malloc(sizeof(BYTE) * Blocksize * Threadsize * LSH_HASH_LEN);
    us_cpu_pt = (BYTE*)malloc(sizeof(BYTE) * TEST_PT_SIZE * Blocksize * Threadsize);

    int i, k = 0;

    for (i = 0; i < Blocksize * Threadsize; i++)
    {
        for (int j = 0; j < TEST_PT_SIZE; j++)
        {
            test_pt[TEST_PT_SIZE * i + j] = BYTE(j);
        }
    }

    BYTE* GPU_pt;
    BYTE* GPU_sv_hashval;
    LSH_Info* GPU_info;

    cudaMalloc((void**)&GPU_pt, sizeof(BYTE) * TEST_PT_SIZE * Blocksize * Threadsize);
    cudaMalloc((void**)&GPU_sv_hashval, sizeof(BYTE) * Blocksize * Threadsize * LSH_HASH_LEN);
    cudaMalloc((void**)&GPU_info, sizeof(LSH_Info));

    for (i = 0; i < TEST_PT_SIZE; i++)
    {
        for (int j = 0; j < Blocksize * Threadsize; j++)
        {
            us_cpu_pt[k++] = test_pt[TEST_PT_SIZE * j + i];
        }
    }
    k = 0;

    printf("\n\nStart...\n");
    cudaMemcpy(GPU_pt, us_cpu_pt, sizeof(BYTE) * TEST_PT_SIZE * Blocksize * Threadsize, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_info, info, sizeof(LSH_Info), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int x = 0; x < 1000; x++) {
        make_hash_val << <Blocksize, Threadsize >> > (GPU_info, GPU_pt, GPU_sv_hashval);
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaMemcpy(sv_hashval, GPU_sv_hashval, sizeof(BYTE) * Blocksize * Threadsize * LSH_HASH_LEN, cudaMemcpyDeviceToHost);
    elapsed_time_ms /= 1000;
    elapsed_time_ms = (Blocksize * Threadsize * LSH_BLOCK_LEN * sizeof(BYTE)) / elapsed_time_ms;
    elapsed_time_ms *= 1000;
    elapsed_time_ms /= (1024 * 1024 * 1024);
    printf("File size = %lld MB, Grid : %ld, Block : %ld, Performance : %4.2f GB/s\n", (Blocksize * Threadsize * LSH_BLOCK_LEN) / (1024 * 1024), Blocksize, Threadsize, elapsed_time_ms);
    getchar();
    getchar();

    cudaGetLastError();
    cudaDeviceSynchronize();


    printf("LSH_HAST_VAL : \n");
    for (i = 0; i < Blocksize * Threadsize * LSH_HASH_LEN; i++)
    {
        if (i % 32 == 0)
        {
            printf("\n%d¹øÂ° hash°ª\n", k + 1);
        }
        printf(" %02X", sv_hashval[i]);
        if ((i + 1) % 8 == 0)
        {
            printf("\n");
        }
        if ((i + 1) % 32 == 0)
        {
            printf("\n");
            k++;
        }
    }

    return;
}

int main()
{
    ULL Blocksize = 1024, Threadsize = 128;

    test_LSH_GPU(Blocksize, Threadsize);

    return LSH_SUCCESS;
}