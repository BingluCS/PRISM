#pragma once

#include <cstdint>
#define PRISM_TYPE_FLOAT F4
#define PRISM_TYPE_DOUBLE F8
#define BLOCK_SIZE 16

typedef uint8_t u1;
typedef uint16_t u2;
typedef uint32_t u4;
typedef uint64_t u8;
typedef unsigned long long ull;
typedef int8_t i1;
typedef int16_t i2;
typedef int32_t i4;
typedef int64_t i8;
typedef float f4;
typedef double f8;
typedef size_t szt;

typedef enum prism_dtype  //
{ __F0 = 0,
  F4 = 4,
  F8 = 8,
  __U0 = 10,
  U1 = 11,
  U2 = 12,
  U4 = 14,
  U8 = 18,
  __I0 = 20,
  I1 = 21,
  I2 = 22,
  I4 = 24,
  I8 = 28,
  ULL = 31 } prism_dtype;


typedef enum prism_mode  //
{ ABS = 0,
  REL = 1,
  PW_REL = 2} prism_mode;

typedef enum Input_type {
    ori_File,
    cmp_File
} itype;

typedef enum Bitplane_type{
    NB,
    SM,
    SA
} btype;