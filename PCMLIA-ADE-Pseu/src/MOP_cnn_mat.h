// ������ļ���Ҫ���ڹ��ڶ�ά��������Ĳ���
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_NN_FLT_TYPE.h"

#define full 0
#define same 1
#define valid 2

typedef struct Mat2DSize {
	int c; // �У���
	int r; // �У��ߣ�
} nSize;

MY_FLT_TYPE** rotate180(MY_FLT_TYPE** mat, nSize matSize);// ����ת180��

void addmat(MY_FLT_TYPE** res, MY_FLT_TYPE** mat1, nSize matSize1, MY_FLT_TYPE** mat2, nSize matSize2);// �������

MY_FLT_TYPE** correlation(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type);// �����

MY_FLT_TYPE** cov(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type); // �������

// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr���ڲ屶��
MY_FLT_TYPE** UpSample(MY_FLT_TYPE** mat, nSize matSize, int upc, int upr);

// ����ά�����Ե��������addw��С��0ֵ��
MY_FLT_TYPE** matEdgeExpand(MY_FLT_TYPE** mat, nSize matSize, int addc, int addr);

// ����ά�����Ե��С������shrinkc��С�ı�
MY_FLT_TYPE** matEdgeShrink(MY_FLT_TYPE** mat, nSize matSize, int shrinkc, int shrinkr);

void savemat(MY_FLT_TYPE** mat, nSize matSize, const char* filename);// �����������

void multifactor(MY_FLT_TYPE** res, MY_FLT_TYPE** mat, nSize matSize, MY_FLT_TYPE factor);// �������ϵ��

MY_FLT_TYPE summat(MY_FLT_TYPE** mat, nSize matSize);// �����Ԫ�صĺ�

char* combine_strings(char* a, char* b);

char* intTochar(int i);

#endif
