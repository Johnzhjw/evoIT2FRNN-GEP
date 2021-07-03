#ifndef __MINST_
#define __MINST_
/*
MINST���ݿ���һ����дͼ�����ݿ⣬����
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_NN_FLT_TYPE.h"

typedef struct MinstImg {
    int c;           // ͼ���
    int r;           // ͼ���
    MY_FLT_TYPE** ImgData; // ͼ�����ݶ�ά��̬����
} MinstImg;

typedef struct MinstImgArr {
    int ImgNum;        // �洢ͼ�����Ŀ
    MinstImg* ImgPtr;  // �洢ͼ������ָ��
}*ImgArr;              // �洢ͼ�����ݵ�����

typedef struct MinstLabel {
    int l;            // �����ǵĳ�
    MY_FLT_TYPE* LabelData; // ����������
} MinstLabel;

typedef struct MinstLabelArr {
    int LabelNum;
    MinstLabel* LabelPtr;
}*LabelArr;              // �洢ͼ���ǵ�����

LabelArr read_Label_IDX_FILE(const char* filename, int len_label); // ����ͼ����
LabelArr read_Label(const char* filename); // ����ͼ����
LabelArr read_Lable_tabel(const char* filetrain, const char* filetest,
                          int num_train, int num_test, int num_class); // ����ͼ����
void     free_Label(LabelArr arr_label);

ImgArr read_Img_IDX_FILE(const char* filename); // ����ͼ��
ImgArr read_Img(const char* filename); // ����ͼ��
ImgArr read_Img_tmp(const char* filename, int nSamp, int nCh, int tHeight, int tWidth); // ����ͼ��
ImgArr read_Img_table(const char* filetrain, const char* filetest,
                      int num_train, int num_test, int num_feature, int num_row, int num_col, int** pixel_index); // ����ͼ��
void   free_Img(ImgArr arr_img);
void interpolate_Img(ImgArr& imgarr, LabelArr& labarr, ImgArr allimgs, LabelArr alllabels,
                     int* num4neighbor, int maxSize_neighborhood, int* flag_train, int num_train, int* flag_class);

void save_Img(ImgArr imgarr, const char filedir[512]); // ��ͼ�����ݱ�����ļ�

typedef struct LabelIndex {
    int len;            // ĳ��ǵĸ�����
    int* IndexData; // index����
} LabelIndex;

typedef struct ArrLabelIndex {
    int LabelNum;
    LabelIndex* LabelIndexPtr;
} ArrLabelIndex;             // �洢ͼ���ǵ�����

typedef struct SetArrLabelIndex {
    int ArrNum;
    ArrLabelIndex* ArrLabelIndexPtr;
}*SetArrLabelIndexPtr;              // �洢ͼ���ǵ�����

SetArrLabelIndexPtr getLabelIndex(LabelArr* arr_Label, int len_arr);
void freeSetArrLabelIndexPtr(SetArrLabelIndexPtr dataPtr);

///////////////////////////////////////////////////////////////////////////
double rnd_uni_CNN(long* idum);
extern int     seed_CNN;
extern long    rnd_uni_init_CNN;
int rnd_CNN(int low, int high);
/* Fisher�CYates shuffle algorithm */
void shuffle_CNN(int* x, int size);
double gaussrand_CNN(double a, double b);

#endif
