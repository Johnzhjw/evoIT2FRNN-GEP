#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "MOP_cnn_mat.h"
#include "MOP_cnn_data.h"
#include "MOP_NN_FLT_TYPE.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

// �����
typedef struct convolutional_layer_CNN {
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	// ��������ģ���Ȩ�طֲ���������һ����ά����
	// ���СΪinChannels*outChannels*mapSize*mapSize��С
	// ��������ά���飬��Ҫ��Ϊ�˱���ȫ���ӵ���ʽ��ʵ���Ͼ���㲢û���õ�ȫ���ӵ���ʽ
	// �����������DeapLearningToolboox���CNN���ӣ����õ�����ȫ����
	MY_FLT_TYPE**** mapData;     //�������ģ�������
	MY_FLT_TYPE**** dmapData;    //�������ģ������ݵľֲ��ݶ�
	MY_FLT_TYPE** mapFlag;

	MY_FLT_TYPE* biasData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	bool isFullConnect; //�Ƿ�Ϊȫ����
	bool* connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�

	// �������ߵĴ�Сͬ�����ά����ͬ
	MY_FLT_TYPE*** v; // ���뼤���������ֵ
	MY_FLT_TYPE*** y; // ���������Ԫ�����

	// ������صľֲ��ݶ�
	MY_FLT_TYPE*** d; // ����ľֲ��ݶ�,��ֵ
} CovLayer_CNN;

// ������ pooling
typedef struct pooling_layer_CNN {
	int inputWidth;   //����ͼ��Ŀ�
	int inputHeight;  //����ͼ��ĳ�
	int mapSize;      //����ģ��Ĵ�С

	int inChannels;   //����ͼ�����Ŀ
	int outChannels;  //���ͼ�����Ŀ

	int poolType;     //Pooling�ķ���
	MY_FLT_TYPE* biasData;   //ƫ��

	MY_FLT_TYPE*** y; // ������������Ԫ�����,�޼����
	MY_FLT_TYPE*** d; // ����ľֲ��ݶ�,��ֵ
} PoolLayer_CNN;

// ����� ȫ���ӵ�������
typedef struct nn_layer_CNN {
	int inputNum;   //�������ݵ���Ŀ
	int outputNum;  //������ݵ���Ŀ

	MY_FLT_TYPE** wData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	MY_FLT_TYPE* biasData;   //ƫ�ã���СΪoutputNum��С
	MY_FLT_TYPE** connFalg;

	// �������ߵĴ�Сͬ�����ά����ͬ
	MY_FLT_TYPE* v; // ���뼤���������ֵ
	MY_FLT_TYPE* y; // ���������Ԫ�����
	MY_FLT_TYPE* d; // ����ľֲ��ݶ�,��ֵ

	bool isFullConnect; //�Ƿ�Ϊȫ����
} OutLayer_CNN;

typedef struct cnn_network {
	int layerNum;
	CovLayer_CNN* C1;
	PoolLayer_CNN* S2;
	CovLayer_CNN* C3;
	PoolLayer_CNN* S4;
	OutLayer_CNN* O5;

	MY_FLT_TYPE* e; // ѵ�����
	MY_FLT_TYPE* L; // ˲ʱ�������

	MY_FLT_TYPE* N_sum;
	MY_FLT_TYPE* N_wrong;
	MY_FLT_TYPE* e_sum;

	MY_FLT_TYPE* N_TP;
	MY_FLT_TYPE* N_TN;
	MY_FLT_TYPE* N_FP;
	MY_FLT_TYPE* N_FN;

	int core_mode;
} CNN;

typedef struct nn_network {
	int layerNum;
	OutLayer_CNN* O1;
	OutLayer_CNN* O2;
	OutLayer_CNN* O3;

	MY_FLT_TYPE* e; // ѵ�����
	MY_FLT_TYPE* L; // ˲ʱ�������

	MY_FLT_TYPE* N_sum;
	MY_FLT_TYPE* N_wrong;
	MY_FLT_TYPE* e_sum;

	MY_FLT_TYPE* N_TP;
	MY_FLT_TYPE* N_TN;
	MY_FLT_TYPE* N_FP;
	MY_FLT_TYPE* N_FN;

	int core_mode;
} NN;

typedef struct train_opts {
	int numepochs; // ѵ���ĵ�������
	int batchSize;
	MY_FLT_TYPE alpha; // ѧϰ����
} CNNOpts;

typedef struct nn_train_opts {
	int numepochs; // ѵ���ĵ�������
	int batchSize;
	MY_FLT_TYPE alpha; // ѧϰ����
} NNOpts;

enum CNN_INIT_MODE {
	INIT_MODE,
	ASSIGN_MODE,
	OUTPUT_MODE
};

void cnnsetup(CNN* cnn, nSize inputSize, int outputSize);
void cnnfree(CNN* cnn);
void cnninit(CNN* cnn, double* x, int mode);
void nnsetup(NN* nn, nSize inputSize, int outputSize);
void nnfree(NN* nn);
void nninit(NN* nn, double* x, int mode);
/*
	CNN�����ѵ������
	inputData��outputData�ֱ����ѵ������
	dataNum����������Ŀ
	*/
void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum);
void cnntrain_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum,
	ArrLabelIndex arr_labelIndex);
// ����cnn����
MY_FLT_TYPE cnntest(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum);
MY_FLT_TYPE cnntest_selected(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum, int* vec_index);
//
void cnn_train_bp(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int* indx_samples, int num_samples);
void nn_train_bp(NN* nn, ImgArr inputData, LabelArr outputData, NNOpts opts, int* indx_samples, int num_samples);
MY_FLT_TYPE cnn_err_train(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index);
MY_FLT_TYPE nn_err_train(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index);
MY_FLT_TYPE cnn_err_train_with_noise(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
	MY_FLT_TYPE*** noise_level);
MY_FLT_TYPE nn_err_train_with_noise(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int** pixel_index,
	MY_FLT_TYPE*** noise_level);
MY_FLT_TYPE cnn_err_train_less(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int num_selected);
MY_FLT_TYPE cnn_err_test(CNN* cnn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index);
MY_FLT_TYPE nn_err_test(NN* nn, ImgArr inputData, LabelArr outputData, int* flag_samples, int target_flag, int** pixel_index);
// ����cnn
void savecnn(CNN* cnn, const char* filename);
void savenn(NN* nn, const char* filename);
// ����cnn������
void importcnn(CNN* cnn, const char* filename);

// ��ʼ�������
CovLayer_CNN* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void CovLayerConnect(CovLayer_CNN* covL, bool* connectModel);
void freeCovLayer(CovLayer_CNN* layer_C);
// ��ʼ��������
PoolLayer_CNN* initPoolLayer(int inputWidth, int inputHeigh, int mapSize, int inChannels, int outChannels, int poolType);
void PoolLayerConnect(PoolLayer_CNN* poolL, bool* connectModel);
void freePoolLayer(PoolLayer_CNN* layer_S);
// ��ʼ�������
OutLayer_CNN* initOutLayer(int inputNum, int outputNum);
void freeOutLayer(OutLayer_CNN* layer_O);

enum CORE_MODE {
	CORE_SIGMA,
	CORE_RELU,
	CORE_TANH,
	CORE_LEAKYRELU,
	CORE_ELU
};

// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
MY_FLT_TYPE activation_Sigma(MY_FLT_TYPE input, MY_FLT_TYPE bas); // sigma�����
MY_FLT_TYPE activation_ReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas);
MY_FLT_TYPE activation_tanh(MY_FLT_TYPE input, MY_FLT_TYPE bas);
MY_FLT_TYPE activation_LeakyReLu(MY_FLT_TYPE input, MY_FLT_TYPE bas); //
MY_FLT_TYPE activation_ELU(MY_FLT_TYPE input, MY_FLT_TYPE bas); //

void cnnff(CNN* cnn, MY_FLT_TYPE** inputData, int core_mode); // �����ǰ�򴫲�
void nnff(NN* nn, MY_FLT_TYPE* inputData, int core_mode); // �����ǰ�򴫲�
void cnnbp(CNN* cnn, MY_FLT_TYPE* outputData, int core_mode); // ����ĺ��򴫲�
void nnbp(NN* nn, MY_FLT_TYPE* outputData, int core_mode); // ����ĺ��򴫲�
void cnnapplygrads(CNN* cnn, CNNOpts opts, MY_FLT_TYPE** inputData);
void nnapplygrads(NN* nn, NNOpts opts, MY_FLT_TYPE* inputData);
void cnnclear(CNN* cnn); // ������vyd����
void nnclear(NN* nn); // ������vyd����

int vecmaxIndex(MY_FLT_TYPE* vec, int veclength);// ������������������

/*
	Pooling Function
	input ��������
	inputNum ����������Ŀ
	mapSize ��ƽ����ģ������
	*/
void avgPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize); // ��ƽ��ֵ
void maxPooling(MY_FLT_TYPE** output, nSize outputSize, MY_FLT_TYPE** input, nSize inputSize, int mapSize); // �����ֵ

/*
	����ȫ����������Ĵ���
	nnSize������Ĵ�С
	*/
void nnff_oneLayer(MY_FLT_TYPE* output, MY_FLT_TYPE* input, MY_FLT_TYPE** wdata, MY_FLT_TYPE** connFlag, MY_FLT_TYPE* bas,
	nSize nnSize); // ����ȫ�����������ǰ�򴫲�

void savecnndata(CNN* cnn, const char* filename, MY_FLT_TYPE** inputdata); // ����CNN�����е��������

#endif
