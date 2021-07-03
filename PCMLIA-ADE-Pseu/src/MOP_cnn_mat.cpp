#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include <assert.h>
#include "MOP_cnn_mat.h"

MY_FLT_TYPE** rotate180(MY_FLT_TYPE** mat, nSize matSize)// ����ת180��
{
	int i, c, r;
	int outSizeW = matSize.c;
	int outSizeH = matSize.r;
	MY_FLT_TYPE** outputData = (MY_FLT_TYPE**)malloc(outSizeH * sizeof(MY_FLT_TYPE*));
	for (i = 0; i < outSizeH; i++)
		outputData[i] = (MY_FLT_TYPE*)malloc(outSizeW * sizeof(MY_FLT_TYPE));

	//printf("\n%s(%d)\n", __FILE__, __LINE__);
	for (r = 0; r < outSizeH; r++) {
		for (c = 0; c < outSizeW; c++) {
			outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];
			//printf("%lf - %lf ", outputData[r][c], mat[outSizeH - r - 1][outSizeW - c - 1]);
		}
	}
	//printf("\n");

	return outputData;
}

// ���ھ������ز��������ѡ��
// ���ﹲ������ѡ��full��same��valid���ֱ��ʾ
// fullָ��ȫ�����������Ĵ�СΪinSize+(mapSize-1)
// sameָͬ������ͬ��С
// validָ��ȫ������Ĵ�С��һ��ΪinSize-(mapSize-1)��С���䲻��Ҫ��������0����

MY_FLT_TYPE** correlation(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type)// �����
{
	// ����Ļ�������ں��򴫲�ʱ���ã������ڽ�Map��ת180���پ��
	// Ϊ�˷�����㣬�����Ƚ�ͼ������һȦ
	// ����ľ��Ҫ�ֳ�������ż��ģ��ͬ����ģ��
	int i, j, c, r;
	int halfmapsizew;
	int halfmapsizeh;
	if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0) { // ģ���СΪż��
		halfmapsizew = (mapSize.c) / 2; // ���ģ��İ���С
		halfmapsizeh = (mapSize.r) / 2;
	}
	else {
		halfmapsizew = (mapSize.c - 1) / 2; // ���ģ��İ���С
		halfmapsizeh = (mapSize.r - 1) / 2;
	}

	// ������Ĭ�Ͻ���fullģʽ�Ĳ�����fullģʽ�������СΪinSize+(mapSize-1)
	int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����
	int outSizeH = inSize.r + (mapSize.r - 1);
	MY_FLT_TYPE** outputData = (MY_FLT_TYPE**)malloc(outSizeH * sizeof(MY_FLT_TYPE*)); // ����صĽ��������
	for (i = 0; i < outSizeH; i++)
		outputData[i] = (MY_FLT_TYPE*)calloc(outSizeW, sizeof(MY_FLT_TYPE));

	// Ϊ�˷�����㣬��inputData����һȦ
	MY_FLT_TYPE** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);

	//printf("\n%s(%d): \n", __FILE__, __LINE__);
	for (j = 0; j < outSizeH; j++) {
		for (i = 0; i < outSizeW; i++) {
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
					//printf("%lf ", outputData[j][i]);
				}
			}
		}
		//printf("\n");
	}

	for (i = 0; i < inSize.r + 2 * (mapSize.r - 1); i++)
		free(exInputData[i]);
	free(exInputData);

	nSize outSize = { outSizeW, outSizeH };
	switch (type) { // ���ݲ�ͬ����������ز�ͬ�Ľ��
	case full: // ��ȫ��С�����
		//printf("\n%s(%d): \n", __FILE__, __LINE__);
		//for(int a = 0; a < outSizeH; a++) {
		//    for(int b = 0; b < outSizeW; b++) {
		//        printf("%lf ", outputData[a][b]);
		//    }
		//    printf("\n");
		//}
		return outputData;
	case same: {
		MY_FLT_TYPE** sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
		for (i = 0; i < outSize.r; i++)
			free(outputData[i]);
		free(outputData);
		return sameres;
	}
	case valid: {
		MY_FLT_TYPE** validres;
		if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)
			validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
		else
			validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);
		for (i = 0; i < outSize.r; i++)
			free(outputData[i]);
		free(outputData);
		//printf("\n%s(%d): \n", __FILE__, __LINE__);
		//for(int a = 0; a < outSize.r - 2 * (halfmapsizeh * 2); a++) {
		//    for(int b = 0; b < outSize.c - 2 * (halfmapsizew * 2); b++) {
		//        printf("%lf ", validres[a][b]);
		//    }
		//    printf("\n");
		//}
		return validres;
	}
	default:
		return outputData;
	}
}

MY_FLT_TYPE** cov(MY_FLT_TYPE** map, nSize mapSize, MY_FLT_TYPE** inputData, nSize inSize, int type) // �������
{
	// ���������������ת180�ȵ�����ģ���������
	MY_FLT_TYPE** flipmap = rotate180(map, mapSize); //��ת180�ȵ�����ģ��
	MY_FLT_TYPE** res = correlation(flipmap, mapSize, inputData, inSize, type);
	int i;
	for (i = 0; i < mapSize.r; i++)
		free(flipmap[i]);
	free(flipmap);
	return res;
}

// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr���ڲ屶��
MY_FLT_TYPE** UpSample(MY_FLT_TYPE** mat, nSize matSize, int upc, int upr)
{
	int i, j, m, n;
	int c = matSize.c;
	int r = matSize.r;
	MY_FLT_TYPE** res = (MY_FLT_TYPE**)malloc((r * upr) * sizeof(MY_FLT_TYPE*)); // ����ĳ�ʼ��
	for (i = 0; i < (r * upr); i++)
		res[i] = (MY_FLT_TYPE*)malloc((c * upc) * sizeof(MY_FLT_TYPE));

	for (j = 0; j < r * upr; j = j + upr) {
		for (i = 0; i < c * upc; i = i + upc) // �������
			for (m = 0; m < upc; m++)
				res[j][i + m] = mat[j / upr][i / upc];

		for (n = 1; n < upr; n++)       //  �ߵ�����
			for (i = 0; i < c * upc; i++)
				res[j + n][i] = res[j][i];
	}
	return res;
}

// ����ά�����Ե��������addw��С��0ֵ��
MY_FLT_TYPE** matEdgeExpand(MY_FLT_TYPE** mat, nSize matSize, int addc, int addr)
{
	// ������Ե����
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	MY_FLT_TYPE** res = (MY_FLT_TYPE**)malloc((r + 2 * addr) * sizeof(MY_FLT_TYPE*)); // ����ĳ�ʼ��
	assert(res);
	for (i = 0; i < (r + 2 * addr); i++) {
		res[i] = (MY_FLT_TYPE*)malloc((c + 2 * addc) * sizeof(MY_FLT_TYPE));
		assert(res[i]);
	}

	for (j = 0; j < r + 2 * addr; j++) {
		for (i = 0; i < c + 2 * addc; i++) {
			if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
				res[j][i] = (MY_FLT_TYPE)0.0;
			else
				res[j][i] = mat[j - addr][i - addc]; // ����ԭ����������
		}
	}
	return res;
}

// ����ά�����Ե��С������shrinkc��С�ı�
MY_FLT_TYPE** matEdgeShrink(MY_FLT_TYPE** mat, nSize matSize, int shrinkc, int shrinkr)
{
	// ��������С������Сaddw������Сaddh
	int i, j;
	int c = matSize.c;
	int r = matSize.r;
	MY_FLT_TYPE** res = (MY_FLT_TYPE**)malloc((r - 2 * shrinkr) * sizeof(MY_FLT_TYPE*)); // �������ĳ�ʼ��
	for (i = 0; i < (r - 2 * shrinkr); i++)
		res[i] = (MY_FLT_TYPE*)malloc((c - 2 * shrinkc) * sizeof(MY_FLT_TYPE));

	for (j = 0; j < r; j++) {
		for (i = 0; i < c; i++) {
			if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc))
				res[j - shrinkr][i - shrinkc] = mat[j][i]; // ����ԭ����������
		}
	}
	return res;
}

void savemat(MY_FLT_TYPE** mat, nSize matSize, const char* filename)
{
	FILE* fp = NULL;
	fp = fopen(filename, "wb");
	if (fp == NULL)
		printf("write file failed\n");

	int i;
	for (i = 0; i < matSize.r; i++)
		fwrite(mat[i], sizeof(MY_FLT_TYPE), matSize.c, fp);
	fclose(fp);
}

void addmat(MY_FLT_TYPE** res, MY_FLT_TYPE** mat1, nSize matSize1, MY_FLT_TYPE** mat2, nSize matSize2)// �������
{
	int i, j;
	if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)
		printf("ERROR: Size is not same!");

	for (i = 0; i < matSize1.r; i++)
		for (j = 0; j < matSize1.c; j++) {
			res[i][j] = mat1[i][j] + mat2[i][j];
			//printf("%lf + %lf = %lf ", mat1[i][j], mat2[i][j], res[i][j]);
		}
	//printf("\n");
}

void multifactor(MY_FLT_TYPE** res, MY_FLT_TYPE** mat, nSize matSize, MY_FLT_TYPE factor)// �������ϵ��
{
	int i, j;
	for (i = 0; i < matSize.r; i++)
		for (j = 0; j < matSize.c; j++)
			res[i][j] = mat[i][j] * factor;
}

MY_FLT_TYPE summat(MY_FLT_TYPE** mat, nSize matSize) // �����Ԫ�صĺ�
{
	MY_FLT_TYPE sum = 0.0;
	int i, j;
	for (i = 0; i < matSize.r; i++)
		for (j = 0; j < matSize.c; j++)
			sum = sum + mat[i][j];
	return sum;
}