#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOP_EVOmore3_FRNN.h"
//#include <Eigen/Dense>
//#include <Eigen/SVD>
//#include <iostream>

//using namespace Eigen;
//using namespace std;

#define PARA_MAX_EVOmore3_FRNN_MEM_REAL 120.0
#define PARA_MAX_EVOmore3_FRNN_MEM_SIGMA 10.0
#define PARA_MAX_EVOmore3_FRNN_MEM_BELL_A 10.0
#define PARA_MAX_EVOmore3_FRNN_MEM_BELL_B 10.0
#define PARA_MAX_EVOmore3_FRNN_MEM_SIGMOID 10.0
#define PARA_MAX_EVOmore3_FRNN_MEM_RATIO 2.0
#define PARA_MAX_EVOmore3_FRNN_W 1.0
#define PARA_MAX_EVOmore3_FRNN_BIAS 120.0

#define penaltyVal_EVOmore3_FRNN (1e6)

static int seed_EVOmore3_FRNN;
static long rnd_uni_init_EVOmore3_FRNN;
static double trainData_EVOmore3_FRNN[2000];
static double testData_EVOmore3_FRNN[2000];
static int trainDataSize_EVOmore3_FRNN;
static int testDataSize_EVOmore3_FRNN;

// FRNN
static int flag_cluster_bin_rule_EVOmore3_FRNN[MAX_NUM_FUZZY_RULE_EVOmore3_FRNN][N_INPUT_ALL_EVOmore3_FRNN];
static int type_fuzzy_mem_func_EVOmore3_FRNN[N_INPUT_ALL_EVOmore3_FRNN][MAX_NUM_CLUSTER_EVOmore3_FRNN];
static int flag_fuzzy_bin_rough_EVOmore3_FRNN[NUM_CLASS_EVOmore3_FRNN][MAX_NUM_FUZZY_RULE_EVOmore3_FRNN];
static double
para_fuzzy_mem_func_EVOmore3_FRNN[N_INPUT_ALL_EVOmore3_FRNN][MAX_NUM_CLUSTER_EVOmore3_FRNN][MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN];
static double para_fuzzy_rough_EVOmore3_FRNN[NUM_CLASS_EVOmore3_FRNN][MAX_NUM_FUZZY_RULE_EVOmore3_FRNN];
static double para_class_output_EVOmore3_FRNN[NUM_CLASS_EVOmore3_FRNN];

static double L1_input_output_EVOmore3_FRNN[N_INPUT_ALL_EVOmore3_FRNN];
static double L2_output_EVOmore3_FRNN[N_INPUT_ALL_EVOmore3_FRNN][MAX_NUM_CLUSTER_EVOmore3_FRNN];
static double L3_fuzzy_rule_values_EVOmore3_FRNN[MAX_NUM_FUZZY_RULE_EVOmore3_FRNN];
static double L4_output_EVOmore3_FRNN[NUM_CLASS_EVOmore3_FRNN];
static double L5_output_y_EVOmore3_FRNN;

static double total_penalty_EVOmore3_FRNN;

//
static void trimLine_EVOmore3_FRNN(char line[]);
static void preprocess_EVOmore3_FRNN(double *individual);
static double precision_EVOmore3_FRNN(double dataSet[], int dataLen);
static double simplicity_EVOmore3_FRNN();

void Initialize_data_EVOmore3_FRNN(int curN, int numN, int trainNo, int testNo, int endNo)
{
    seed_EVOmore3_FRNN = 237;
    rnd_uni_init_EVOmore3_FRNN = -(long)seed_EVOmore3_FRNN;
    for(int i = 0; i < curN; i++) {
        seed_EVOmore3_FRNN = (seed_EVOmore3_FRNN + 111) % 1235;
        rnd_uni_init_EVOmore3_FRNN = -(long)seed_EVOmore3_FRNN;
    }

    char filename[1024] = "../Data_all/AllFileNames_FRNN";
    FILE* fpt;

    if((fpt = fopen(filename, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }

    trainDataSize_EVOmore3_FRNN = 0;
    testDataSize_EVOmore3_FRNN = 0;

    char StrLine[1024];
    int seq = 0;
    for(seq = 1; seq < trainNo; seq++) {
        // fgets(StrLine, 1024, fpt);
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
    }
    for(seq = trainNo; seq < testNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_EVOmore3_FRNN(StrLine);

        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        // fgets(StrLine, 1024, fpt_data);
        if(fgets(StrLine, 1024, fpt_data) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
        int tmp;
        int tmp_size;
        double elem;

        tmp = fscanf(fpt_data, "%d", &tmp_size);
        if(tmp == EOF) {
            printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
            exit(2000);
        }

        for(int i = 0; i < tmp_size; i++) {
            tmp = fscanf(fpt_data, "%lf", &elem);
            if(tmp == EOF) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2000);
            }
            trainData_EVOmore3_FRNN[trainDataSize_EVOmore3_FRNN + i] = elem;
        }

        trainDataSize_EVOmore3_FRNN += tmp_size;

        fclose(fpt_data);
    }
    for(seq = testNo; seq < endNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine_EVOmore3_FRNN(StrLine);

        FILE* fpt_data;// = fopen(StrLine, "r");
        if((fpt_data = fopen(StrLine, "r")) == NULL) {
            printf("%s(%d): File open error!\n", __FILE__, __LINE__);
            exit(10000);
        }
        // fgets(StrLine, 1024, fpt_data);
        if(fgets(StrLine, 1024, fpt_data) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
        int tmp;
        int tmp_size;
        double elem;

        tmp = fscanf(fpt_data, "%d", &tmp_size);
        if(tmp == EOF) {
            printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
            exit(2000);
        }

        for(int i = 0; i < tmp_size; i++) {
            tmp = fscanf(fpt_data, "%lf", &elem);
            if(tmp == EOF) {
                printf("\n%s(%d):data are not enough...\n", __FILE__, __LINE__);
                exit(2000);
            }
            testData_EVOmore3_FRNN[testDataSize_EVOmore3_FRNN + i] = elem;
        }

        testDataSize_EVOmore3_FRNN += tmp_size;

        fclose(fpt_data);
    }

    fclose(fpt);

    // EVOmore3_FRNN结构的初始化
    // None

    return;
}

void SetLimits_EVOmore3_FRNN(double* minLimit, double* maxLimit, int nx)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVOmore3_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVOmore3_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVOmore3_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVOmore3_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore3_FRNN;
    int offset6 = offset5 + NUM_CLASS_EVOmore3_FRNN;

    for(int i = offset0; i < offset1; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 6 - 1e-6;
    }
    for(int i = offset1; i < offset2; i++) {
        minLimit[i] = 0;
        maxLimit[i] = MAX_NUM_CLUSTER_EVOmore3_FRNN + 1 - 1e-6;
    }
    for(int i = offset2; i < offset3; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 2 - 1e-6;
    }
    for(int i = offset3; i < offset4; i += MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN) {
        minLimit[i + 0] = 0.0;
        minLimit[i + 1] = 1e-2;
        minLimit[i + 2] = 1e-2;
        minLimit[i + 3] = 1e-2;
        minLimit[i + 4] = -PARA_MAX_EVOmore3_FRNN_MEM_SIGMOID;
        minLimit[i + 5] = 0.0;
        minLimit[i + 6] = 0.0;
        minLimit[i + 7] = 0.0;
        minLimit[i + 8] = 0.0;
        maxLimit[i + 0] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        maxLimit[i + 1] = PARA_MAX_EVOmore3_FRNN_MEM_SIGMA;
        maxLimit[i + 2] = PARA_MAX_EVOmore3_FRNN_MEM_BELL_A;
        maxLimit[i + 3] = PARA_MAX_EVOmore3_FRNN_MEM_BELL_B;
        maxLimit[i + 4] = PARA_MAX_EVOmore3_FRNN_MEM_SIGMOID;
        maxLimit[i + 5] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        maxLimit[i + 6] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        maxLimit[i + 7] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        maxLimit[i + 8] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
    }
    for(int i = offset4; i < offset5; i++) {
        minLimit[i] = 0;
        maxLimit[i] = PARA_MAX_EVOmore3_FRNN_W;
    }
    for(int i = offset5; i < offset6; i++) {
        minLimit[i] = 0.0;
        maxLimit[i] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
    }

    return;
}

int CheckLimits_EVOmore3_FRNN(double* x, int nx)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVOmore3_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVOmore3_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVOmore3_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVOmore3_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore3_FRNN;
    int offset6 = offset5 + NUM_CLASS_EVOmore3_FRNN;

    for(int i = offset0; i < offset1; i++) {
        if(x[i] < 0 || x[i] > 6 - 1e-6) {
            printf("%s(%d): Check limits FAIL - EVOmore3_FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 6 - 1e-6);
            return false;
        }
    }
    for(int i = offset1; i < offset2; i++) {
        if(x[i] < 0 || x[i] > MAX_NUM_CLUSTER_EVOmore3_FRNN + 1 - 1e-6) {
            printf("%s(%d): Check limits FAIL - EVOmore3_FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, MAX_NUM_CLUSTER_EVOmore3_FRNN + 1 - 1e-6);
            return false;
        }
    }
    for(int i = offset2; i < offset3; i++) {
        if(x[i] < 0 || x[i] > 2 - 1e-6) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 2 - 1e-6);
            return false;
        }
    }
    for(int i = offset3; i < offset4; i += MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN) {
        double tmp_min[MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN], tmp_max[MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN];
        tmp_min[0] = 0.0;
        tmp_min[1] = 1e-2;
        tmp_min[2] = 1e-2;
        tmp_min[3] = 1e-2;
        tmp_min[4] = -PARA_MAX_EVOmore3_FRNN_MEM_SIGMOID;
        tmp_min[5] = 0.0;
        tmp_min[6] = 0.0;
        tmp_min[7] = 0.0;
        tmp_min[8] = 0.0;
        tmp_max[0] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        tmp_max[1] = PARA_MAX_EVOmore3_FRNN_MEM_SIGMA;
        tmp_max[2] = PARA_MAX_EVOmore3_FRNN_MEM_BELL_A;
        tmp_max[3] = PARA_MAX_EVOmore3_FRNN_MEM_BELL_B;
        tmp_max[4] = PARA_MAX_EVOmore3_FRNN_MEM_SIGMOID;
        tmp_max[5] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        tmp_max[6] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        tmp_max[7] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        tmp_max[8] = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        for(int j = 0; j < MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN; j++) {
            if(x[i + j] < tmp_min[j] || x[i + j] > tmp_max[j]) {
                printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                       __FILE__, __LINE__, i, x[i + j], tmp_min[j], tmp_max[j]);
                return false;
            }
        }
    }
    for(int i = offset4; i < offset5; i++) {
        if(x[i] < 0 || x[i] > PARA_MAX_EVOmore3_FRNN_W) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, PARA_MAX_EVOmore3_FRNN_W);
            return false;
        }
    }
    for(int i = offset5; i < offset6; i++) {
        double tmp_min, tmp_max;
        tmp_min = 0.0;
        tmp_max = PARA_MAX_EVOmore3_FRNN_MEM_REAL;
        if(x[i] < tmp_min || x[i] > tmp_max) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], tmp_min, tmp_max);
            return false;
        }
    }

    return true;
}

static void preprocess_EVOmore3_FRNN(double *individual)
{
    int offset0 = 0;
    int offset1 = offset0 + DIM_CLUSTER_MEM_TYPE_EVOmore3_FRNN;
    int offset2 = offset1 + DIM_CLUSTER_BIN_RULE_EVOmore3_FRNN;
    int offset3 = offset2 + DIM_FUZZY_BIN_ROUGH_EVOmore3_FRNN;
    int offset4 = offset3 + DIM_CLUSTER_MEM_PARA_EVOmore3_FRNN;
    int offset5 = offset4 + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore3_FRNN;
    int offset6 = offset5 + NUM_CLASS_EVOmore3_FRNN;

    //int i;
    int tmp_i, tmp_j, tmp_k;
    for(int i = offset0; i < offset1; i++) {
        tmp_i = (i - offset0) / MAX_NUM_CLUSTER_EVOmore3_FRNN;
        tmp_j = (i - offset0 - tmp_i * MAX_NUM_CLUSTER_EVOmore3_FRNN);
        type_fuzzy_mem_func_EVOmore3_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset1; i < offset2; i++) {
        tmp_i = (i - offset1) / N_INPUT_ALL_EVOmore3_FRNN;
        tmp_j = (i - offset1 - tmp_i * N_INPUT_ALL_EVOmore3_FRNN);
        flag_cluster_bin_rule_EVOmore3_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset2; i < offset3; i++) {
        tmp_i = (i - offset2) / MAX_NUM_FUZZY_RULE_EVOmore3_FRNN;
        tmp_j = (i - offset2 - tmp_i * MAX_NUM_FUZZY_RULE_EVOmore3_FRNN);
        flag_fuzzy_bin_rough_EVOmore3_FRNN[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset3; i < offset4; i++) {
        tmp_i = (i - offset3) / (MAX_NUM_CLUSTER_EVOmore3_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN);
        tmp_j = (i - offset3 - tmp_i * MAX_NUM_CLUSTER_EVOmore3_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN) /
                MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN;
        tmp_k = (i - offset3 - tmp_i * MAX_NUM_CLUSTER_EVOmore3_FRNN * MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN - tmp_j *
                 MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN);
        para_fuzzy_mem_func_EVOmore3_FRNN[tmp_i][tmp_j][tmp_k] = individual[i];
    }
    for(int i = offset4; i < offset5; i++) {
        tmp_i = (i - offset4) / MAX_NUM_FUZZY_RULE_EVOmore3_FRNN;
        tmp_j = (i - offset4 - tmp_i * MAX_NUM_FUZZY_RULE_EVOmore3_FRNN);
        para_fuzzy_rough_EVOmore3_FRNN[tmp_i][tmp_j] = individual[i];
    }
    for(int i = offset5; i < offset6; i++) {
        para_class_output_EVOmore3_FRNN[i - offset5] = individual[i];
    }

    return;
}

void Fitness_EVOmore3_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M)
{
    preprocess_EVOmore3_FRNN(individual);

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    //double f_normp = 0.0;

    f_prcsn = precision_EVOmore3_FRNN(trainData_EVOmore3_FRNN, trainDataSize_EVOmore3_FRNN);

    //
    f_simpl = simplicity_EVOmore3_FRNN();

    fitness[0] = f_prcsn + total_penalty_EVOmore3_FRNN;
    fitness[1] = f_simpl + total_penalty_EVOmore3_FRNN;

    return;
}

void Fitness_EVOmore3_FRNN_testSet(double* individual, double* fitness)
{
    preprocess_EVOmore3_FRNN(individual);

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    //double f_normp = 0.0;

    f_prcsn = precision_EVOmore3_FRNN(testData_EVOmore3_FRNN, testDataSize_EVOmore3_FRNN);

    //
    f_simpl = simplicity_EVOmore3_FRNN();

    fitness[0] = f_prcsn + total_penalty_EVOmore3_FRNN;
    fitness[1] = f_simpl + total_penalty_EVOmore3_FRNN;

    return;
}

static void trimLine_EVOmore3_FRNN(char line[])
{
    int i = 0;

    while(line[i] != '\0') {
        if(line[i] == '\r' || line[i] == '\n') {
            line[i] = '\0';
            break;
        }
        i++;
    }
}

static double precision_EVOmore3_FRNN(double dataSet[], int dataLen)
{
    double f_prcsn = 0.0;

    //0 1 2
    int offset_data[N_INPUT_EVOmore3_FRNN];// = { 0, 1, 2 };
    for(int i = 0; i < N_INPUT_EVOmore3_FRNN; i++) {
        offset_data[i] = i;
    }

    for(int i = offset_data[N_INPUT_EVOmore3_FRNN - 1]; i < dataLen - 1; i++) {
        //Layer 1
        for(int j = 0; j < N_INPUT_EVOmore3_FRNN; j++) {
            L1_input_output_EVOmore3_FRNN[j] = dataSet[i - offset_data[j]];
        }
        for(int j = N_INPUT_EVOmore3_FRNN; j < N_INPUT_ALL_EVOmore3_FRNN; j++) {
            L1_input_output_EVOmore3_FRNN[j] = L1_input_output_EVOmore3_FRNN[j - N_INPUT_EVOmore3_FRNN] -
                                               L1_input_output_EVOmore3_FRNN[j - N_INPUT_EVOmore3_FRNN + 1];
        }
        //Layer 2
        for(int j = 0; j < N_INPUT_ALL_EVOmore3_FRNN; j++) {
            double tmp_func_para[MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN];
            double tmp_func_para_sorted[MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN];
            for(int k = 0; k < MAX_NUM_CLUSTER_EVOmore3_FRNN; k++) {
                for(int l = 0; l < MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN; l++) {
                    tmp_func_para[l] = para_fuzzy_mem_func_EVOmore3_FRNN[j][k][l];
                    tmp_func_para_sorted[l] = tmp_func_para[l];
                }
                for(int a = MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN - 4; a < MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN; a++) {
                    for(int b = a + 1; b < MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN; b++) {
                        if(tmp_func_para_sorted[a] > tmp_func_para_sorted[b]) {
                            double tmp_d = tmp_func_para_sorted[a];
                            tmp_func_para_sorted[a] = tmp_func_para_sorted[b];
                            tmp_func_para_sorted[b] = tmp_d;
                        }
                    }
                }
                double c0 = tmp_func_para_sorted[0];
                double sigma = tmp_func_para_sorted[1];
                double alpha = fabs(tmp_func_para_sorted[2]) + 1e-6;
                double beta = tmp_func_para_sorted[3];
                double gamma = tmp_func_para_sorted[4];
                double p_a = tmp_func_para_sorted[5];
                double p_b = tmp_func_para_sorted[6];
                double p_c = tmp_func_para_sorted[7];
                double p_d = tmp_func_para_sorted[8];
                double tmp1;
                double tmp2;
                switch(type_fuzzy_mem_func_EVOmore3_FRNN[j][k]) {
                case 0:
                    L2_output_EVOmore3_FRNN[j][k] = exp(-(L1_input_output_EVOmore3_FRNN[j] - c0) * (L1_input_output_EVOmore3_FRNN[j] - c0) /
                                                        2.0 /
                                                        (sigma * sigma));
                    break;
                case 1:
                    tmp1 = (L1_input_output_EVOmore3_FRNN[j] - c0) / alpha;
                    tmp1 = tmp1 * tmp1;
                    tmp2 = beta;
                    if(tmp2 < 0) tmp2 = -tmp2;
                    if(tmp2 == 0) tmp2 = 1e-6;
                    tmp1 = pow(tmp1, tmp2);
                    L2_output_EVOmore3_FRNN[j][k] = 1.0 / (1.0 + tmp1);
                    break;
                case 2:
                    L2_output_EVOmore3_FRNN[j][k] = 1.0 / (1.0 + exp(-gamma * (L1_input_output_EVOmore3_FRNN[j] - c0)));
                    break;
                case 3:
                    if(L1_input_output_EVOmore3_FRNN[j] <= p_a)
                        L2_output_EVOmore3_FRNN[j][k] = 0.0;
                    else if(L1_input_output_EVOmore3_FRNN[j] <= p_b)
                        L2_output_EVOmore3_FRNN[j][k] = (L1_input_output_EVOmore3_FRNN[j] - p_a) / (p_b - p_a);
                    else if(L1_input_output_EVOmore3_FRNN[j] <= p_c)
                        L2_output_EVOmore3_FRNN[j][k] = 1.0;
                    else if(L1_input_output_EVOmore3_FRNN[j] <= p_d)
                        L2_output_EVOmore3_FRNN[j][k] = (p_d - L1_input_output_EVOmore3_FRNN[j]) / (p_d - p_c);
                    else
                        L2_output_EVOmore3_FRNN[j][k] = 0.0;
                    break;
                case 4:
                    if(L1_input_output_EVOmore3_FRNN[j] <= p_a)
                        L2_output_EVOmore3_FRNN[j][k] = 0.0;
                    else if(L1_input_output_EVOmore3_FRNN[j] <= (p_b + p_c) / 2.0)
                        L2_output_EVOmore3_FRNN[j][k] = (L1_input_output_EVOmore3_FRNN[j] - p_a) / ((p_b + p_c) / 2.0 - p_a);
                    else if(L1_input_output_EVOmore3_FRNN[j] <= p_d)
                        L2_output_EVOmore3_FRNN[j][k] = (p_d - L1_input_output_EVOmore3_FRNN[j]) / (p_d - (p_b + p_c) / 2.0);
                    else
                        L2_output_EVOmore3_FRNN[j][k] = 0.0;
                    break;
                case 5:
                    if(L1_input_output_EVOmore3_FRNN[j] <= p_a)
                        L2_output_EVOmore3_FRNN[j][k] = 1.0;
                    else if(L1_input_output_EVOmore3_FRNN[j] <= (p_a + p_d) / 2.0)
                        L2_output_EVOmore3_FRNN[j][k] = 1.0 - 2.0 * (L1_input_output_EVOmore3_FRNN[j] - p_a) / (p_d - p_a) *
                                                        (L1_input_output_EVOmore3_FRNN[j] - p_a) / (p_d - p_a);
                    else if(L1_input_output_EVOmore3_FRNN[j] <= p_d)
                        L2_output_EVOmore3_FRNN[j][k] = 2.0 * (p_d - L1_input_output_EVOmore3_FRNN[j]) / (p_d - p_a) *
                                                        (p_d - L1_input_output_EVOmore3_FRNN[j]) / (p_d - p_a);
                    else
                        L2_output_EVOmore3_FRNN[j][k] = 0.0;
                    break;
                default:
                    printf("%s(%d): Unknown MF, exiting...\n", __FILE__, __LINE__);
                    exit(-1234567);
                    break;
                }
            }
        }
        //printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
        //  L2_output[0][0], L2_output[0][1], L2_output[0][2],
        //  L2_output[1][0], L2_output[1][1], L2_output[1][2],
        //  L2_output[2][0], L2_output[2][1], L2_output[2][2],
        //  L2_output[3][0], L2_output[3][1], L2_output[3][2],
        //  L2_output[4][0], L2_output[4][1], L2_output[4][2]);
        //Layer 3
        memset(L3_fuzzy_rule_values_EVOmore3_FRNN, 0, MAX_NUM_FUZZY_RULE_EVOmore3_FRNN * sizeof(double));
        //int step1 = MAX_NUM_CLUSTER_EVOmore3_FRNN;
        //int step2 = step1 * MAX_NUM_CLUSTER_EVOmore3_FRNN;
        //int step3 = step2 * MAX_NUM_CLUSTER_EVOmore3_FRNN;
        //int step4 = step3 * MAX_NUM_CLUSTER_EVOmore3_FRNN;
        //int step5 = step4 * MAX_NUM_CLUSTER_EVOmore3_FRNN;
        for(int a = 0; a < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; a++) {
            int flag_tmp = 0;
            double product = 1.0;
            for(int b = 0; b < N_INPUT_ALL_EVOmore3_FRNN; b++) {
                if(flag_cluster_bin_rule_EVOmore3_FRNN[a][b]) {
                    int tmp_ind = flag_cluster_bin_rule_EVOmore3_FRNN[a][b] - 1;
                    product *= L2_output_EVOmore3_FRNN[b][tmp_ind];
                    flag_tmp++;
                }
            }
            if(flag_tmp) {
                L3_fuzzy_rule_values_EVOmore3_FRNN[a] = product;
            } else {
                L3_fuzzy_rule_values_EVOmore3_FRNN[a] = 0.0;
            }
        }
        double tmp_sum_L4 = 0.0;
        //Layer 4
        for(int j = 0; j < NUM_CLASS_EVOmore3_FRNN; j++) {
            L4_output_EVOmore3_FRNN[j] = 0.0;
            for(int k = 0; k < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; k++) {
                if(flag_fuzzy_bin_rough_EVOmore3_FRNN[j][k])
                    L4_output_EVOmore3_FRNN[j] += para_fuzzy_rough_EVOmore3_FRNN[j][k] * L3_fuzzy_rule_values_EVOmore3_FRNN[k];
            }
            tmp_sum_L4 += L4_output_EVOmore3_FRNN[j];
        }
        ////
        //for (int j = 0; j < NUM_CLASS_EVOmore3_FRNN; j++) {
        //  if (tmp_sum_L4 > 0)
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_EVOmore3_FRNN - 1]) = L4_output[j] / tmp_sum_L4;
        //  else
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_EVOmore3_FRNN - 1]) = L4_output[j];
        //}
        //O_targetOutput(i - offset_data[N_INPUT_EVOmore3_FRNN - 1], 0) = trainData[i + 1];
        //Layer 5
        L5_output_y_EVOmore3_FRNN = 0.0;
        for(int j = 0; j < NUM_CLASS_EVOmore3_FRNN; j++) {
            double tmp = para_class_output_EVOmore3_FRNN[j] * L4_output_EVOmore3_FRNN[j];// *L1_input_output[j];
            //double tmp = L4_output[j] * L1_input_output[j];
            if(tmp_sum_L4 > 0)
                tmp /= tmp_sum_L4;
            L5_output_y_EVOmore3_FRNN += tmp;
        }
        ////
        //double tmp_mean = 0.0;
        //double tmp_std = 0.0;
        //for (int j = 0; j < N_INPUT_EVOmore3_FRNN; j++) {
        //  tmp_mean += L1_input_output[j];
        //}
        //tmp_mean /= N_INPUT_EVOmore3_FRNN;
        //for (int j = 0; j < N_INPUT_EVOmore3_FRNN; j++) {
        //  tmp_std += (L1_input_output[j] - tmp_mean)*(L1_input_output[j] - tmp_mean);
        //}
        //tmp_std = sqrt(tmp_std / N_INPUT_EVOmore3_FRNN);
        //L5_output_y = L5_output_y*tmp_std + tmp_mean;
        // Error
        f_prcsn += (L5_output_y_EVOmore3_FRNN - dataSet[i + 1]) * (L5_output_y_EVOmore3_FRNN - dataSet[i + 1]);
    }

    //MatrixXd L4TL4 = L4DataAllSamples * L4DataAllSamples.transpose();
    //MatrixXd L4TY = L4DataAllSamples * O_targetOutput;
    //JacobiSVD<MatrixXd> svd(L4TL4, ComputeThinU | ComputeThinV);
    //MatrixXd L5Weights = svd.solve(L4TY);
    //MatrixXd O_predicted = L5Weights.transpose() * L4DataAllSamples;
    ////cout << "Before:" << endl << L4TL4 << endl;
    ////cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    ////cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
    ////cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
    ////MatrixXd tmpMat = L4TL4;
    ////tmpMat.setZero(L4TL4.rows(), L4TL4.cols());
    ////for (int i = 0; i < tmpMat.rows(); i++) {
    ////    tmpMat(i, i) = svd.singularValues()(i);
    ////}
    ////cout << "After:" << endl << svd.matrixU() * tmpMat * svd.matrixV().transpose() << endl;

    // precision objective
    //f_prcsn = 0.0;
    //for (int i = offset_data[N_INPUT_EVOmore3_FRNN - 1]; i < trainDataSize - 1; i++) {
    //  // Error
    //  f_prcsn += (O_predicted(0, i - offset_data[N_INPUT_EVOmore3_FRNN - 1]) - trainData[i + 1]) *
    //      (O_predicted(0, i - offset_data[N_INPUT_EVOmore3_FRNN - 1]) - trainData[i + 1]);
    //}
    f_prcsn = sqrt(f_prcsn / (dataLen - offset_data[N_INPUT_EVOmore3_FRNN - 1] - 1));

    return f_prcsn;
}

static double simplicity_EVOmore3_FRNN()
{
    //
    double f_simpl = 0.0;
    total_penalty_EVOmore3_FRNN = 0.0;
    //for (int i = 0; i < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; i++) {
    //  if (flag_fuzzy_rule[i])
    //      f_simpl++;
    //}
    //int flag_no_fuzzy_rule = 0;
    //if (f_simpl == 0)
    //  flag_no_fuzzy_rule = 1;
    int tmp1[MAX_NUM_FUZZY_RULE_EVOmore3_FRNN], tmp2[NUM_CLASS_EVOmore3_FRNN],
        tmp3[N_INPUT_ALL_EVOmore3_FRNN][MAX_NUM_CLUSTER_EVOmore3_FRNN];
    for(int i = 0; i < N_INPUT_ALL_EVOmore3_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_CLUSTER_EVOmore3_FRNN; j++) {
            tmp3[i][j] = 0;
        }
    }
    for(int i = 0; i < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < N_INPUT_ALL_EVOmore3_FRNN; j++) {
            if(flag_cluster_bin_rule_EVOmore3_FRNN[i][j]) {
                tmp1[i]++;
                int tmp_ind = flag_cluster_bin_rule_EVOmore3_FRNN[i][j] - 1;
                tmp3[j][tmp_ind]++;
            }
        }
        f_simpl += (double)tmp1[i] / N_INPUT_ALL_EVOmore3_FRNN;
    }
    for(int i = 0; i < NUM_CLASS_EVOmore3_FRNN; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; j++) {
            if(tmp1[j] && flag_fuzzy_bin_rough_EVOmore3_FRNN[i][j]) {
                tmp2[i]++;
            }
        }
        f_simpl += (double)tmp2[i] / MAX_NUM_FUZZY_RULE_EVOmore3_FRNN;
    }
    f_simpl /= (MAX_NUM_FUZZY_RULE_EVOmore3_FRNN + NUM_CLASS_EVOmore3_FRNN);
    //f_simpl /= MAX_NUM_FUZZY_RULE_EVOmore3_FRNN;

    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    int tmp_sum = 0;
    for(int i = 0; i < MAX_NUM_FUZZY_RULE_EVOmore3_FRNN; i++) {
        tmp_sum += tmp1[i];
    }
    if(tmp_sum == 0.0) {
        total_penalty_EVOmore3_FRNN += penaltyVal_EVOmore3_FRNN;
    }
    //tmp_sum = 0;
    //for(int i = 0; i < NUM_CLASS_EVOmore3_FRNN; i++) {
    //    tmp_sum += tmp2[i];
    //}
    //if(tmp_sum == 0.0) {
    //    f_prcsn += 1e6;
    //    f_simpl += 1e6;
    //}
    for(int i = 0; i < NUM_CLASS_EVOmore3_FRNN; i++) {
        if(tmp2[i] == 0.0) {
            total_penalty_EVOmore3_FRNN += penaltyVal_EVOmore3_FRNN;
        }
    }

    return f_simpl;
}
