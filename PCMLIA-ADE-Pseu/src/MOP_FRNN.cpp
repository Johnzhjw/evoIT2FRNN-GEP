#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOP_FRNN.h"
//#include <Eigen/Dense>
//#include <Eigen/SVD>
//#include <iostream>

//using namespace Eigen;
//using namespace std;

#define PARA_MAX_FRNN_MEM 120.0
#define PARA_MAX_FRNN_W 120.0

static int seed_FRNN;
static long rnd_uni_init_FRNN;
static double trainData[2000];
static double testData[2000];
static int trainDataSize;
static int testDataSize;

// FRNN
static int flag_fuzzy_mem_func[N_INPUT_FRNN][MAX_NUM_CLUSTER_FRNN];
static int flag_fuzzy_rule[MAX_NUM_FUZZY_RULE];
static int type_fuzzy_mem_func[N_INPUT_FRNN][MAX_NUM_CLUSTER_FRNN];
static int flag_fuzzy_rough[NUM_CLASS_FRNN][MAX_NUM_FUZZY_RULE];
static double para_fuzzy_mem_func[N_INPUT_FRNN][MAX_NUM_CLUSTER_FRNN][MAX_NUM_CLUSTER_MEM_PPARA];
static double para_fuzzy_rough[NUM_CLASS_FRNN][MAX_NUM_FUZZY_RULE];
static double para_class_output[NUM_CLASS_FRNN];

static double L1_input_output[N_INPUT_FRNN];
static double L2_output[N_INPUT_FRNN][MAX_NUM_CLUSTER_FRNN];
static double L3_fuzzy_rule_values[MAX_NUM_FUZZY_RULE];
static double L4_output[NUM_CLASS_FRNN];
static double L5_output_y;

//
static void trimLine(char line[]);
static double precision_FRNN(double dataSet[], int dataLen);

void Initialize_data_FRNN(int curN, int numN, int trainNo, int testNo, int endNo)
{
    seed_FRNN = 237;
    rnd_uni_init_FRNN = -(long)seed_FRNN;
    for(int i = 0; i < curN; i++) {
        seed_FRNN = (seed_FRNN + 111) % 1235;
        rnd_uni_init_FRNN = -(long)seed_FRNN;
    }

    char filename[1024] = "../Data_all/AllFileNames_FRNN";
    FILE* fpt;

    if((fpt = fopen(filename, "r")) == NULL) {
        printf("%s(%d): File open error!\n", __FILE__, __LINE__);
        exit(10000);
    }

    trainDataSize = 0;
    testDataSize = 0;

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
        trimLine(StrLine);

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
            trainData[trainDataSize + i] = elem;
        }

        trainDataSize += tmp_size;

        fclose(fpt_data);
    }
    for(seq = testNo; seq < endNo; seq++) {
        // fgets(StrLine, 1024, fpt);// column name
        if(fgets(StrLine, 1024, fpt) == NULL) {
            printf("%s(%d): No more line\n", __FILE__, __LINE__);
            exit(-1);
        }
        trimLine(StrLine);

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
            testData[testDataSize + i] = elem;
        }

        testDataSize += tmp_size;

        fclose(fpt_data);
    }

    fclose(fpt);

    // FRNN结构的初始化
    // None

    return;
}

void SetLimits_FRNN(double* minLimit, double* maxLimit, int nx)
{
    int i;
    int i_offset;
    for(i = 0; i < DIM_CLUSTER_MEM; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 2 - 1e-6;
    }
    i_offset = DIM_CLUSTER_MEM;
    for(i = i_offset; i < i_offset + DIM_CLUSTER_MEM_TYPE; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 6 - 1e-6;
    }
    i_offset += DIM_CLUSTER_MEM_TYPE;
    for(i = i_offset; i < i_offset + DIM_FUZZY_ROUGH; i++) {
        minLimit[i] = 0;
        maxLimit[i] = 2 - 1e-6;
    }
    i_offset += DIM_FUZZY_ROUGH;
    for(i = i_offset; i < i_offset + DIM_CLUSTER_MEM_PARA; i++) {
        minLimit[i] = -PARA_MAX_FRNN_MEM;
        maxLimit[i] = PARA_MAX_FRNN_MEM;
    }
    i_offset += DIM_CLUSTER_MEM_PARA;
    for(i = i_offset; i < i_offset + DIM_FUZZY_ROUGH_MEM_PARA; i++) {
        minLimit[i] = 0;
        maxLimit[i] = PARA_MAX_FRNN_W;
    }
    i_offset += DIM_FUZZY_ROUGH_MEM_PARA;
    for(i = i_offset; i < DIM_FRNN; i++) {
        minLimit[i] = -PARA_MAX_FRNN_W;
        maxLimit[i] = PARA_MAX_FRNN_W;
    }

    return;
}

int CheckLimits_FRNN(double* x, int nx)
{
    int i;
    int i_offset;
    for(i = 0; i < DIM_CLUSTER_MEM; i++) {
        if(x[i] < 0 || x[i] > 2 - 1e-6) {
            printf("%s(%d): Check limits FAIL - FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 2 - 1e-6);
            return false;
        }
    }
    i_offset = DIM_CLUSTER_MEM;
    for(i = i_offset; i < i_offset + DIM_CLUSTER_MEM_TYPE; i++) {
        if(x[i] < 0 || x[i] > 6 - 1e-6) {
            printf("%s(%d): Check limits FAIL - FRNN: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 6 - 1e-6);
            return false;
        }
    }
    i_offset += DIM_CLUSTER_MEM_TYPE;
    for(i = i_offset; i < i_offset + DIM_FUZZY_ROUGH; i++) {
        if(x[i] < 0 || x[i] > 2 - 1e-6) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, 2 - 1e-6);
            return false;
        }
    }
    i_offset += DIM_FUZZY_ROUGH;
    for(i = i_offset; i < i_offset + DIM_CLUSTER_MEM_PARA; i++) {
        if(x[i] < -PARA_MAX_FRNN_MEM || x[i] > PARA_MAX_FRNN_MEM) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], -PARA_MAX_FRNN_MEM, PARA_MAX_FRNN_MEM);
            return false;
        }
    }
    i_offset += DIM_CLUSTER_MEM_PARA;
    for(i = i_offset; i < i_offset + DIM_FUZZY_ROUGH_MEM_PARA; i++) {
        if(x[i] < 0 || x[i] > PARA_MAX_FRNN_W) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], 0.0, PARA_MAX_FRNN_W);
            return false;
        }
    }
    i_offset += DIM_FUZZY_ROUGH_MEM_PARA;
    for(i = i_offset; i < DIM_FRNN; i++) {
        if(x[i] < -PARA_MAX_FRNN_W || x[i] > PARA_MAX_FRNN_W) {
            printf("%s(%d): Check limits FAIL - LeNet: %d, %.16e not in [%.16e, %.16e]\n",
                   __FILE__, __LINE__, i, x[i], -PARA_MAX_FRNN_W, PARA_MAX_FRNN_W);
            return false;
        }
    }

    return true;
}

void Fitness_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M)
{
    int offset0 = 0;
    int offset1 = DIM_CLUSTER_MEM;
    int offset2 = DIM_CLUSTER_MEM;
    int offset3 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE;
    int offset4 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH;
    int offset5 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH + DIM_CLUSTER_MEM_PARA;
    int offset6 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH + DIM_CLUSTER_MEM_PARA +
                  DIM_FUZZY_ROUGH_MEM_PARA;

    //int i;
    int tmp_i, tmp_j, tmp_k;
    for(int i = offset0; i < offset1; i++) {
        tmp_i = i / MAX_NUM_CLUSTER_FRNN;
        tmp_j = i - tmp_i * MAX_NUM_CLUSTER_FRNN;
        flag_fuzzy_mem_func[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = 0; i < MAX_NUM_FUZZY_RULE; i++) {
        flag_fuzzy_rule[i] = 1;
    }
    for(int i = offset2; i < offset3; i++) {
        tmp_i = (i - offset2) / MAX_NUM_CLUSTER_FRNN;
        tmp_j = (i - offset2 - tmp_i * MAX_NUM_CLUSTER_FRNN);
        type_fuzzy_mem_func[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset3; i < offset4; i++) {
        tmp_i = (i - offset3) / MAX_NUM_FUZZY_RULE;
        tmp_j = (i - offset3 - tmp_i * MAX_NUM_FUZZY_RULE);
        flag_fuzzy_rough[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset4; i < offset5; i++) {
        tmp_i = (i - offset4) / (MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA);
        tmp_j = (i - offset4 - tmp_i * MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA) / MAX_NUM_CLUSTER_MEM_PPARA;
        tmp_k = (i - offset4 - tmp_i * MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA - tmp_j * MAX_NUM_CLUSTER_MEM_PPARA);
        para_fuzzy_mem_func[tmp_i][tmp_j][tmp_k] = individual[i];
    }
    for(int i = offset5; i < offset6; i++) {
        tmp_i = (i - offset5) / MAX_NUM_FUZZY_RULE;
        tmp_j = (i - offset5 - tmp_i * MAX_NUM_FUZZY_RULE);
        para_fuzzy_rough[tmp_i][tmp_j] = individual[i];
    }
    for(int i = offset6; i < DIM_FRNN; i++) {
        para_class_output[i - offset6] = individual[i];
    }

    //
    int count_fuzzy_rule = 0;
    for(int a = 0; a < MAX_NUM_CLUSTER_FRNN; a++) {
        int flag1 = flag_fuzzy_mem_func[0][a];
        for(int b = 0; b < MAX_NUM_CLUSTER_FRNN; b++) {
            int flag2 = flag_fuzzy_mem_func[1][b];
            for(int c = 0; c < MAX_NUM_CLUSTER_FRNN; c++) {
                int flag3 = flag_fuzzy_mem_func[2][c];
                for(int d = 0; d < MAX_NUM_CLUSTER_FRNN; d++) {
                    int flag4 = flag_fuzzy_mem_func[3][d];
                    for(int e = 0; e < MAX_NUM_CLUSTER_FRNN; e++) {
                        int flag5 = flag_fuzzy_mem_func[4][e];
                        //if (flag1 && flag2 && flag3 && flag4 && flag5) {
                        //}
                        //else {
                        //  flag_fuzzy_rule[count_fuzzy_rule] = 0;
                        //}
                        //count_fuzzy_rule++;
                        for(int f = 0; f < MAX_NUM_CLUSTER_FRNN; f++) {
                            int flag6 = flag_fuzzy_mem_func[5][f];
                            if(flag1 && flag2 && flag3 && flag4 && flag5 && flag6) {
                            } else {
                                flag_fuzzy_rule[count_fuzzy_rule] = 0;
                            }
                            count_fuzzy_rule++;
                        }
                    }
                }
            }
        }
    }

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    double f_normp = 0.0;

    f_prcsn = precision_FRNN(trainData, trainDataSize);

    //
    f_simpl = 0.0;
    //for (int i = 0; i < MAX_NUM_FUZZY_RULE; i++) {
    //  if (flag_fuzzy_rule[i])
    //      f_simpl++;
    //}
    //int flag_no_fuzzy_rule = 0;
    //if (f_simpl == 0)
    //  flag_no_fuzzy_rule = 1;
    double tmp1[N_INPUT_FRNN], tmp2[NUM_CLASS_FRNN];
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < MAX_NUM_CLUSTER_FRNN; j++) {
            if(flag_fuzzy_mem_func[i][j])
                tmp1[i]++;
        }
        tmp1[i] /= MAX_NUM_CLUSTER_FRNN;
        f_simpl += tmp1[i];
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < MAX_NUM_FUZZY_RULE; j++) {
            if(flag_fuzzy_rule[j] && flag_fuzzy_rough[i][j]) {
                tmp2[i]++;
            }
        }
        tmp2[i] /= MAX_NUM_FUZZY_RULE;
        f_simpl += tmp2[i];
    }
    f_simpl /= (N_INPUT_FRNN + NUM_CLASS_FRNN);

    //
    int tmp_count = 0;
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_CLUSTER_FRNN; j++) {
            if(flag_fuzzy_mem_func[i][j]) {
                tmp_count += MAX_NUM_CLUSTER_MEM_PPARA;
                for(int k = 0; k < MAX_NUM_CLUSTER_MEM_PPARA; k++) {
                    f_normp += fabs(para_fuzzy_mem_func[i][j][k]) / PARA_MAX_FRNN_MEM;
                }
            }
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_FUZZY_RULE; j++) {
            if(flag_fuzzy_rule[j] && flag_fuzzy_rough[i][j]) {
                tmp_count++;
                f_normp += fabs(para_fuzzy_rough[i][j]) / PARA_MAX_FRNN_W;
            }
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        tmp_count++;
        f_normp += fabs(para_class_output[i]) / PARA_MAX_FRNN_W;
    }
    if(tmp_count) {
        f_normp /= tmp_count;
    } else {
        f_normp = 0.0;
    }

    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        if(tmp1[i] == 0) {
            f_prcsn += 1e6;
            f_simpl += 1e6;
            f_normp += 1e6;
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        if(tmp2[i] == 0) {
            f_prcsn += 1e6;
            f_simpl += 1e6;
            f_normp += 1e6;
        }
    }
    fitness[0] = f_prcsn;
    fitness[1] = f_simpl;
    //fitness[2] = f_normp;

    return;
}

void Fitness_FRNN_testSet(double* individual, double* fitness)
{
    int offset0 = 0;
    int offset1 = DIM_CLUSTER_MEM;
    int offset2 = DIM_CLUSTER_MEM;
    int offset3 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE;
    int offset4 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH;
    int offset5 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH + DIM_CLUSTER_MEM_PARA;
    int offset6 = DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH + DIM_CLUSTER_MEM_PARA +
                  DIM_FUZZY_ROUGH_MEM_PARA;

    //int i;
    int tmp_i, tmp_j, tmp_k;
    for(int i = offset0; i < offset1; i++) {
        tmp_i = i / MAX_NUM_CLUSTER_FRNN;
        tmp_j = i - tmp_i * MAX_NUM_CLUSTER_FRNN;
        flag_fuzzy_mem_func[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = 0; i < MAX_NUM_FUZZY_RULE; i++) {
        flag_fuzzy_rule[i] = 1;
    }
    for(int i = offset2; i < offset3; i++) {
        tmp_i = (i - offset2) / MAX_NUM_CLUSTER_FRNN;
        tmp_j = (i - offset2 - tmp_i * MAX_NUM_CLUSTER_FRNN);
        type_fuzzy_mem_func[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset3; i < offset4; i++) {
        tmp_i = (i - offset3) / MAX_NUM_FUZZY_RULE;
        tmp_j = (i - offset3 - tmp_i * MAX_NUM_FUZZY_RULE);
        flag_fuzzy_rough[tmp_i][tmp_j] = (int)individual[i];
    }
    for(int i = offset4; i < offset5; i++) {
        tmp_i = (i - offset4) / (MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA);
        tmp_j = (i - offset4 - tmp_i * MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA) / MAX_NUM_CLUSTER_MEM_PPARA;
        tmp_k = (i - offset4 - tmp_i * MAX_NUM_CLUSTER_FRNN * MAX_NUM_CLUSTER_MEM_PPARA - tmp_j * MAX_NUM_CLUSTER_MEM_PPARA);
        para_fuzzy_mem_func[tmp_i][tmp_j][tmp_k] = individual[i];
    }
    for(int i = offset5; i < offset6; i++) {
        tmp_i = (i - offset5) / MAX_NUM_FUZZY_RULE;
        tmp_j = (i - offset5 - tmp_i * MAX_NUM_FUZZY_RULE);
        para_fuzzy_rough[tmp_i][tmp_j] = individual[i];
    }
    for(int i = offset6; i < DIM_FRNN; i++) {
        para_class_output[i - offset6] = individual[i];
    }

    //
    int count_fuzzy_rule = 0;
    for(int a = 0; a < MAX_NUM_CLUSTER_FRNN; a++) {
        int flag1 = flag_fuzzy_mem_func[0][a];
        for(int b = 0; b < MAX_NUM_CLUSTER_FRNN; b++) {
            int flag2 = flag_fuzzy_mem_func[1][b];
            for(int c = 0; c < MAX_NUM_CLUSTER_FRNN; c++) {
                int flag3 = flag_fuzzy_mem_func[2][c];
                for(int d = 0; d < MAX_NUM_CLUSTER_FRNN; d++) {
                    int flag4 = flag_fuzzy_mem_func[3][d];
                    for(int e = 0; e < MAX_NUM_CLUSTER_FRNN; e++) {
                        int flag5 = flag_fuzzy_mem_func[4][e];
                        //if (flag1 && flag2 && flag3 && flag4 && flag5) {
                        //}
                        //else {
                        //  flag_fuzzy_rule[count_fuzzy_rule] = 0;
                        //}
                        //count_fuzzy_rule++;
                        for(int f = 0; f < MAX_NUM_CLUSTER_FRNN; f++) {
                            int flag6 = flag_fuzzy_mem_func[5][f];
                            if(flag1 && flag2 && flag3 && flag4 && flag5 && flag6) {
                            } else {
                                flag_fuzzy_rule[count_fuzzy_rule] = 0;
                            }
                            count_fuzzy_rule++;
                        }
                    }
                }
            }
        }
    }

    double f_prcsn = 0.0;
    double f_simpl = 0.0;
    double f_normp = 0.0;

    f_prcsn = precision_FRNN(testData, testDataSize);

    //
    f_simpl = 0.0;
    //for (int i = 0; i < MAX_NUM_FUZZY_RULE; i++) {
    //  if (flag_fuzzy_rule[i])
    //      f_simpl++;
    //}
    //int flag_no_fuzzy_rule = 0;
    //if (f_simpl == 0)
    //  flag_no_fuzzy_rule = 1;
    double tmp1[N_INPUT_FRNN], tmp2[NUM_CLASS_FRNN];
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        tmp1[i] = 0;
        for(int j = 0; j < MAX_NUM_CLUSTER_FRNN; j++) {
            if(flag_fuzzy_mem_func[i][j])
                tmp1[i]++;
        }
        tmp1[i] /= MAX_NUM_CLUSTER_FRNN;
        f_simpl += tmp1[i];
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        tmp2[i] = 0;
        for(int j = 0; j < MAX_NUM_FUZZY_RULE; j++) {
            if(flag_fuzzy_rule[j] && flag_fuzzy_rough[i][j]) {
                tmp2[i]++;
            }
        }
        tmp2[i] /= MAX_NUM_FUZZY_RULE;
        f_simpl += tmp2[i];
    }
    f_simpl /= (N_INPUT_FRNN + NUM_CLASS_FRNN);

    //
    int tmp_count = 0;
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_CLUSTER_FRNN; j++) {
            if(flag_fuzzy_mem_func[i][j]) {
                tmp_count += MAX_NUM_CLUSTER_MEM_PPARA;
                for(int k = 0; k < MAX_NUM_CLUSTER_MEM_PPARA; k++) {
                    f_normp += fabs(para_fuzzy_mem_func[i][j][k]) / PARA_MAX_FRNN_MEM;
                }
            }
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        for(int j = 0; j < MAX_NUM_FUZZY_RULE; j++) {
            if(flag_fuzzy_rule[j] && flag_fuzzy_rough[i][j]) {
                tmp_count++;
                f_normp += fabs(para_fuzzy_rough[i][j]) / PARA_MAX_FRNN_W;
            }
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        tmp_count++;
        f_normp += fabs(para_class_output[i]) / PARA_MAX_FRNN_W;
    }
    if(tmp_count) {
        f_normp /= tmp_count;
    } else {
        f_normp = 0.0;
    }

    //
    //if (flag_no_fuzzy_rule) {
    //  f_prcsn += 1e6;
    //  f_simpl += 1e6;
    //  f_normp += 1e6;
    //}
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        if(tmp1[i] == 0) {
            f_prcsn += 1e6;
            f_simpl += 1e6;
            f_normp += 1e6;
        }
    }
    for(int i = 0; i < NUM_CLASS_FRNN; i++) {
        if(tmp2[i] == 0) {
            f_prcsn += 1e6;
            f_simpl += 1e6;
            f_normp += 1e6;
        }
    }
    fitness[0] = f_prcsn;
    fitness[1] = f_simpl;
    //fitness[2] = f_normp;

    return;
}

static void trimLine(char line[])
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

static double precision_FRNN(double dataSet[], int dataLen)
{
    double f_prcsn = 0.0;

    //0 1 2 3 4 5
    int offset_data[N_INPUT_FRNN];
    for(int i = 0; i < N_INPUT_FRNN; i++) {
        offset_data[i] = i;
    }

    //MatrixXd L4DataAllSamples(NUM_CLASS_FRNN, trainDataSize - offset_data[N_INPUT_FRNN - 1] - 1);
    //MatrixXd O_targetOutput(trainDataSize - offset_data[N_INPUT_FRNN - 1] - 1, 1);

    //for (int i = offset_data[N_INPUT_FRNN - 1]; i < trainDataSize - 1; i++) {
    //  //Layer 1
    //  for (int j = 0; j < N_INPUT_FRNN; j++) {
    //      L1_input_output[j] = trainData[i - offset_data[j]];
    //  }
    //  //Layer 2
    //  for (int j = 0; j < N_INPUT_FRNN; j++) {
    //      double tmp_func_para[MAX_NUM_CLUSTER_MEM_PPARA];
    //      double tmp_func_para_sorted[MAX_NUM_CLUSTER_MEM_PPARA];
    //      for (int k = 0; k < MAX_NUM_CLUSTER_FRNN; k++) {
    //          if (flag_fuzzy_mem_func[j][k]) {
    //              for (int l = 0; l < MAX_NUM_CLUSTER_MEM_PPARA; l++) {
    //                  tmp_func_para[l] = para_fuzzy_mem_func[j][k][l];
    //                  tmp_func_para_sorted[l] = tmp_func_para[l];
    //              }
    //              for (int a = 1; a < MAX_NUM_CLUSTER_MEM_PPARA; a++) {
    //                  for (int b = a + 1; b < MAX_NUM_CLUSTER_MEM_PPARA; b++) {
    //                      if (tmp_func_para_sorted[a] > tmp_func_para_sorted[b]) {
    //                          double tmp_d = tmp_func_para_sorted[a];
    //                          tmp_func_para_sorted[a] = tmp_func_para_sorted[b];
    //                          tmp_func_para_sorted[b] = tmp_d;
    //                      }
    //                  }
    //              }
    //              double tmp1;
    //              int tmp2;
    //              switch (type_fuzzy_mem_func[j][k])
    //              {
    //              case 0:
    //                  L2_output[j][k] = exp(-(L1_input_output[j] - tmp_func_para[3])*(L1_input_output[j] - tmp_func_para[3]) / 2.0 / (tmp_func_para[0] * tmp_func_para[0]));
    //                  break;
    //              case 1:
    //                  tmp1 = (L1_input_output[j] - tmp_func_para[3]) / tmp_func_para[1];
    //                  tmp1 = tmp1*tmp1;
    //                  tmp2 = tmp_func_para[2];
    //                  if (tmp2 < 0) tmp2 = -tmp2;
    //                  if (tmp2 == 0) tmp2 = 1e-6;
    //                  tmp1 = pow(tmp1, tmp2);
    //                  L2_output[j][k] = 1.0 / (1.0 + tmp1);
    //                  break;
    //              case 2:
    //                  L2_output[j][k] = 1.0 / (1.0 + exp(-tmp_func_para[1] * (L1_input_output[j] - tmp_func_para[3])));
    //                  break;
    //              case 3:
    //                  if (L1_input_output[j] <= tmp_func_para_sorted[1])
    //                      L2_output[j][k] = 0.0;
    //                  else if (L1_input_output[j] <= tmp_func_para_sorted[2])
    //                      L2_output[j][k] = (L1_input_output[j] - tmp_func_para_sorted[1]) / (tmp_func_para_sorted[2] - tmp_func_para_sorted[1]);
    //                  else if (L1_input_output[j] <= tmp_func_para_sorted[3])
    //                      L2_output[j][k] = 1.0;
    //                  else if (L1_input_output[j] <= tmp_func_para_sorted[4])
    //                      L2_output[j][k] = (tmp_func_para_sorted[4] - L1_input_output[j]) / (tmp_func_para_sorted[4] - tmp_func_para_sorted[3]);
    //                  else
    //                      L2_output[j][k] = 0.0;
    //                  break;
    //              case 4:
    //                  if (L1_input_output[j] <= tmp_func_para_sorted[1])
    //                      L2_output[j][k] = 0.0;
    //                  else if (L1_input_output[j] <= (tmp_func_para_sorted[2] + tmp_func_para_sorted[3]) / 2.0)
    //                      L2_output[j][k] = (L1_input_output[j] - tmp_func_para_sorted[1]) / ((tmp_func_para_sorted[2] + tmp_func_para_sorted[3]) / 2.0 - tmp_func_para_sorted[1]);
    //                  else if (L1_input_output[j] <= tmp_func_para_sorted[4])
    //                      L2_output[j][k] = (tmp_func_para_sorted[4] - L1_input_output[j]) / (tmp_func_para_sorted[4] - (tmp_func_para_sorted[2] + tmp_func_para_sorted[3]) / 2.0);
    //                  else
    //                      L2_output[j][k] = 0.0;
    //                  break;
    //              case 5:
    //                  if (L1_input_output[j] <= tmp_func_para_sorted[1])
    //                      L2_output[j][k] = 1.0;
    //                  else if (L1_input_output[j] <= (tmp_func_para_sorted[2] + tmp_func_para_sorted[3]) / 2.0)
    //                      L2_output[j][k] = 1.0 - 2.0 * (L1_input_output[j] - tmp_func_para_sorted[1]) / (tmp_func_para_sorted[4] - tmp_func_para_sorted[1]) * (L1_input_output[j] - tmp_func_para_sorted[1]) / (tmp_func_para_sorted[4] - tmp_func_para_sorted[1]);
    //                  else if (L1_input_output[j] <= tmp_func_para_sorted[4])
    //                      L2_output[j][k] = 2.0 * (tmp_func_para_sorted[4] - L1_input_output[j]) / (tmp_func_para_sorted[4] - tmp_func_para_sorted[1]) * (L1_input_output[j] - tmp_func_para_sorted[1]) / (tmp_func_para_sorted[4] - tmp_func_para_sorted[1]);
    //                  else
    //                      L2_output[j][k] = 0.0;
    //                  break;
    //              default:
    //                  break;
    //              }
    //          }
    //          else {
    //              L2_output[j][k] = 0.0;
    //          }
    //      }
    //  }
    //  //Layer 3
    //  memset(L3_fuzzy_rule_values, 0, MAX_NUM_FUZZY_RULE * sizeof(double));
    //  int step1 = MAX_NUM_CLUSTER_FRNN;
    //  int step2 = step1*MAX_NUM_CLUSTER_FRNN;
    //  int step3 = step2*MAX_NUM_CLUSTER_FRNN;
    //  int step4 = step3*MAX_NUM_CLUSTER_FRNN;
    //  int step5 = step4*MAX_NUM_CLUSTER_FRNN;
    //  int idx_fuzzy_rule = 0;
    //  for (int a = 0; a < MAX_NUM_CLUSTER_FRNN; a++) {
    //      int flag1 = flag_fuzzy_mem_func[0][a];
    //      if (!flag1) continue;
    //      double con1 = L2_output[0][a];
    //      for (int b = 0; b < MAX_NUM_CLUSTER_FRNN; b++) {
    //          int flag2 = flag_fuzzy_mem_func[1][b];
    //          if (!flag2) continue;
    //          double con2 = L2_output[1][b];
    //          for (int c = 0; c < MAX_NUM_CLUSTER_FRNN; c++) {
    //              int flag3 = flag_fuzzy_mem_func[2][c];
    //              if (!flag3) continue;
    //              double con3 = L2_output[2][c];
    //              for (int d = 0; d < MAX_NUM_CLUSTER_FRNN; d++) {
    //                  int flag4 = flag_fuzzy_mem_func[3][d];
    //                  if (!flag4) continue;
    //                  double con4 = L2_output[3][d];
    //                  for (int e = 0; e < MAX_NUM_CLUSTER_FRNN; e++) {
    //                      int flag5 = flag_fuzzy_mem_func[4][e];
    //                      if (!flag5) continue;
    //                      double con5 = L2_output[4][e];
    //                      for (int f = 0; f < MAX_NUM_CLUSTER_FRNN; f++) {
    //                          int flag6 = flag_fuzzy_mem_func[5][f];
    //                          if (!flag6) continue;
    //                          double con6 = L2_output[5][f];
    //                          idx_fuzzy_rule = a*step5 + b*step4 + c*step3 + d*step2 + e*step1 + f;
    //                          if (!flag_fuzzy_rule[idx_fuzzy_rule]) continue;
    //                          L3_fuzzy_rule_values[idx_fuzzy_rule] =
    //                              con1*con2*con3*con4*con5*con6;
    //                      }
    //                  }
    //              }
    //          }
    //      }
    //  }
    //  double tmp_sum_L4 = 0.0;
    //  //Layer 4
    //  for (int j = 0; j < NUM_CLASS_FRNN; j++) {
    //      L4_output[j] = 0.0;
    //      for (int k = 0; k < MAX_NUM_FUZZY_RULE; k++) {
    //          if (flag_fuzzy_rule[k] && flag_fuzzy_rough[j][k])
    //              L4_output[j] += para_fuzzy_rough[j][k] * L3_fuzzy_rule_values[k];
    //      }
    //      tmp_sum_L4 += L4_output[j];
    //  }
    //  //
    //  for (int j = 0; j < NUM_CLASS_FRNN; j++) {
    //      if (tmp_sum_L4 > 0)
    //          L4DataAllSamples(j, i - offset_data[N_INPUT_FRNN - 1]) = L4_output[j] / tmp_sum_L4;
    //      else
    //          L4DataAllSamples(j, i - offset_data[N_INPUT_FRNN - 1]) = L4_output[j];
    //  }
    //  O_targetOutput(i - offset_data[N_INPUT_FRNN - 1], 0) = trainData[i + 1];
    //  ////Layer 5
    //  //L5_output_y = 0.0;
    //  //for (int j = 0; j < NUM_CLASS_FRNN; j++) {
    //  //  double tmp = para_class_output[j] * L4_output[j];// *L1_input_output[j];
    //  //  //double tmp = L4_output[j] * L1_input_output[j];
    //  //  if (tmp_sum_L4 > 0)
    //  //      tmp /= tmp_sum_L4;
    //  //  L5_output_y += tmp;
    //  //}
    //  //// Error
    //  //f_prcsn += (L5_output_y - trainData[i + 1]) * (L5_output_y - trainData[i + 1]);
    //}

    //MatrixXd L4TL4 = L4DataAllSamples * L4DataAllSamples.transpose();
    //MatrixXd L4TY = L4DataAllSamples * O_targetOutput;
    //JacobiSVD<MatrixXd> svd(L4TL4, ComputeThinU | ComputeThinV);
    //MatrixXd L5Weights = svd.solve(L4TY);
    ////cout << L5Weights << endl;

    //for (int i = 0; i < NUM_CLASS_FRNN; i++) {
    //  para_class_output[i] = L5Weights(i, 0);
    //}

    //MatrixXd L4DataAllSamples(NUM_CLASS_FRNN, trainDataSize - offset_data[N_INPUT_FRNN - 1] - 1);
    //MatrixXd O_targetOutput(trainDataSize - offset_data[N_INPUT_FRNN - 1] - 1, 1);

    for(int i = offset_data[N_INPUT_FRNN - 1]; i < dataLen - 1; i++) {
        //Layer 1
        for(int j = 0; j < N_INPUT_FRNN; j++) {
            L1_input_output[j] = dataSet[i - offset_data[j]];
        }
        //Layer 2
        for(int j = 0; j < N_INPUT_FRNN; j++) {
            double tmp_func_para[MAX_NUM_CLUSTER_MEM_PPARA];
            double tmp_func_para_sorted[MAX_NUM_CLUSTER_MEM_PPARA];
            for(int k = 0; k < MAX_NUM_CLUSTER_FRNN; k++) {
                if(flag_fuzzy_mem_func[j][k]) {
                    for(int l = 0; l < MAX_NUM_CLUSTER_MEM_PPARA; l++) {
                        tmp_func_para[l] = para_fuzzy_mem_func[j][k][l];
                        tmp_func_para_sorted[l] = tmp_func_para[l];
                    }
                    for(int a = MAX_NUM_CLUSTER_MEM_PPARA - 4; a < MAX_NUM_CLUSTER_MEM_PPARA; a++) {
                        for(int b = a + 1; b < MAX_NUM_CLUSTER_MEM_PPARA; b++) {
                            if(tmp_func_para_sorted[a] > tmp_func_para_sorted[b]) {
                                double tmp_d = tmp_func_para_sorted[a];
                                tmp_func_para_sorted[a] = tmp_func_para_sorted[b];
                                tmp_func_para_sorted[b] = tmp_d;
                            }
                        }
                    }
                    double c0 = tmp_func_para_sorted[0];
                    double delta = tmp_func_para_sorted[1];
                    double alpha = fabs(tmp_func_para_sorted[2]) + 1e-6;
                    double beta = tmp_func_para_sorted[3];
                    double gamma = tmp_func_para_sorted[4];
                    double p_a = tmp_func_para_sorted[5];
                    double p_b = tmp_func_para_sorted[6];
                    double p_c = tmp_func_para_sorted[7];
                    double p_d = tmp_func_para_sorted[8];
                    double tmp1;
                    double tmp2;
                    switch(type_fuzzy_mem_func[j][k]) {
                    case 0:
                        L2_output[j][k] = exp(-(L1_input_output[j] - c0) * (L1_input_output[j] - c0) / 2.0 / (delta * delta));
                        break;
                    case 1:
                        tmp1 = (L1_input_output[j] - c0) / alpha;
                        tmp1 = tmp1 * tmp1;
                        tmp2 = beta;
                        if(tmp2 < 0) tmp2 = -tmp2;
                        if(tmp2 == 0) tmp2 = 1e-6;
                        tmp1 = pow(tmp1, tmp2);
                        L2_output[j][k] = 1.0 / (1.0 + tmp1);
                        break;
                    case 2:
                        L2_output[j][k] = 1.0 / (1.0 + exp(-gamma * (L1_input_output[j] - c0)));
                        break;
                    case 3:
                        if(L1_input_output[j] <= p_a)
                            L2_output[j][k] = 0.0;
                        else if(L1_input_output[j] <= p_b)
                            L2_output[j][k] = (L1_input_output[j] - p_a) / (p_b - p_a);
                        else if(L1_input_output[j] <= p_c)
                            L2_output[j][k] = 1.0;
                        else if(L1_input_output[j] <= p_d)
                            L2_output[j][k] = (p_d - L1_input_output[j]) / (p_d - p_c);
                        else
                            L2_output[j][k] = 0.0;
                        break;
                    case 4:
                        if(L1_input_output[j] <= p_a)
                            L2_output[j][k] = 0.0;
                        else if(L1_input_output[j] <= (p_b + p_c) / 2.0)
                            L2_output[j][k] = (L1_input_output[j] - p_a) / ((p_b + p_c) / 2.0 - p_a);
                        else if(L1_input_output[j] <= p_d)
                            L2_output[j][k] = (p_d - L1_input_output[j]) / (p_d - (p_b + p_c) / 2.0);
                        else
                            L2_output[j][k] = 0.0;
                        break;
                    case 5:
                        if(L1_input_output[j] <= p_a)
                            L2_output[j][k] = 1.0;
                        else if(L1_input_output[j] <= (p_b + p_c) / 2.0)
                            L2_output[j][k] = 1.0 - 2.0 * (L1_input_output[j] - p_a) / (p_d - p_a) * (L1_input_output[j] - p_a) / (p_d - p_a);
                        else if(L1_input_output[j] <= p_d)
                            L2_output[j][k] = 2.0 * (p_d - L1_input_output[j]) / (p_d - p_a) * (p_d - L1_input_output[j]) / (p_d - p_a);
                        else
                            L2_output[j][k] = 0.0;
                        break;
                    default:
                        break;
                    }
                } else {
                    L2_output[j][k] = 0.0;
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
        memset(L3_fuzzy_rule_values, 0, MAX_NUM_FUZZY_RULE * sizeof(double));
        int step1 = MAX_NUM_CLUSTER_FRNN;
        int step2 = step1 * MAX_NUM_CLUSTER_FRNN;
        int step3 = step2 * MAX_NUM_CLUSTER_FRNN;
        int step4 = step3 * MAX_NUM_CLUSTER_FRNN;
        int step5 = step4 * MAX_NUM_CLUSTER_FRNN;
        int idx_fuzzy_rule = 0;
        for(int a = 0; a < MAX_NUM_CLUSTER_FRNN; a++) {
            int flag1 = flag_fuzzy_mem_func[0][a];
            if(!flag1) continue;
            double con1 = L2_output[0][a];
            for(int b = 0; b < MAX_NUM_CLUSTER_FRNN; b++) {
                int flag2 = flag_fuzzy_mem_func[1][b];
                if(!flag2) continue;
                double con2 = L2_output[1][b];
                for(int c = 0; c < MAX_NUM_CLUSTER_FRNN; c++) {
                    int flag3 = flag_fuzzy_mem_func[2][c];
                    if(!flag3) continue;
                    double con3 = L2_output[2][c];
                    for(int d = 0; d < MAX_NUM_CLUSTER_FRNN; d++) {
                        int flag4 = flag_fuzzy_mem_func[3][d];
                        if(!flag4) continue;
                        double con4 = L2_output[3][d];
                        for(int e = 0; e < MAX_NUM_CLUSTER_FRNN; e++) {
                            int flag5 = flag_fuzzy_mem_func[4][e];
                            if(!flag5) continue;
                            double con5 = L2_output[4][e];
                            //idx_fuzzy_rule = a*step4 + b*step3 + c*step2 + d*step1 + e;
                            //if (!flag_fuzzy_rule[idx_fuzzy_rule]) continue;
                            //L3_fuzzy_rule_values[idx_fuzzy_rule] =
                            //  con1*con2*con3*con4*con5;
                            for(int f = 0; f < MAX_NUM_CLUSTER_FRNN; f++) {
                                int flag6 = flag_fuzzy_mem_func[5][f];
                                if(!flag6) continue;
                                double con6 = L2_output[5][f];
                                idx_fuzzy_rule = a * step5 + b * step4 + c * step3 + d * step2 + e * step1 + f;
                                if(!flag_fuzzy_rule[idx_fuzzy_rule]) continue;
                                L3_fuzzy_rule_values[idx_fuzzy_rule] =
                                    con1 * con2 * con3 * con4 * con5 * con6;
                            }
                        }
                    }
                }
            }
        }
        double tmp_sum_L4 = 0.0;
        //Layer 4
        for(int j = 0; j < NUM_CLASS_FRNN; j++) {
            L4_output[j] = 0.0;
            for(int k = 0; k < MAX_NUM_FUZZY_RULE; k++) {
                if(flag_fuzzy_rule[k] && flag_fuzzy_rough[j][k])
                    L4_output[j] += para_fuzzy_rough[j][k] * L3_fuzzy_rule_values[k];
            }
            tmp_sum_L4 += L4_output[j];
        }
        ////
        //for (int j = 0; j < NUM_CLASS_FRNN; j++) {
        //  if (tmp_sum_L4 > 0)
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_FRNN - 1]) = L4_output[j] / tmp_sum_L4;
        //  else
        //      L4DataAllSamples(j, i - offset_data[N_INPUT_FRNN - 1]) = L4_output[j];
        //}
        //O_targetOutput(i - offset_data[N_INPUT_FRNN - 1], 0) = trainData[i + 1];
        //Layer 5
        L5_output_y = 0.0;
        for(int j = 0; j < NUM_CLASS_FRNN; j++) {
            double tmp = para_class_output[j] * L4_output[j];// *L1_input_output[j];
            //double tmp = L4_output[j] * L1_input_output[j];
            if(tmp_sum_L4 > 0)
                tmp /= tmp_sum_L4;
            L5_output_y += tmp;
        }
        ////
        //double tmp_mean = 0.0;
        //double tmp_std = 0.0;
        //for (int j = 0; j < N_INPUT_FRNN; j++) {
        //  tmp_mean += L1_input_output[j];
        //}
        //tmp_mean /= N_INPUT_FRNN;
        //for (int j = 0; j < N_INPUT_FRNN; j++) {
        //  tmp_std += (L1_input_output[j] - tmp_mean)*(L1_input_output[j] - tmp_mean);
        //}
        //tmp_std = sqrt(tmp_std / N_INPUT_FRNN);
        //L5_output_y = L5_output_y*tmp_std + tmp_mean;
        // Error
        f_prcsn += (L5_output_y - dataSet[i + 1]) * (L5_output_y - dataSet[i + 1]);
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
    //for (int i = offset_data[N_INPUT_FRNN - 1]; i < trainDataSize - 1; i++) {
    //  // Error
    //  f_prcsn += (O_predicted(0, i - offset_data[N_INPUT_FRNN - 1]) - trainData[i + 1]) *
    //      (O_predicted(0, i - offset_data[N_INPUT_FRNN - 1]) - trainData[i + 1]);
    //}
    f_prcsn = sqrt(f_prcsn / (dataLen - offset_data[N_INPUT_FRNN - 1] - 1));

    return f_prcsn;
}
