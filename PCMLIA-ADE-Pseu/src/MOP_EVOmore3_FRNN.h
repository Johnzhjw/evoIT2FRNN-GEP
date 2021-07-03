#ifndef _MOP_EVOMORE3_FRNN_
#define _MOP_EVOMORE3_FRNN_

//////////////////////////////////////////////////////////////////////////
//EVOmore3_FRNN - fuzzy rough neural network
#define N_INPUT_EVOmore3_FRNN 3
#define N_IN_RSD_EVOmore3_FRNN (N_INPUT_EVOmore3_FRNN - 1)
#define N_INPUT_ALL_EVOmore3_FRNN (N_INPUT_EVOmore3_FRNN + N_IN_RSD_EVOmore3_FRNN)
#define N_OUTPUT_EVOmore3_FRNN 1
#define MAX_NUM_CLUSTER_EVOmore3_FRNN 3
#define MIN_NUM_CLUSTER_EVOmore3_FRNN 1
#define MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN 9
#define NUM_CLASS_EVOmore3_FRNN 3
#define MAX_NUM_FUZZY_RULE_EVOmore3_FRNN (2 * N_INPUT_ALL_EVOmore3_FRNN + 1)
#define DIM_CLUSTER_BIN_RULE_EVOmore3_FRNN (MAX_NUM_FUZZY_RULE_EVOmore3_FRNN * N_INPUT_ALL_EVOmore3_FRNN)
#define DIM_CLUSTER_MEM_EVOmore3_FRNN (N_INPUT_ALL_EVOmore3_FRNN * MAX_NUM_CLUSTER_EVOmore3_FRNN)
#define DIM_CLUSTER_MEM_TYPE_EVOmore3_FRNN DIM_CLUSTER_MEM_EVOmore3_FRNN
#define DIM_CLUSTER_MEM_PARA_EVOmore3_FRNN (MAX_NUM_CLUSTER_MEM_PPARA_EVOmore3_FRNN * DIM_CLUSTER_MEM_EVOmore3_FRNN)
#define DIM_FUZZY_BIN_ROUGH_EVOmore3_FRNN (NUM_CLASS_EVOmore3_FRNN * MAX_NUM_FUZZY_RULE_EVOmore3_FRNN)
#define DIM_FUZZY_ROUGH_MEM_PARA_EVOmore3_FRNN (NUM_CLASS_EVOmore3_FRNN * MAX_NUM_FUZZY_RULE_EVOmore3_FRNN)

#define DIM_EVOmore3_FRNN (DIM_CLUSTER_MEM_TYPE_EVOmore3_FRNN + DIM_CLUSTER_BIN_RULE_EVOmore3_FRNN + DIM_FUZZY_BIN_ROUGH_EVOmore3_FRNN + DIM_CLUSTER_MEM_PARA_EVOmore3_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore3_FRNN + NUM_CLASS_EVOmore3_FRNN)
#define DIM_OBJ_EVOmore3_FRNN 2

void Initialize_data_EVOmore3_FRNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Fitness_EVOmore3_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M);
void Fitness_EVOmore3_FRNN_testSet(double* individual, double* fitness);
void SetLimits_EVOmore3_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_EVOmore3_FRNN(double* x, int nx);

#endif
