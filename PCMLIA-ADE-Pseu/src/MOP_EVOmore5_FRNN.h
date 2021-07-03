#ifndef _MOP_EVOMORE5_FRNN_
#define _MOP_EVOMORE5_FRNN_

//////////////////////////////////////////////////////////////////////////
//EVOmore5_FRNN - fuzzy rough neural network
#define N_INPUT_EVOmore5_FRNN 3
#define N_IN_RSD_EVOmore5_FRNN (N_INPUT_EVOmore5_FRNN - 1)
#define N_INPUT_ALL_EVOmore5_FRNN (N_INPUT_EVOmore5_FRNN + N_IN_RSD_EVOmore5_FRNN)
#define N_OUTPUT_EVOmore5_FRNN 1
#define MAX_NUM_CLUSTER_EVOmore5_FRNN 3
#define MIN_NUM_CLUSTER_EVOmore5_FRNN 1
#define MAX_NUM_CLUSTER_MEM_PPARA_EVOmore5_FRNN (9 + 1)
#define NUM_CLASS_EVOmore5_FRNN 3
#define MAX_NUM_FUZZY_RULE_EVOmore5_FRNN (2 * N_INPUT_ALL_EVOmore5_FRNN + 1)
#define DIM_CLUSTER_BIN_RULE_EVOmore5_FRNN (MAX_NUM_FUZZY_RULE_EVOmore5_FRNN * N_INPUT_ALL_EVOmore5_FRNN)
#define DIM_CLUSTER_MEM_EVOmore5_FRNN (N_INPUT_ALL_EVOmore5_FRNN * MAX_NUM_CLUSTER_EVOmore5_FRNN)
#define DIM_CLUSTER_MEM_TYPE_EVOmore5_FRNN DIM_CLUSTER_MEM_EVOmore5_FRNN
#define DIM_CLUSTER_MEM_PARA_EVOmore5_FRNN (MAX_NUM_CLUSTER_MEM_PPARA_EVOmore5_FRNN * DIM_CLUSTER_MEM_EVOmore5_FRNN)
#define DIM_FUZZY_BIN_ROUGH_EVOmore5_FRNN (NUM_CLASS_EVOmore5_FRNN * MAX_NUM_FUZZY_RULE_EVOmore5_FRNN)
#define DIM_FUZZY_ROUGH_MEM_PARA_EVOmore5_FRNN (NUM_CLASS_EVOmore5_FRNN * MAX_NUM_FUZZY_RULE_EVOmore5_FRNN)
#define NUM_PARA_CONSE_EVOmore5_FRNN (N_INPUT_EVOmore5_FRNN + 1)
#define DIM_CONSEQUENCE_EVOmore5_FRNN (NUM_CLASS_EVOmore5_FRNN * (2 * NUM_PARA_CONSE_EVOmore5_FRNN))
#define DIM_CONSE_WEIGHT_EVOmore5_FRNN 1

#define DIM_EVOmore5_FRNN (DIM_CLUSTER_MEM_TYPE_EVOmore5_FRNN + DIM_CLUSTER_BIN_RULE_EVOmore5_FRNN + DIM_FUZZY_BIN_ROUGH_EVOmore5_FRNN + DIM_CLUSTER_MEM_PARA_EVOmore5_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore5_FRNN + DIM_CONSEQUENCE_EVOmore5_FRNN + DIM_CONSE_WEIGHT_EVOmore5_FRNN)
#define DIM_OBJ_EVOmore5_FRNN 2

void Initialize_data_EVOmore5_FRNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Fitness_EVOmore5_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M);
void Fitness_EVOmore5_FRNN_testSet(double* individual, double* fitness);
void SetLimits_EVOmore5_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_EVOmore5_FRNN(double* x, int nx);

#endif
