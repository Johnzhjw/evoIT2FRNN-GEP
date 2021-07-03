#ifndef _MOP_EVOMORE4_FRNN_
#define _MOP_EVOMORE4_FRNN_

//////////////////////////////////////////////////////////////////////////
//EVOmore4_FRNN - fuzzy rough neural network
#define N_INPUT_EVOmore4_FRNN 3
#define N_IN_RSD_EVOmore4_FRNN (N_INPUT_EVOmore4_FRNN - 1)
#define N_INPUT_ALL_EVOmore4_FRNN (N_INPUT_EVOmore4_FRNN + N_IN_RSD_EVOmore4_FRNN)
#define N_OUTPUT_EVOmore4_FRNN 1
#define MAX_NUM_CLUSTER_EVOmore4_FRNN 3
#define MIN_NUM_CLUSTER_EVOmore4_FRNN 1
#define MAX_NUM_CLUSTER_MEM_PPARA_EVOmore4_FRNN 9
#define NUM_CLASS_EVOmore4_FRNN 3
#define MAX_NUM_FUZZY_RULE_EVOmore4_FRNN (2 * N_INPUT_ALL_EVOmore4_FRNN + 1)
#define DIM_CLUSTER_BIN_RULE_EVOmore4_FRNN (MAX_NUM_FUZZY_RULE_EVOmore4_FRNN * N_INPUT_ALL_EVOmore4_FRNN)
#define DIM_CLUSTER_MEM_EVOmore4_FRNN (N_INPUT_ALL_EVOmore4_FRNN * MAX_NUM_CLUSTER_EVOmore4_FRNN)
#define DIM_CLUSTER_MEM_TYPE_EVOmore4_FRNN DIM_CLUSTER_MEM_EVOmore4_FRNN
#define DIM_CLUSTER_MEM_PARA_EVOmore4_FRNN (MAX_NUM_CLUSTER_MEM_PPARA_EVOmore4_FRNN * DIM_CLUSTER_MEM_EVOmore4_FRNN)
#define DIM_FUZZY_BIN_ROUGH_EVOmore4_FRNN (NUM_CLASS_EVOmore4_FRNN * MAX_NUM_FUZZY_RULE_EVOmore4_FRNN)
#define DIM_FUZZY_ROUGH_MEM_PARA_EVOmore4_FRNN (NUM_CLASS_EVOmore4_FRNN * MAX_NUM_FUZZY_RULE_EVOmore4_FRNN)
#define NUM_PARA_CONSE_EVOmore4_FRNN (N_INPUT_EVOmore4_FRNN + 1)
#define DIM_CONSEQUENCE_EVOmore4_FRNN (NUM_CLASS_EVOmore4_FRNN * NUM_PARA_CONSE_EVOmore4_FRNN)

#define DIM_EVOmore4_FRNN (DIM_CLUSTER_MEM_TYPE_EVOmore4_FRNN + DIM_CLUSTER_BIN_RULE_EVOmore4_FRNN + DIM_FUZZY_BIN_ROUGH_EVOmore4_FRNN + DIM_CLUSTER_MEM_PARA_EVOmore4_FRNN + DIM_FUZZY_ROUGH_MEM_PARA_EVOmore4_FRNN + DIM_CONSEQUENCE_EVOmore4_FRNN)
#define DIM_OBJ_EVOmore4_FRNN 2

void Initialize_data_EVOmore4_FRNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Fitness_EVOmore4_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M);
void Fitness_EVOmore4_FRNN_testSet(double* individual, double* fitness);
void SetLimits_EVOmore4_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_EVOmore4_FRNN(double* x, int nx);

#endif
