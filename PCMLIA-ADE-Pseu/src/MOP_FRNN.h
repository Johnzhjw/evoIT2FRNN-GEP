#ifndef _MOP_FRNN_
#define _MOP_FRNN_

//////////////////////////////////////////////////////////////////////////
//FRNN - fuzzy rough neural network
#define N_INPUT_FRNN 6
#define N_OUTPUT_FRNN 1
#define MAX_NUM_CLUSTER_FRNN 3
#define MIN_NUM_CLUSTER_FRNN 1
#define MAX_NUM_CLUSTER_MEM_PPARA 9
#define NUM_CLASS_FRNN 3
#define MAX_NUM_FUZZY_RULE (3*3*3*3*3*3)
#define DIM_CLUSTER_MEM (N_INPUT_FRNN * MAX_NUM_CLUSTER_FRNN)
#define DIM_CLUSTER_MEM_TYPE DIM_CLUSTER_MEM
#define DIM_CLUSTER_MEM_PARA (MAX_NUM_CLUSTER_MEM_PPARA * DIM_CLUSTER_MEM)
#define DIM_FUZZY_ROUGH (NUM_CLASS_FRNN * MAX_NUM_FUZZY_RULE)
#define DIM_FUZZY_ROUGH_MEM_PARA (NUM_CLASS_FRNN * MAX_NUM_FUZZY_RULE)

#define DIM_FRNN (DIM_CLUSTER_MEM + DIM_CLUSTER_MEM_TYPE + DIM_FUZZY_ROUGH + DIM_CLUSTER_MEM_PARA + DIM_FUZZY_ROUGH_MEM_PARA + NUM_CLASS_FRNN)
#define DIM_OBJ_FRNN 2

void Initialize_data_FRNN(int curN, int numN, int trainNo, int testNo, int endNo);
void Fitness_FRNN(double* individual, double* fitness, double *constrainV, int nx, int M);
void Fitness_FRNN_testSet(double* individual, double* fitness);
void SetLimits_FRNN(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_FRNN(double* x, int nx);

#endif
