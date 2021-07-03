#ifndef _MOP_IWSN_SECURITY_1F_OLD_
#define _MOP_IWSN_SECURITY_1F_OLD_

//////////////////////////////////////////////////////////////////////////
//IWSN_Security_1F - One floor old
#define N_DIREC_S_1F_OLD 30
#define N_RELAY_S_1F_OLD 10
#define D_DIREC_S_1F_OLD 4
#define D_RELAY_S_1F_OLD 2
#define N_S_KEYS_OLD     3

#define DIM_IWSN_S_1F_OLD (N_DIREC_S_1F_OLD * D_DIREC_S_1F_OLD + N_RELAY_S_1F_OLD * D_RELAY_S_1F_OLD + N_S_KEYS_OLD)
#define DIM_OBJ_IWSN_S_1F_OLD 5

void Fitness_IWSN_S_1F_OLD(double* individual, double* fitness, double *constrainV, int nx, int M);
void SetLimits_IWSN_S_1F_OLD(double* minLimit, double* maxLimit, int nx);
int  CheckLimits_IWSN_S_1F_OLD(double* x, int nx);

#endif
