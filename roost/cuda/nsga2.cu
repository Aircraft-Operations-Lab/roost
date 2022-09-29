#include <math_constants.h>
#include <math.h>
#include <helper_math.h>

#define N_trjs {{n_pop}}
#define N_objectives {{n_objectives}}
#define N_J_padded {{n_objectives_padded}}

extern "C" { // because we compile with no_extern_c=True, need this to 
             // prevent name mangling when recovering the function
    __global__ void ndsort_compute_Sp(float objectives[N_trjs][N_J_padded], int Sp[N_trjs][N_trjs])
    {
        int p = blockIdx.x;
        int q = threadIdx.x;
        int dominates = 1;
        for (int i=0; i<N_objectives; i++) {
            float J_p = objectives[p][i];
            float J_q = objectives[q][i];
            int dominates_i = (J_p < J_q) ? 1 : 0;
            dominates *= dominates_i;
        }
        Sp[p][q] = dominates;
// 		float fuel_burn_p = objectives[p][0];
// 		float flight_time_p = objectives[p][1];
// 		float fmax_p = objectives[p][3];
// 		float fmin_p = objectives[p][2];
// 		
// 		float fuel_burn_q = objectives[q][0];
// 		float flight_time_q = objectives[q][1];
// 		float fmax_q = objectives[q][3];
// 		float fmin_q = objectives[q][2];
// 			
// 		bool wdiff = ( fmax_p - fmin_p ) < ( fmax_q - fmin_q );
// 		int dominates_fb =  (fuel_burn_p < fuel_burn_q) ? 1 : 0;
// 		int dominates_ft = (flight_time_p < flight_time_q) ? 1 : 0;
// 		//int dominates_CI = (fuel_burn_p + {{CI}}*flight_time_p < fuel_burn_q + {{CI}}*flight_time_q) ? 1 : 0;
// 		int dominates_w = (wdiff) ? 1 : 0;
// 		Sp[p][q] = dominates_fb*dominates_ft*dominates_w;
		//Sp[p][q] = dominates_CI*dominates_w;
    }
    __global__ void ndsort_compute_np(int Sp[N_trjs][N_trjs], float np[N_trjs], int F[N_trjs])
    {
        int i = threadIdx.x;
        int n = 0;
        for (int j=0; j<N_trjs; j++) {
            n += Sp[j][i];
        }
        np[i] = n;
        if (n == 0) {
            F[i] = 0;
        }
    }
    __global__ void ndsort_propagate_front(int Sp[N_trjs][N_trjs], float np[N_trjs], int F[N_trjs], int i)
    {
        int q = threadIdx.x;
        int nq = np[q];
        if (F[q] == -1) {
            for (int p=0; p<N_trjs; p++) {
                if ( (Sp[p][q] == 1) && (F[p] == i)) {
                    nq--;
                }
            }
            np[q] = nq;
            if (nq == 0) {
                F[q] = i + 1;
            }
        }
    }
}


