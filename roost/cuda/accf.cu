__device__ float compute_EI_NOx(float fuel_flow, float Mach, float C1, float C2)
{
    float ff_ci = C1 * (fuel_flow/2) * __expf(0.2f * __powf(Mach, 2));
    return C2 * __expf(  3.215f +  0.7783f * __logf(ff_ci));
}

__device__ float compute_accf_o3(float t, float z)
{
    float accf_o3 = 15 * (-5.20e-11f + 2.30e-13f * t + 4.85e-16f * z - 2.04e-18f * t * z) * 1.37f / 11;
    return accf_o3;
}

__device__ float compute_accf_ch4(float z, float fin)
{
    float accf_ch4 = 10.5f * (-9.83e-13f + 1.99e-18f * z - 6.32e-16f * fin + 6.12e-21f * z * fin)* 1.29f * 1.18f / 35;
    return accf_ch4;
}

__device__ float compute_accf_h2o(float pv)
{
    float accf_h2o =  15 * (4.05e-16f + 1.48e-10f * pv) / 3;
    return accf_h2o;
}

__device__ float compute_accf_dCont (float olr)
{
    float accf_dCont = 13.9f * 0.0151f * 1e-10f * (-1.7f - 0.0088f * olr) * 0.42f/ 3;
    return accf_dCont;
}

__device__ float compute_accf_nCont (float t)
{
    float accf_nCont = 13.9f * 0.0151f * (1e-10f * (0.0073f * __powf(10, 0.0107f * t) - 1.03f)) * 0.42f/ 3;
    return accf_nCont;
}
