#define TAS_STEP 10 // usando 10 m/s por 0.7 deg de paso
#define LAT_IDX 0
#define LON_IDX 1
#define TAS_IDX 2
#define FL_IDX 3
#define COURSE_IDX 4
#define D2NEXT_IDX 5

#define PHASE_CLIMB 0
#define PHASE_LEVELOFF 1
#define PHASE_ACCEL 2
#define PHASE_CRUISE 3
#define PHASE_DECEL 4
#define PHASE_DESCENT 5
#define PHASE_LEVELOFF_2 6
#define PHASE_HOLD 7
//
//{% for i in range(n_members) %}
//    texture<float4, cudaTextureType3D> T_{{i}};
//    texture<float4, cudaTextureType3D> U_{{i}};
//    texture<float4, cudaTextureType3D> V_{{i}};
//    texture<float4, cudaTextureType3D> T_lower_{{i}};
//    texture<float4, cudaTextureType3D> U_lower_{{i}};
//    texture<float4, cudaTextureType3D> V_lower_{{i}};
//{% endfor %}

__device__ float lerp_stable(float a, float b, float frac)
{
    return a*(1.0f - frac) + b*frac;
}

__device__ EnvironmentVars env_from_values_P_Hp(float4 values, float P, float Hp)
{
    //float4 values = get_wxcube_0(wx_m, apos.lat_deg, apos.lon_deg, apos.t, P);
    float2 wind = make_float2(values.y, values.z);
    return EnvironmentVars(P, values.x, Hp, wind);
}

__device__ EnvironmentVars env_from_apos(int wx_m, AircraftPosition apos)
{
    float P = Hp2P(apos.Hp);
    float4 values = get_wxcube_0(wx_m, apos.lat_deg, apos.lon_deg, apos.t, P);
    return env_from_values_P_Hp(values, P, apos.Hp);
}

__device__ float4 env_var(int wx_m, AircraftPosition apos)
{
    float P = Hp2P(apos.Hp);
    return get_wxcube_0(wx_m, apos.lat_deg, apos.lon_deg, apos.t, P);
}

__device__ float4 accf_1_from_apos(int wx_m, AircraftPosition apos)
{
    float P = Hp2P(apos.Hp);
    return get_wxcube_1(wx_m, apos.lat_deg, apos.lon_deg, apos.t, P);
}

__device__ float4 accf_2_from_apos(int wx_m, AircraftPosition apos)
{
    float P = Hp2P(apos.Hp);
    return get_wxcube_2(wx_m, apos.lat_deg, apos.lon_deg, apos.t, P);
}

__device__ float sigmoid_threshold(float cr)
{
    return 0.5 + 0.5*cr/sqrtf(1 + cr*cr);
}

__device__ int get_d2go_idx()
{
    return {{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}};
}

__device__ int get_local_d2go_idx(int edge_idx)
{
    return {{n_crossroad_vars+2*n_edges}} + edge_idx;
}

__device__ int get_climb_coeff(int level)
{
    return {{n_crossroad_vars+3*n_edges}} + level;
}

__device__ int get_desc_coeff(int level)
{
    return {{n_crossroad_vars + 2*climb_descent_profile_coeffs}} + level;
}

extern "C" {

    // ----------------------------------------------
   __global__ void update_fp_vars(float rewards [{{n_plans}}],
                                  float random_samples[{{n_directions}}][{{n_flight_plan_vars}}],
                                  float rewards_mean_and_std[2],
                                  float flight_plan_vars[{{n_flight_plan_vars}}],
                                  float velocity[{{n_flight_plan_vars}}],
                                  float step_size,
                                  float velocity_beta)
   {
        __shared__ float volatile sdata[{{n_directions_c2}}];
        int tid = threadIdx.x;  // n directions
        int j = blockIdx.x;     // n flight plan vars
        float x_update;
        float nu;
        if (rewards_mean_and_std[1] > 0) {
            if (tid < {{n_directions}}) {
                sdata[tid] = (rewards[2*tid] - rewards[2*tid+1]) * random_samples[tid][j];
            } else {
                sdata[tid] = 0;
            }
            __syncthreads();

            // do reduction in shared mem
            for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid < s){
                    sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if(tid == 0) {
                if (j<{{n_crossroad_vars}}){
                    nu = {{nu_cr}};
                } else if ({{n_crossroad_vars}}<=j && j<({{n_crossroad_vars+n_edges}})){
                    nu = {{nu_M}};
                } else if ({{n_crossroad_vars+n_edges}}<=j && j<({{n_crossroad_vars+2*n_edges}})){
                    nu = {{nu_fl}};
                } else if ({{n_crossroad_vars+2*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges}})){
                    nu = {{nu_d2g}};
                } else if ({{n_crossroad_vars+3*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}})){
                    nu = {{nu_tas}};
                } else {
                    nu = {{nu_d2g}};
                }
                x_update = step_size / {{n_directions}}  * nu * sdata[0] / rewards_mean_and_std[1];
                x_update += velocity_beta*velocity[j];
                flight_plan_vars[j] = flight_plan_vars[j] + x_update;
                velocity[j] = x_update;
                // refactor
            }
        } else {
            if(tid == 0) {
                velocity[j] = 0;
            }
        }
   }

    // ----------------------------------------------
   __global__ void clamp_fp_vars(float flight_plan_vars[{{n_flight_plan_vars}}], float vlims[2][{{climb_descent_profile_coeffs}}])

   {
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i;
        float FL;
        float Mmin;
        float Mmax;
        int level;
        float decvar;


        if (j < {{n_edges}}) {
            // Clamp FL first
            i = {{n_crossroad_vars + n_edges}} + j;
            FL = flight_plan_vars[i];
            flight_plan_vars[i] = clamp(FL, (float) {{FL_min}}, (float) {{FL_max}});

            // Then clamp M
            i = {{n_crossroad_vars}} + j;
            Mmin = {{M_low_FL_min}} + {{(M_low_FL_max - M_low_FL_min)/(FL_max - FL_min)}}*(FL - {{FL_min}});
            Mmax = {{M_high_FL_min}} + {{(M_high_FL_max - M_high_FL_min)/(FL_max - FL_min)}}*(FL - {{FL_min}});
            decvar = flight_plan_vars[i];
            flight_plan_vars[i] = clamp(decvar, (float) Mmin,(float)  Mmax);

            // Clamp local d2g delta
            i = {{n_crossroad_vars + 2*n_edges}} + j;
            decvar = flight_plan_vars[i];
            flight_plan_vars[i] = clamp(decvar, (float)  -{{local_d2g_delta}}, (float) {{local_d2g_delta}});
        }

        if (j < {{climb_descent_profile_coeffs}}) {
            level = j;
            for (int k=0; k<2; k++) {
                i = {{n_crossroad_vars + 3*n_edges}} + j + {{climb_descent_profile_coeffs}}*k;
                decvar = flight_plan_vars[i];
                flight_plan_vars[i] = clamp(decvar, vlims[0][level], vlims[1][level]);
            }
        }

        if (j==0) { // distance to go for starting descent
            i = {{n_crossroad_vars + 3*n_edges + 2*climb_descent_profile_coeffs}};
            decvar = flight_plan_vars[i];
            flight_plan_vars[i] = clamp(decvar, (float) {{d2gdesc_min}}, (float) {{d2gdesc_max}});
        }

   }

    // ----------------------------------------------
      __global__ void initialize_fp_vars(float flight_plan_vars[{{n_flight_plan_vars}}], float vlims[2][{{climb_descent_profile_coeffs}}])

   {
        int j = threadIdx.x + blockDim.x * blockIdx.x;     // n flight plan vars
        int level;
        if (j < {{n_flight_plan_vars}}) {
            if (j<{{n_crossroad_vars}}){
                flight_plan_vars[j] = 0.0;
            } else if ({{n_crossroad_vars}}<=j && j<({{n_crossroad_vars}}+{{n_edges}})){
                flight_plan_vars[j] = 0.74;
            } else if ({{n_crossroad_vars+n_edges}}<=j && j<{{n_crossroad_vars+2*n_edges}}) {
                flight_plan_vars[j] = {{FL0_cruise}} + 20;
            } else if ({{n_crossroad_vars+2*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges}})){
                flight_plan_vars[j] = 0.0;
            } else if ({{n_crossroad_vars+3*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}})){
                level = (j - {{n_crossroad_vars+3*n_edges}}) % {{climb_descent_profile_coeffs}};
                flight_plan_vars[j] = 0.67*vlims[0][level] + 0.33*vlims[1][level];
            } else {
                flight_plan_vars[j] = 120;
            }
        }
   }





    // ----------------------------------------------
   __global__ void get_objectives(float summaries[{{n_plans}}][{{n_scenarios}}][10],
                               float rewards[{{n_plans}}],
                               float CI,
                               float climate_index)
   {
        __shared__ float volatile sdata[{{n_scenarios}}];
        int tid = threadIdx.x;  // scenario id
        int i = blockIdx.x; // plan id
        float avg_J;

        sdata[tid] = + {{cost_index}} * ( {{CI_fuel}} * summaries[i][tid][0] + CI * summaries[i][tid][1] );
        {% if perform_descent %}
        sdata[tid] += {{FL_penalty_coeff}}f * summaries[i][tid][2] * summaries[i][tid][2];
        {% endif %}
        {% if compute_accf %}
        sdata[tid] += climate_index * ( {{CI_nox}} * (summaries[i][tid][3] + summaries[i][tid][5]) + {{CI_h2o}} * summaries[i][tid][4] + 
        		{{CI_contrail}} * summaries[i][tid][6] + {{CI_co2}} * summaries[i][tid][7] + {{CI_contrail_dis}} * summaries[i][tid][8]);
        {% endif %}
        
        {% if compute_emissions %}
        sdata[tid] += {{emission_index}} * (summaries[i][tid][9]);
        {% endif %}
        
        
        //sdata[tid] = + summaries[i][tid][2];
        __syncthreads();

        // do reduction in shared mem
        for(int s=blockDim.x/2; s>0; s>>=1){
            if(tid < s){
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }
        // write result for this block to global mem
        avg_J = sdata[0] / {{n_scenarios}};
        {% if DI_contrails > 0 %} // compute standard deviation of contrails
            float avg_contrails;
            float cdiff;
            sdata[tid] = summaries[i][tid][6];
            __syncthreads();
            for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid < s){
                    sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
            }
            avg_contrails = sdata[0] / {{n_scenarios}};
            cdiff = (summaries[i][tid][6] - avg_contrails);
            sdata[tid] = cdiff*cdiff;
            __syncthreads();
            for(int s=blockDim.x/2; s>0; s>>=1){
                if(tid < s){
                    sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
            }
            if (tid == 0) {
                avg_J += {{DI_contrails}} * sqrtf(sdata[0] / {{n_scenarios}});
            }
        {% endif %}
        if (tid == 0) {
            rewards[i] = avg_J;
        }
   }

    // ----------------------------------------------
   __global__ void objectives_mean(float rewards[{{n_plans}}],
                        float rewards_mean_and_std[2])
   {
        __shared__ float volatile sdata[{{n_plans_c2}}];
        int tid = threadIdx.x;
        if (tid < {{n_plans}}) {
            sdata[tid] = rewards[tid];
        } else {
            sdata[tid] = 0;
        }
        __syncthreads();

        // do reduction in shared mem
        for(int s=blockDim.x/2; s>0; s>>=1){
            if(tid < s){
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }
        // write result for this block to global mem
        if(tid == 0) {
            rewards_mean_and_std[0] = sdata[0] / {{n_plans}};
        }
   }

    // ----------------------------------------------
   __global__ void objectives_stdev(float rewards[{{n_plans}}],
                       float rewards_mean_and_std[2])
   {
        __shared__ float volatile sdata[{{n_plans_c2}}];
        int tid = threadIdx.x;
        float variance;
        float rewards_mean = rewards_mean_and_std[0];
        float diff;
        if (tid < {{n_plans}}) {
            diff = rewards[tid] - rewards_mean;
            sdata[tid] = diff*diff;
        } else {
            sdata[tid] = 0;
        }
        __syncthreads();

        // do reduction in shared mem
        for(int s=blockDim.x/2; s>0; s>>=1){
            if(tid < s){
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }

        if(tid == 0) {
            variance = sdata[0] / {{n_plans}};
            if (variance > 0) {
                rewards_mean_and_std[1] = sqrtf(variance);
            } else {
                rewards_mean_and_std[1] = 0;
            }
        }

   }

    // ----------------------------------------------
   __global__ void generate_rollouts(float flight_plan_vars[{{n_flight_plan_vars}}],
                                     float random_samples[{{n_directions}}][{{n_flight_plan_vars}}],
                                     float decision_variables[{{n_plans}}][{{n_flight_plan_vars}}],
                                     float velocity[{{n_flight_plan_vars}}],
                                     float vlims[2][{{climb_descent_profile_coeffs}}],
                                     float beta_velocity,
                                     float noise_scaling)

   {
       // (1D) flight_plan_vars = (CR, TAS, FL)  / (n_crossroad_vars + n_edges + n_edges) = n_flight_plan_vars
       // (2D) random_samples -- n_flight_plan_vars x n_directions;   n_directions = n_plans / 2
       // (2D) decision_variables -- n_flight_plan_vars x n_plans

        int i = blockIdx.x;  // direction idx
        int tid = threadIdx.x; // decision variable index
        int pm = threadIdx.y; // whether it's plus or minus
        int j;
        int vars_per_thread = {{fp_vars_per_thread}};
        float nu;               // noise magnitude (depends on the flight plan variable)
        float increment;
        float sign;
        float decvar;
        float FL;
        float tas_min;
        float tas_max;
        int level;
        // int offset;

        __shared__ float volatile FL_array[{{n_edges}}];

        for (j = tid * vars_per_thread; j < (tid+1) * vars_per_thread; j++) {
            if (j<{{n_flight_plan_vars}}) {
                if (j<{{n_crossroad_vars}}){
                    nu = {{nu_cr}};
                } else if ({{n_crossroad_vars}}<=j && j<({{n_crossroad_vars+n_edges}})){
                    nu = {{nu_M}};
                } else if ({{n_crossroad_vars+n_edges}}<=j && j<({{n_crossroad_vars+2*n_edges}})){
                    nu = {{nu_fl}};
                } else if ({{n_crossroad_vars+2*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges}})){
                    nu = {{nu_d2g}};
                } else if ({{n_crossroad_vars+3*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}})){
                    nu = {{nu_tas}};
                } else {
                    nu = {{nu_d2g}};
                }
                increment = noise_scaling * nu * random_samples[i][j];
                sign = (float) 2*pm - 1;
                // offset = pm*{{n_directions}};
                {% if nesterov_acceleration %}
                    decvar = flight_plan_vars[j] + beta_velocity * velocity[j] + sign * increment;
                {% else %}
                    decvar = flight_plan_vars[j] + sign * increment;
                {% endif %}
                decision_variables[2*i + pm][j] = decvar;
            }
        }
        __syncthreads();

        // clamp FL first
        for (j = tid * vars_per_thread; j < (tid+1) * vars_per_thread; j++) {
            if (j<{{n_flight_plan_vars}}) {
                // clamp flight level
                if (j >= {{n_crossroad_vars + n_edges}} && j<{{n_crossroad_vars+2*n_edges}}){
                    decvar = decision_variables[2*i + pm][j];
                    decvar = fminf(decvar, {{FL_max}});
                    decvar = fmaxf(decvar, {{FL_min}});
                    FL_array[j - {{n_crossroad_vars}} - {{n_edges}}] = decvar;
                    decision_variables[2*i + pm][j] = decvar;
                }
            }
        }

        __syncthreads();

        // clamp Mach, climb/descent TAS and descent @ distance 2 go afterwards
        for (j = tid * vars_per_thread; j < (tid+1) * vars_per_thread; j++) {
            if (j<{{n_flight_plan_vars}}) {
                decvar = decision_variables[2*i + pm][j];
                if ({{n_crossroad_vars}}<=j && j<({{n_crossroad_vars}}+{{n_edges}})){
                    FL = FL_array[j - {{n_crossroad_vars}}];
                    tas_min = {{M_low_FL_min}} + {{(M_low_FL_max - M_low_FL_min)/(FL_max - FL_min)}}*(FL - {{FL_min}});
                    tas_max = {{M_high_FL_min}} + {{(M_high_FL_max - M_high_FL_min)/(FL_max - FL_min)}}*(FL - {{FL_min}});
                    decvar = fmaxf(tas_min, fminf(decvar, tas_max));
                }  else if ({{n_crossroad_vars+3*n_edges}}<=j && j<({{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}})){
                    level = (j - {{n_crossroad_vars+3*n_edges}}) % {{climb_descent_profile_coeffs}};
                    decvar = fmaxf(vlims[0][level], fminf(decvar, vlims[1][level]));
                } else if ({{n_crossroad_vars+3*n_edges + 2*climb_descent_profile_coeffs}} <= j){
                    decvar = fmaxf({{d2gdesc_min}}, fminf(decvar, {{d2gdesc_max}}));
                }
                decision_variables[2*i + pm][j] = decvar;
            }
        }
    }
   
      __global__ void evaluate_weather_grib_1block(const float *lats,
                                                const float *lons,
                                                const float FL,
                                                const float t,
                                                float *output
                                               )
    {
        float lat = lats[threadIdx.y];
        float lon = lons[threadIdx.x];
        AircraftPosition apos = ac_pos_from_FL(make_float2(lat, lon), FL, t);
        EnvironmentVars env = env_from_apos(0, apos);
        output[4*(blockDim.x*threadIdx.y + threadIdx.x)] = env.T;
        output[4*(blockDim.x*threadIdx.y + threadIdx.x) + 1] = env.wind.x;
        output[4*(blockDim.x*threadIdx.y + threadIdx.x) + 2] = env.wind.y;
        output[4*(blockDim.x*threadIdx.y + threadIdx.x) + 3] = -1;
    }
    
    __global__ void evaluate_tex(const int flag,
                                const float c1,
                                const float c2,
                                const float c3,
                                float *output
                                )
    {
        output[4*threadIdx.x] = tex3D(wxcube_0_0, c1, c2, c3).x;
        output[4*threadIdx.x + 1] = tex3D(wxcube_0_0, c1, c2, c3).y;
        output[4*threadIdx.x + 2] = tex3D(wxcube_0_0, c1, c2, c3).z;
        output[4*threadIdx.x + 3] = tex3D(wxcube_0_0, c1, c2, c3).w;

    }

   // ----------------------------------------------
   __global__ void generate_guidance_matrix(const float decision_variables[{{n_plans}}][{{n_flight_plan_vars}}],
                                            const float execution_noise[{{n_plans}}][{{n_scenarios}}][{{n_random_vars}}],
                                            const int nn2edge[{{n_nodes}}][{{n_nodes}}],
                                            const int crossroads_table[{{n_nodes}}][{{crt_row_length}}],
                                            const float edge_lengths[{{n_edges}}],
                                            const int edge_n_points[{{n_edges}}],
                                            int guidance[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4],
                                            float guidance_f[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4])
   {
        int i = blockIdx.x;  // flight_plan_idx
        int j = threadIdx.x; // sample idx
        int edge_idx;
        int current_node_idx;
        int next_node_idx;
        int n_successors;
        int req_crossroads;
        int crossroads_index;
        float crossroads_values[4]; // if successors > 4 then needs 3 or more
        current_node_idx = {{origin_idx}};
        int row[{{crt_row_length}}];
        int s_offset;
        float FL;
        //float old_FL;
        FL = {{FL0_cruise}};
        for (int k=0; k < {{max_nodes_per_path}}; k++) {
            if (current_node_idx != {{destination_idx}}) 
            {
                // compute the next node
                for (int l=0; l < {{crt_row_length}}; l++){
                    row[l] = crossroads_table[current_node_idx][l];
                }
                // assuming no more than 4 successors, thus hardcode
                n_successors = row[0];
                req_crossroads = 0;
                if (n_successors > 1) {
                    if (n_successors == 2) {
                        req_crossroads = 1;
                    } else if (n_successors < 5) {
                        req_crossroads = 2; // 3, 4 successors
                    } else if (n_successors < 9) {
                        req_crossroads = 3; // 5, 6, 7, 8 successors
                    } else {
                        req_crossroads = 4;
                    }
                }
                // __syncthreads;
                s_offset = 0;
                for (int l=0; l < req_crossroads; l++){
                    crossroads_index = row[l+1];
                    crossroads_values[l] = decision_variables[i][crossroads_index];
                    {% if cr_execution_noise_mode == "common" %}
                    if (sigmoid_threshold(crossroads_values[l]) > execution_noise[0][j][crossroads_index]) {
                    {% else %}
                    if (sigmoid_threshold(crossroads_values[l]) > execution_noise[i][j][crossroads_index]) {
                    {% endif %}
                        s_offset += (1<<l);
                    }
                }
                if (s_offset >= n_successors) {
                    s_offset = n_successors - 1;
                }
                // __syncthreads;
                next_node_idx = row[1 + req_crossroads + s_offset];
                edge_idx = nn2edge[current_node_idx][next_node_idx];
                
                FL = decision_variables[i][{{n_crossroad_vars}} + {{n_edges}} + edge_idx];


                // node: write down the current one
                // M: round to closest .01
                // FL
                // edge index
                guidance[i][k][j][0] = current_node_idx;
                guidance[i][k][j][1] = __float2int_rn(100*decision_variables[i][{{n_crossroad_vars}} + edge_idx]);
                guidance[i][k][j][2] = FL;
                guidance[i][k][j][3] = edge_idx;
                
                guidance_f[i][k][j][0] = edge_lengths[edge_idx];
                guidance_f[i][k][j][1] = edge_lengths[edge_idx]/(edge_n_points[edge_idx]-1);

                // distance-to-go-at-descent
                guidance_f[i][k][j][2] = decision_variables[i][get_d2go_idx()]; // +
                                         //decision_variables[i][get_local_d2go_idx(edge_idx)];

                current_node_idx = next_node_idx;
            } else {
                guidance[i][k][j][0] = current_node_idx;
            }
        }
   }
   

// ----------------------------------------------

    __global__ void clear_guidance(int guidance[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4],
                                  float guidance_f[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4])
    {
        for (int i=0; i<{{max_nodes_per_path}}; i++)
        {
            guidance[blockIdx.x][i][threadIdx.y][threadIdx.x] = 0;
        }
        for (int i=0; i<{{max_nodes_per_path}}; i++)
        {
            guidance_f[blockIdx.x][i][threadIdx.y][threadIdx.x] = 0.0f;
        }

    }

    // ----------------------------------------------

    __global__ void clear_profiles(float profiles[{{n_plans}}][{{max_subnodes_per_path}}][36][{{n_scenarios}}])
    {
        for (int i=0; i<{{max_subnodes_per_path}}; i++)
        {
            profiles[blockIdx.x][i][blockIdx.y][threadIdx.x] = 0.0f;
        }

    }
    // ----------------------------------------------
    __global__ void generate_4D_profile(const int guidance[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4],
                                        const float execution_noise[{{n_plans}}][{{n_scenarios}}][{{n_random_vars}}],
                                       const int edge_n_points[{{n_edges}}],
                                       const float guidance_f[{{n_plans}}][{{max_nodes_per_path}}][{{n_scenarios}}][4],
                                       const float edge_subnode_coordinates[{{n_edges}}][{{max_subpoints_per_edge}}][2],
                                       float profiles[{{n_plans}}][{{max_subnodes_per_path}}][36][{{n_scenarios}}])
   {

        // TODO: put accumulated distance in contiguous memory address
        int i = blockIdx.x;  // flight_plan_idx
        int j = threadIdx.x; // scenario idx
        int current_node_idx;
        int next_node_idx;
        int current_subnode;
        int subnodes_in_this_edge;
        int k_g;
        int edge_idx;
        float lat;
        float lon;
        float next_lat;
        float next_lon;
        float M;
        float M_edge;
        float M_target;
        int FL_edge;
        int FL;
        float d_step;
        float M_max_step;
        float FL_max_step;
        float M_guidance;

        
        float FL_target = {{FL0_cruise}};
        float FL_guidance = {{FL0_cruise}};
        float nm_since_last_FL_change = {{FL_change_threshold_nm/2}};
        {% if initial_phase == 'CLIMB' %}
            float nm_since_last_M_change = {{M_change_climb_offset}};
        {% else %}
            float nm_since_last_M_change = {{M_change_threshold_nm*2}};
        {% endif %}
        float FL_change_probability = 0;
        float FL_change_parameter = 0;
        float M_change_probability = 0;
        float M_change_parameter = 0;
        float acc_distance = 0;

        current_subnode = 0;
        subnodes_in_this_edge = 1;
        current_node_idx = {{origin_idx}};
        next_node_idx = {{origin_idx}};
        k_g = -1;
        lat = {{origin_lat}};
        lon = {{origin_lon}};
        M = {{M0_cruise}};
        M_guidance = {{M0_cruise}};
        M_target = {{M0_cruise}};
        FL = {{FL0_cruise}};
        
        for (int k=0; k < {{max_subnodes_per_path}}; k++)
        {
            if (current_node_idx != {{destination_idx}}) 
            {
                if (current_subnode == subnodes_in_this_edge - 1) {
                    k_g += 1;
                    current_node_idx = next_node_idx;
                    next_node_idx = guidance[i][k_g+1][j][0];
                    edge_idx = guidance[i][k_g][j][3];
                    subnodes_in_this_edge = edge_n_points[edge_idx],
                    current_subnode = 0;
                    M_edge = guidance[i][k_g][j][1] * 0.01f;
                    //tas_edge = 210;
                    d_step = guidance_f[i][k_g][j][1];
                    M_max_step = d_step/1500/25;
                    FL_edge = (float) guidance[i][k_g][j][2];
                    
                }
                profiles[i][k][LAT_IDX][j] = lat;
                profiles[i][k][LON_IDX][j] = lon;
                if (current_node_idx == {{destination_idx}}) {
                    continue;
                }
                next_lat = edge_subnode_coordinates[edge_idx][current_subnode + 1][0];
                next_lon = edge_subnode_coordinates[edge_idx][current_subnode + 1][1];
                profiles[i][k][TAS_IDX][j] = M;
                profiles[i][k][FL_IDX][j] = FL;
                profiles[i][k][COURSE_IDX][j] = track_approx_haversine(make_float2(lat, lon),
                                                                       make_float2(next_lat, next_lon), 
                                                                       0.3048*100*FL);
                profiles[i][k][D2NEXT_IDX][j] = d_step;
                profiles[i][k][6][j] = (float) current_subnode;
                profiles[i][k][7][j] = (float) subnodes_in_this_edge;
                acc_distance += d_step;
                profiles[i][k][31][j] = acc_distance;
                profiles[i][k][29][j] = guidance_f[i][k_g][j][2];
                
                FL_guidance += d_step/{{FL_change_regularization_distance_nm * 1852}}*((float) FL_edge - FL_guidance);
                FL_change_parameter = (nm_since_last_FL_change - {{FL_change_threshold_nm}})/{{FL_change_variability_nm}};
                FL_change_probability = 0.5 + 0.5 * rsqrtf(__fmaf_rz(FL_change_parameter,FL_change_parameter,1)) * FL_change_parameter;
                profiles[i][k][27][j] = FL_change_parameter;
                profiles[i][k][28][j] = FL_change_probability;
                if (FL_change_probability > execution_noise[i][j][{{n_crossroad_vars}} + edge_idx]) {
                    if (FL_guidance > FL_target + 10) {
                        FL_target += 20;
                        nm_since_last_FL_change = 0;
                    //} else if (FL_guidance < FL_target - 5) {
                    //    FL_target -= 20;
                    //    nm_since_last_FL_change = 0;
                    } else {
                        nm_since_last_FL_change += d_step/1852;
                    }
                } else {
                    nm_since_last_FL_change += d_step/1852;
                }
//
                
                FL_max_step = d_step*{{0.015*3.28084/100}};
                if (k == 0) {
                    M_guidance = M_edge;
                    {% if initial_phase == 'CLIMB' %}
                    M = M_edge;
                    FL_guidance = (float) FL_edge;
                    while (FL_target + 10 < FL_edge) {
                        FL_target += 20;
                    }
                    {% else %}
                    M = {{M0_cruise}};
                    {% endif %}
                } else {
                    M_guidance = M_guidance + d_step/{{M_guidance_regularization_nm * 1852.0}}*(M_edge - M_guidance);
                }
                profiles[i][k][24][j] = M_edge;
                M_change_parameter = (nm_since_last_M_change - {{M_change_threshold_nm}})/{{M_change_variability_nm}};
                M_change_probability = 0.5 + 0.5 * rsqrtf(__fmaf_rz(M_change_parameter,M_change_parameter,1)) * M_change_parameter;
                if (M_change_probability > execution_noise[i][j][{{n_crossroad_vars + n_edges}} + edge_idx]) {
                    if (fabsf(M_guidance - M_target) > .01f) {
                        M_target = .01f* lrintf(100 * M_guidance);
                        nm_since_last_M_change = 0;
                    //} else if (FL_guidance < FL_target - 5) {
                    //    FL_target -= 20;
                    //    nm_since_last_FL_change = 0;
                    } else {
                        nm_since_last_M_change += d_step/1852;
                    }
                } else {
                    nm_since_last_M_change += d_step/1852;
                }
//

                FL = clamp((float) FL_target, FL - FL_max_step, FL + FL_max_step);
                M = clamp((float) M_target, M - M_max_step, M + M_max_step);

                current_subnode ++;
                lat = next_lat;
                lon = next_lon;
            }  else {
                profiles[i][k][TAS_IDX][j] = 0.0f;
            }
        }

        for (int k=0; k < {{max_subnodes_per_path}}; k++)
        {
            // distance to go
            profiles[i][k][30][j] = acc_distance - profiles[i][k][31][j];
        }
   }

    // --------------------------------------------
   __global__ void create_cd_curves(float decision_variables[{{n_plans}}][{{n_flight_plan_vars}}],
                                    float climbdescent_curves[{{n_plans}}][2][2][{{climb_descent_profile_coeffs}}])
    {
        int fp_idx = blockIdx.x;  // flight_plan index
        int coeff_idx = threadIdx.x; // coefficient index
        int cd_idx = threadIdx.y; // climb or descent index
        int tas_or_slope = threadIdx.z; // h -> TAS or h -> TAS slope profile
        float val;
        float val2;
        if (tas_or_slope == 0) {
            val = decision_variables[fp_idx][get_climb_coeff(coeff_idx) + cd_idx*{{climb_descent_profile_coeffs}}];
        } else {
            val = decision_variables[fp_idx][get_climb_coeff(coeff_idx) + cd_idx*{{climb_descent_profile_coeffs}}];
            if (coeff_idx < {{climb_descent_profile_coeffs - 1}}) {
                val2 = decision_variables[fp_idx][get_climb_coeff(coeff_idx + 1) + cd_idx*{{climb_descent_profile_coeffs}}];
            } else {
                val2 = val;
            }
            val = (val2 - val)/{{45000*0.3048/climb_descent_profile_coeffs}};
        }
        climbdescent_curves[fp_idx][cd_idx][tas_or_slope][coeff_idx] = val;
    }

    // --------------------------------------------
   __global__ void __launch_bounds__({{n_scenarios}}) integrate_profile(float profiles[{{n_plans}}][{{max_subnodes_per_path}}][36][{{n_scenarios}}],
                                     float climbdescent_curves[{{n_plans}}][2][2][{{climb_descent_profile_coeffs}}],
                                     float ic_noise[{{n_plans}}][{{n_scenarios}}][2],
                                     float summaries[{{n_plans}}][{{n_scenarios}}][10],
                                     int write_profile)
   {
        // Integrate with Heun's method
        int iteration_count = 0;
        int i = blockIdx.x;  // flight_plan_idx
        int j = threadIdx.x; // scenario idx
        int phase = PHASE_{{initial_phase}};
        int wx_m = (i + gridDim.x * iteration_count) % {{n_members}};
        //int wx_m = i  % {{n_members}}; //((iteration_count * gridDim.x + i) * blockDim.x + j) % {{n_members}};
        float t = {{t0}} + {{t0_stdev}} * ic_noise[i][j][0];
        float m = {{m0}} + {{m0_stdev}} * ic_noise[i][j][1];
        float tas = {{tas0}};
        float lat = {{origin_lat}};
        float lon = {{origin_lon}};
        float course;
        float FL = {{FL0}};
        float d_step;
        float M;
        float d_step_previous;
        float h_slope;
        float gs;
        float gs_r;
        float t_next;
        float m_next;
        float gs_next;
        float tas_next;
        float lat_next;
        float lon_next;
        float FL_next;
        float gamma_next;
        float gamma;
        float acel;
        float acel_next;
        float acel_distance;
        float fc_;
        float fc_next;
        float distance_acc;
        float eta = 0;
        int h_idx = 0;
        float tas_low = 0;
        float tas_high = 0;
        float tas_target = 0;
        float h_frac = 0;
        float CT = 0;
        float thr = 0;
        float drag = 0;
        float drag_next = 0;
        float thr_next = 0;
        float CT_next = 0;
        float delta_t;
        float descent_at_d2go = 100;
        float d2go;
        //float energy_slope;
        float delta_v2 = 0;
        float eta_frac = 0;
        float delta_mass = 0;
        distance_acc = 0;
        gamma = 0.05;
        d_step_previous = 1;
        AircraftPosition apos = ac_pos_from_FL(make_float2(lat, lon), {{FL0}}, {{t0}});
//        Weather_3D_t weather = make_weather_3D_t(wx_m);
//        Weather_3D_t_lower weather_lower = make_weather_3D_t_lower(wx_m);
        //
        {% if compute_accf %}
            float accf_H2O = 0;
            float accf_CH4 = 0;
            float accf_O3 = 0;
            float accf_O3_s = 0;
            float accf_contrails = 0;
            float accf_contrails_c = 0;
            float issr = 0;
            float accf_dCont = 0;
            float accf_nCont = 0;
            float EI_NOx = 0;
            float4 wx_values_atm;
            float4 wx_values;
            float contrail_distance = 0;
        {% endif %}
        
        {% if compute_emissions %}
            float Nox_emissions = 0;
            float Nox_emissions_c = 0;
        {% endif %}
  
        AircraftPosition apos_next = apos;
        AircraftState acs(tas, 0.0, 0.0, m, 0.0);
        AircraftState acs_next = acs;
        gs = 0;
        delta_t = 0;
        descent_at_d2go = 0;
        EnvironmentVars env = env_from_apos(wx_m, apos);

        if (phase == PHASE_CLIMB) {
            CT = CT_max_MCRZ(env, acs, 0);
            thr = thrust(env, CT);
        }
        EnvironmentVars env_next = env;
        for (int k=0; k < {{max_subnodes_per_path}}-1; k++) {
            M = profiles[i][k][TAS_IDX][j];
            tas_next = M*env.a();
            lat_next = profiles[i][k+1][LAT_IDX][j];
            lon_next = profiles[i][k+1][LON_IDX][j];
            d_step   = profiles[i][k][D2NEXT_IDX][j];
            course   = profiles[i][k][COURSE_IDX][j];
            FL_next  = profiles[i][k][FL_IDX][j];
            d2go     = profiles[i][k][30][j];
            {% if perform_descent %}
                descent_at_d2go = profiles[i][k][29][j];
            {% endif %}
            if (tas_next > 1.0f) {
                // Phase transitions
                // we check climb -> leveloff after leveloff -> accel since we do not want to skip one leveloff step
                // we check accel -> cruise after leveloff -> accel since an accel step might be unnecessary

                // first-order approximation of the time in the incoming step for transition calculations
                delta_t = delta_t * d_step / d_step_previous;
                acs.tas = tas;
                acs.m = m;
                FL = fminf(FL, 445);
                if (phase ==  PHASE_LEVELOFF) {
                    phase = PHASE_ACCEL; // only do one step of leveloff
                }

                if (phase ==  PHASE_ACCEL) {
                    if ( m * (tas_next - tas) <= (thr - drag) * delta_t ) {
                        //energy_slope = -1;
                        phase = PHASE_CRUISE;
                    }
                }

                if (phase ==  PHASE_CLIMB) {
                    h_idx = floor(FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}});
                    h_frac = (FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}}) - h_idx;
//                    h_idx = 0; // debug
//                    h_frac = .5f; // debug
                    h_idx = max(0, h_idx);
                    eta = 0.5*(climbdescent_curves[i][0][1][h_idx] + climbdescent_curves[i][0][1][h_idx+1]);
                    tas_low = climbdescent_curves[i][0][0][h_idx];
                    tas_high = climbdescent_curves[i][0][0][h_idx+1];
                    tas_target = tas_low*(1 - h_frac) + tas_high*h_frac;
                    eta += 0.5*(tas_target - tas)/(d_step*sinf(gamma));
                    eta = fmaxf(0.0f, eta);
                    eta_frac = FL/FL_next;
                    eta = (1 - eta_frac)*eta + eta_frac * (tas_next - tas)/( (FL_next - FL)*30.48 );
                    acs.gamma = gamma;
                    // one fixed point iteration is precise enough
                    drag = D_clean(env, acs, climbdescent_CL(env, acs, gamma));
                    {% if avoid_negative_thrust %}
                    thr = fmaxf(thr, 0);
                    {% endif %}
                    gamma = asinf(fminf({{gamma_max}}, fmaxf((thr - drag)/m/(tas * eta + g), 0.01)));
                    gamma_next = gamma;
                    //energy_slope = -1.0f;
                    if (FL > FL_next) {
                        phase = PHASE_ACCEL;
                    } else {
                        // Leveloff can start when
                        // note that those values will be overwritten if phase != LEVELOFF
                        delta_v2 = 0.5*(tas_next*tas_next - tas*tas);
                        thr = drag + m/delta_t/tas*(g * (FL_next - FL)*30.48 + delta_v2);
                        CT = CT_from_thrust(env, thr);
                        // should correct for D/T variation with altitude
                        if (CT <= CT_max_MCRZ(env, acs, 0)) {
                            phase = PHASE_LEVELOFF;
                        //} else if (FL + (tas * sinf(gamma) * delta_t)/30.48 >= FL_next) {
                            //energy_slope = delta_v2/g/(FL_next - FL);
                        }
                    }
                }

                if (phase == PHASE_CRUISE) {
                    if (d2go < descent_at_d2go*1852 + {{FL_change_threshold_nm * 1852}}) {
                        FL_next = FL;
                    } if (d2go < descent_at_d2go*1852) {
                        phase = PHASE_DECEL;
                        h_idx = floor(FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}});
                        h_frac = (FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}}) - h_idx;
                        h_idx = max(0, h_idx);
                        tas_low = climbdescent_curves[i][1][0][h_idx];
                        tas_high = climbdescent_curves[i][1][0][h_idx+1];
                        tas_target = tas_low*(1 - h_frac) + tas_high*h_frac;
                    }
                }

                if (phase == PHASE_LEVELOFF_2) {
                    phase = PHASE_HOLD;
                }

                if (phase == PHASE_HOLD) {
                    tas_next = tas;
                    FL_next = {{FL_STAR}};
                }



                gs = -1.0f; // so that, if not computed (due to phase), it can be computed afterwards

                // Remove CT / thrust calcs and move to transitions;
                if (phase == PHASE_CLIMB || phase == PHASE_ACCEL) {
                    CT = CT_max_MCRZ(env, acs, 0);
                } else if (phase == PHASE_DECEL || phase == PHASE_DESCENT) {
                    //old_CT = CT;
                    CT = CT_min(env, acs);
                }

                //if (phase == PHASE_LEVELOFF) {

                //}

                if (phase != PHASE_CRUISE && phase != PHASE_LEVELOFF) {
                    thr = thrust(env, CT);
                }

                if (phase == PHASE_DECEL) {
                    //if ( tas + (thr - drag) * delta_t / m <= tas_target) {
                        phase = PHASE_DESCENT;
                        gamma = - 0.05f;
                    //}
                }


                if (phase == PHASE_DESCENT) {
                    h_idx = floor(FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}});
                    h_frac = (FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}}) - h_idx;
//                    h_idx = 0; // debug
//                    h_frac = .5f; // debug
                    h_idx = max(0, h_idx);
                    eta = 0.5* (climbdescent_curves[i][1][1][h_idx] + climbdescent_curves[i][1][1][max(h_idx - 1, 0)]);

                    tas_low = climbdescent_curves[i][1][0][h_idx];
                    tas_high = climbdescent_curves[i][1][0][h_idx+1];
                    tas_target = tas_low*(1 - h_frac) + tas_high*h_frac;
                    eta += 0.5*(tas_target - tas)/(d_step*sinf(gamma));
                    eta = fmaxf(0.0f, eta);
                    acs.gamma = gamma;
                    // one fixed point iteration is precise enough
                    drag = D_clean(env, acs, climbdescent_CL(env, acs, gamma));
                    gamma = asinf(fminf({{gamma_max}}, fmaxf((thr - drag)/m/(tas * eta + g), {{gamma_min}})));
                    acs.gamma = gamma;
                    gamma_next = gamma;
                    profiles[i][k][14][j] = FL + gs * sin(gamma) * delta_t / 30.48;
                    if (FL + gs * sin(gamma) * delta_t / 30.48 <=  {{FL_STAR}}) {
                        h_idx = floor(FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}});
                        h_frac = (FL / {{top_cd_profile*0.01/climb_descent_profile_coeffs}}) - h_idx;
                        h_idx = max(0, h_idx);
                        tas_low = climbdescent_curves[i][1][0][h_idx];
                        tas_high = climbdescent_curves[i][1][0][h_idx+1];
                        tas_next = tas_low*(1 - h_frac) + tas_high*h_frac;
                        phase = PHASE_LEVELOFF_2;
                        FL_next = {{FL_STAR}};
                    }
                }


                if (phase == PHASE_CRUISE || phase == PHASE_LEVELOFF || phase == PHASE_LEVELOFF_2) {
                    h_slope = (FL_next - FL)*30.48/d_step;
                    gs = groundspeed_slope(env, tas, course, h_slope);
                    gamma = asinf(gs*h_slope/tas);
                } else if (phase == PHASE_ACCEL || phase == PHASE_HOLD || phase == PHASE_DECEL) {
                    gamma = 0;
                    gamma_next = 0;
                    h_slope = 0;
                    drag = D_clean(env, acs, climbdescent_CL(env, acs, 0));
                }
                if (gs < 0.0f) {
                    gs = groundspeed(env, tas, course, gamma);
                }
                acs.gamma = gamma;
                gs_r = 1.0/gs;
                delta_t = d_step*gs_r;
                t_next = t + delta_t; // intermediate Heun estimate
                if (phase != PHASE_CRUISE) {
                    FL_next = FL + delta_t * tas * sinf(gamma) / 30.48;
                }
                apos_next = ac_pos_from_FL(make_float2(lat_next, lon_next), FL_next, t_next);
                env_next = env_from_apos(wx_m, apos_next);
//                if (phase == PHASE_CRUISE || phase == PHASE_LEVELOFF || phase == PHASE_ACCEL || phase == PHASE_DECEL) {
//                    env_next = weather.get_env(apos_next);
//                } else {
//                    env_next = weather_lower.get_env(apos_next);
//                }

                if (phase == PHASE_CRUISE || phase == PHASE_HOLD || phase == PHASE_LEVELOFF_2) {
                    gs_next = groundspeed_slope(env_next, tas_next, course, h_slope);
                    gamma_next = asinf(gs_next/tas_next*h_slope);
                    t_next = t + 0.5*d_step*(gs_r + 1.0/gs_next);
                    acel_distance = (tas_next - tas)/d_step;
                    acel = acel_distance*gs;
                    acel_next = acel_distance*gs_next;
                    fc_ = fc_acel_gamma(env, acs, acel, gamma);
                } else {
                    acel = (thr - drag)/m - g*sinf(gamma);
                    gamma_next = gamma;
                    tas_next = tas + acel * delta_t;
                    gs_next = groundspeed(env_next, tas_next, course, gamma);
                    t_next = t + 0.5*d_step*(gs_r + 1.0/gs_next); // acel_next
                    fc_ = fc_CT(env, acs, CT);
                    m_next = m - fc_*(t_next - t);
                    AircraftState acs_next_(tas_next, 0.0, gamma_next, m_next, 0.0);
                    acs_next = acs_next_;
                    if (phase == PHASE_CLIMB || phase == PHASE_ACCEL ) {
                        CT_next = CT_max_MCRZ(env_next, acs_next, 0);
                    } else if (phase == PHASE_DECEL || phase == PHASE_DESCENT) {
                        //old_CT = CT;
                        CT_next = CT_min(env_next, acs_next);
                    } else if (phase == PHASE_LEVELOFF ) {
                        CT_next = CT;
                    }
                    thr_next = thrust(env_next, CT_next);
                    drag_next = D_clean(env_next, acs_next, climbdescent_CL(env_next, acs_next, gamma_next));
;
                    acel_next = (thr_next - drag_next)/m_next - g*sinf(gamma_next);
                    tas_next = tas + 0.5*(acel + acel_next) * (t_next - t);

                    FL_next = FL + (t_next - t) * sinf(gamma) / 30.48 * 0.5 * (tas + tas_next);
                }
                m_next = m - fc_*(t_next - t);// intermediate Heun estimate
                AircraftState acs_next_(tas_next, 0.0, gamma_next, m_next, 0.0);
                acs_next = acs_next_;




                if (phase == PHASE_CRUISE) {
                    fc_next = fc_acel_gamma(env_next, acs_next, acel_next, gamma_next);
                } else {
                    fc_next = fc_CT(env_next, acs_next, CT);
                }

                // write directly to m instead of m_next
                if (write_profile > 0) {
                    profiles[i][k][6][j] = m;
                    profiles[i][k][27][j] = accf_contrails;
                    profiles[i][k][28][j] = accf_CH4;
                    profiles[i][k][29][j] = accf_H2O;
                    profiles[i][k][30][j] = accf_O3;
                    profiles[i][k][31][j]= accf_contrails_c;
                    profiles[i][k][32][j]= Nox_emissions;
                    profiles[i][k][33][j]= Nox_emissions_c;
                    profiles[i][k][34][j]= contrail_distance;
                    profiles[i][k][7][j] = env_next.rho;
                    profiles[i][k][8][j] = FL;
                    profiles[i][k][9][j] = gamma;
                    profiles[i][k][10][j] = tas;
                    profiles[i][k][11][j] = tas_next;
                    profiles[i][k][12][j] = gs;
                    profiles[i][k][13][j] = gs_next;
                    profiles[i][k][14][j] = M;
                    profiles[i][k][15][j] = thr;
                    profiles[i][k][16][j] = drag;
                    profiles[i][k][17][j] = t;
                    profiles[i][k][18][j] = delta_t;
                    profiles[i][k][19][j] = t_next;
                    profiles[i][k][20][j] = fc_;
                    profiles[i][k][21][j] = fc_next;
                    profiles[i][k][22][j] = acel;
                    profiles[i][k][23][j] = (float) phase;

                    //profiles[i][k][24][j] = (float) Mg;
                    profiles[i][k][25][j] =  CT;
                    profiles[i][k][26][j] =  eta;
                }
                d_step_previous = d_step;
                delta_mass = 0.5*(t_next - t)*(fc_ + fc_next);
                m -= delta_mass;



                // readying next iteration
                tas = tas_next;
                t = t_next;
                lat = lat_next;
                lon = lon_next;
                apos = ac_pos_from_FL(make_float2(lat, lon), FL, t);
                env = env_next;
                acs = acs_next;
                FL = FL_next;
                distance_acc += d_step;
                CT = CT_next;
                thr = thr_next;

                {% if compute_accf %}
                    if (phase != PHASE_CRUISE) {
                        M = tas/env_next.a();
                    }
                    wx_values_atm = env_var(wx_m, apos); // 't', 'u', 'v', 'z'
                    wx_values = accf_1_from_apos(wx_m, apos); // 'r', 'pv', 'C1', 'C2'
                    EI_NOx = compute_EI_NOx(fc_next, M, wx_values.z, wx_values.w);
                    if (wx_values_atm.x <= 235 && wx_values.x >= 0.95) {
                        issr = 1.0;
                    } else {
                        issr = 0.0;
                    }
                    accf_O3_s = 0.001f * compute_accf_o3 (wx_values_atm.x, wx_values_atm.w)  * EI_NOx  * delta_mass;

                    if (accf_O3_s > 0) {
                        accf_O3 += accf_O3_s;
                    } else {
                        accf_O3 += 0.0;
                    }

                    accf_H2O += compute_accf_h2o (wx_values.y)  * delta_mass;
                    accf_nCont = compute_accf_nCont (wx_values_atm.x); 
                    
                    wx_values = accf_2_from_apos(wx_m, apos); // 'olr', 'aCCF_CH4'
                    accf_dCont = compute_accf_dCont (wx_values.x);

                    if (0.001f * wx_values.y * EI_NOx * delta_mass > 0) {
                        accf_CH4 += 0;
                    } else {
                        accf_CH4 += 0.001f * wx_values.y * EI_NOx * delta_mass;
                    }

                    accf_contrails += 0.001f * d_step * issr * ({{nCon_index}} * accf_nCont + {{dCon_index}} * accf_dCont);
                    accf_contrails_c =  0.001f * d_step * issr * ({{nCon_index}} * accf_nCont + {{dCon_index}} * accf_dCont);
                    contrail_distance += 0.001f * d_step * issr;

                {% endif %}    
                    
                {% if compute_emissions %}  
                    if (phase != PHASE_CRUISE) {
                        M = tas/env_next.a();
                    }
                    Nox_emissions  += EI_NOx * delta_mass;
                    Nox_emissions_c =  EI_NOx;
                {% endif %}
            }
        }
        summaries[i][j][0] =  {{m0}} - m;
        summaries[i][j][1] = t - {{t0}};
        summaries[i][j][2] = fmaxf(FL - {{FL_STAR}}, 0);
        {% if compute_accf %}
        summaries[i][j][3] = accf_CH4;
        summaries[i][j][4] = accf_H2O;
        summaries[i][j][5] = accf_O3;
        summaries[i][j][6] = accf_contrails;
        summaries[i][j][7] = 10.1 * 6.94e-16f * ({{m0}} - m); //ATR_CO2;
        summaries[i][j][8] = contrail_distance;
        {% endif %}
        {% if compute_accf %}
        summaries[i][j][9] = Nox_emissions; 
        {% endif %}
   }
}


