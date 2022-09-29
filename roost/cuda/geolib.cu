#define N_trjs {{N_trjs}}
#define N_nodes {{N_nodes}}
#define N_ec {{N_ec}}

#define D2R 0.017453292519943295
#define R_N 6378137.0
#define R_M 6356752.3142

//__device__ float3 latlon_to_xyz(float2 latlon)
//{
//    return make_float3(sin(latlon.y)*cos(latlon.x), -cos(latlon.x)*cos(latlon.y), sin(latlon.x));
//}
//
//__device__ float2 xyz_to_latlon(float3 xyz)
//{
//    return make_float2(asin(xyz.z), atan2(xyz.x, -xyz.y));
//}
//
//__device__ float3 orthogonal_rodrigues_rotation(float3 v, float3 k, float angle)
//{
//	return v*cos(angle) + cross(k, v)*sin(angle);
//}
//
//__device__ struct GeoLocation {
// float lat;
// float lon;
// float3 p;
// __device__ GeoLocation( float a, float b, float3 xyz) : lat(a), lon(b), p(xyz) {}
////  __device__ float2 get_latlon( void ) { return make_float2(lat*pi/180, lon*pi/180); }
////  __device__ float3 get_xyz( void ) { return latlon_to_xyz(make_float2(lat*pi/180, lon*pi/180)); }
//};
//
//
//__device__ struct GeoData {
// GeoLocation origin;
// GeoLocation destination;
// float3 ref_p;
// float3 rot_axis_r;
// float3 rot_axis_q;
// float theta;
// float half_theta;
//
// __device__ GeoData(GeoLocation a, GeoLocation b, float3 mp, float3 ax_r, float3 ax_q, float th) :
//      origin(a), destination(b), ref_p(mp), rot_axis_r(ax_r), rot_axis_q(ax_q), theta(th), half_theta(th/2) {}
// __device__ float3 rq_to_p(float r, float q)
// {
//	float3 pr = orthogonal_rodrigues_rotation(ref_p, rot_axis_r, half_theta*r);
//    return orthogonal_rodrigues_rotation(pr, rot_axis_q, half_theta*0.5*q);
// }
//__device__ float2 rq_to_latlon(float r, float q)
// {
//    float3 p = rq_to_p(r, q);
//    return xyz_to_latlon(p);
// }
//__device__ void to_array(float* gd_arr)
// {
//    gd_arr[0] = origin.lat;
//    gd_arr[1] = origin.lon;
//    gd_arr[2] = destination.lat;
//    gd_arr[3] = destination.lon;
//    gd_arr[4] = ref_p.x;
//    gd_arr[5] = ref_p.y;
//    gd_arr[6] = ref_p.z;
//    gd_arr[7] = rot_axis_r.x;
//    gd_arr[8] = rot_axis_r.y;
//    gd_arr[9] = rot_axis_r.z;
//    gd_arr[10] = rot_axis_q.x;
//    gd_arr[11] = rot_axis_q.y;
//    gd_arr[12] = rot_axis_q.z;
//    gd_arr[13] = theta;
// }
//};
//
//
//__device__ float3 p1p2_to_midpoint(float3 p1, float3 p2)
//{
//    float3 mp = 0.5*(p1 + p2);
//    return normalize(mp);
//}
//
//__device__ GeoLocation make_geoloc(float2 loc)
//{
//    return GeoLocation(loc.x, loc.y, latlon_to_xyz(loc));
//}
//
//__device__ GeoData make_geodata(float2 origin, float2 destination)
//{
//    GeoLocation geo_origin = make_geoloc(origin);
//    GeoLocation geo_destination = make_geoloc(destination);
//    float3 mp = p1p2_to_midpoint(geo_origin.p, geo_destination.p);
//    float3 ax_r = cross(geo_origin.p, geo_destination.p - geo_origin.p);
//    ax_r = normalize(ax_r);
//    float3 ax_q = normalize(geo_destination.p - geo_origin.p);
//    float theta = 2*asin(length(geo_destination.p - geo_origin.p)/2);
//    GeoData gd(geo_origin, geo_destination, mp, ax_r, ax_q, theta);
//    return gd;
//}
//
//
//__device__ GeoData make_geodata_from_array(float* gd_arr)
//{
//    GeoLocation geo_origin = make_geoloc(make_float2(gd_arr[0], gd_arr[1]));
//    GeoLocation geo_destination = make_geoloc(make_float2(gd_arr[2], gd_arr[3]));
//    float3 mp = {gd_arr[4], gd_arr[5], gd_arr[6]};
//    float3 ax_r = {gd_arr[7], gd_arr[8], gd_arr[9]};
//    float3 ax_q = {gd_arr[10], gd_arr[11], gd_arr[12]};
//    GeoData gd(geo_origin, geo_destination, mp, ax_r, ax_q, gd_arr[13]);
//    return gd;
//}

__device__ float track_approx_haversine(float2 ll1, float2 ll2, float z)
{
    float lat1 = (ll1.x)*D2R;
    float lat2 = ll2.x*D2R;
    float dlat = lat2 - lat1;
    float dlon = (ll2.y - ll1.y)*D2R;
    float dist_lon = dlon*(R_N + z)/sqrtf(cos(lat1)*cos(lat2));
    float dist_lat = dlat*(R_M + z);
    float course = atan2(dist_lon, dist_lat);
    return course;
}

__device__ float distance_approx_haversine(float2 ll1, float2 ll2, float z)
{
    float2 diff = (ll2 - ll1)*D2R;
    float dlat = diff.x;
    float dlon = diff.y;
    float lat1 = ll1.x;
    float lat2 = ll2.x;
    float d_sq_lat = powf(dlat*(R_M+z), 2);
    float d_sq_lon = powf(dlon*(R_N+z), 2);
    return sqrtf(d_sq_lat + d_sq_lon*cos(lat1)*cos(lat2));
}
//
//extern "C" { // because we compile with no_extern_c=True, need this to
//             // prevent name mangling when recovering the function
//    __global__ void orig_dest_to_gdarr(float* origdest, float* gdarr)
//    {
//        float2 orig = {origdest[0], origdest[1]};
//        float2 dest = {origdest[2], origdest[3]};
//        GeoData gd = make_geodata(orig, dest);
//        gd.to_array(gdarr);
//    }
//
//     __global__ void test_gdarr_to_gdarr(float* gdarr1, float* gdarr2)
//    {
//        GeoData gd = make_geodata_from_array(gdarr1);
//        gd.to_array(gdarr2);
//    }
//
//    __global__ void ec_to_latlon(float* gdarr, float ec[N_trjs][N_ec], float ll_array[N_trjs][N_nodes][2])
//    {
//        GeoData gd = make_geodata_from_array(gdarr);
//        int n_nodes_per_block = N_nodes/blockDim.x;
//        int i = blockIdx.x;
//        int j = threadIdx.x + n_nodes_per_block*blockIdx.y;
//        float r = -1.0 + (((float) j)/(N_nodes-1))*2;
//        float q = 0;
//        for (int k=0; k<N_ec; k++) {
//            q += ec[i][k]*cos(0.5*pi*(k + r*(1+k))); // powf(0.5, k/2)*
//        }
//        float2 ll = gd.rq_to_latlon(r, q);
//        ll_array[i][j][0] = ll.x;
//        ll_array[i][j][1] = ll.y;
//    }
//
//    __global__ void compute_dist_course(float ll_array[N_trjs][N_nodes][2], float distances[N_trjs][N_nodes-1], float courses[N_trjs][N_nodes-1], float z)
//    {
//        //int n_nodes_per_block = N_nodes/blockDim.x;
//        int i = blockIdx.x;
//        int j = threadIdx.x; // + n_nodes_per_block*blockIdx.y;
//        float2 ll_prev = make_float2(ll_array[i][j][0], ll_array[i][j][1]);
//        float2 ll_next = make_float2(ll_array[i][j+1][0], ll_array[i][j+1][1]);
//        distances[i][j] = distance_approx_haversine(ll_prev, ll_next, z);
//        courses[i][j] = track_approx_haversine(ll_prev, ll_next, z);
//    }
//
//}
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
