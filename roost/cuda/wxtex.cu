
{% for i in range(n_cubes) %}
    {% for j in range(n_members) %}
        texture<float4, cudaTextureType3D> wxcube_{{i}}_{{j}};
    {% endfor %}
{% endfor %}

{% for i in range(n_cubes) %}
    __device__ float4 get_wxcube_{{i}}(int member_idx, float lat_deg, float lon_deg, float t, float P)
    {
    float lat_c = (lat_deg - {{latitude_min}})/{{latitude_range}} + 0.5/{{n_latitude}}; // + .5 again?
    float lon_c = (lon_deg - {{longitude_min}})/{{longitude_range}} + 0.5/{{n_longitude}};
    float t_c = t/{{time_range}};
    float4 out;
    float4 out2;
    int P_idx = 0;
    float P_frac = 0.0f;
    P *= 0.01; // to hPa
    {% for (P_low, P_high, P_diff) in P_triplets %}
        {% if loop.index0 != 0%}else{% endif %} if (P < {{P_high}}) {
            P_idx = {{loop.index0}};
            P_frac = (P - {{P_low}})/{{P_diff}};
        }
    {% endfor %}
    t_c = 0.5/{{n_levels * n_time}} + {{1.0/n_levels}}*P_idx + {{(n_time - 1)/(n_levels * n_time)}}*t_c;
    if (member_idx == 0) {
        out = tex3D(wxcube_{{i}}_0, lat_c, lon_c, t_c);
        out2 = tex3D(wxcube_{{i}}_0, lat_c, lon_c, t_c + {{1.0/n_levels}});
    {% for k in range(1, n_members) %}
        } else if (member_idx == {{k}}) {
            out = tex3D(wxcube_{{i}}_{{k}}, lat_c, lon_c, t_c);
            out2 = tex3D(wxcube_{{i}}_{{k}}, lat_c, lon_c, t_c + {{1.0/n_levels}});
    {% endfor %}
    }
    out *= 1 - P_frac;
    out += P_frac * out2;
    out2 = make_float4({% for o in wx_offsets[i] %}{{o}}{% if loop.index0 != 3%}, {% endif %}{% endfor %});
    return out - out2; // - offset
    }
{% endfor %}