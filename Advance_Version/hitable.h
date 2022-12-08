#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "aabb.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material* mat_ptr;
};

class hitable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const = 0;
};

#endif
