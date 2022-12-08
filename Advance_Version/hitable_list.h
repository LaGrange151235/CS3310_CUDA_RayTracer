#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"
#include "aabb.h"

class hitable_list : public hitable {
public:
    __device__ hitable_list() {}
    __device__ hitable_list(hitable** l, int n) { list = l; list_size = n; }
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;
    hitable** list;
    int list_size;
};

__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool hitable_list::bounding_box(double time0, double time1, aabb& output_box) const {
    if (list_size <= 0) return false;

    aabb temp_box;
    bool first_box = true;

    for (int i = 0; i < list_size; i++) {
        if (!list[i]->bounding_box(time0, time1, temp_box)) return false;
        first_box = false;

    }

    return true;
}

#endif
