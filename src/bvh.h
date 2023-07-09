#ifndef BVH_H
#define BVH_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hitable.h"
#include "hitable_list.h"

#include <algorithm>


class bvh_node : public hitable {
public:
    __device__ bvh_node();

    __device__ bvh_node(const hitable_list& list, double time0, double time1)
        : bvh_node(list.list, 0, list.list_size, time0, time1)
    {}

    __device__ bvh_node(hitable** src_objects, size_t start, size_t end, double time0, double time1);

    __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(double time0, double time1, aabb& output_box) const;

public:
    std::shared_ptr<hitable> left;
    std::shared_ptr<hitable> right;
    aabb box;
};

__device__ inline float random_float() {
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0);
}

__device__ inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

__device__ inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(min, max + 1));
}

__device__ inline bool box_compare(const hitable* a, const hitable* b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b))
        std::cerr << "No bounding box in bvh_node constructor.\n";

    return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ bool box_x_compare(const hitable* a, const hitable* b) {
    return box_compare(a, b, 0);
}

__device__ bool box_y_compare(const hitable* a, const hitable* b) {
    return box_compare(a, b, 1);
}

__device__ bool box_z_compare(const hitable* a, const hitable* b) {
    return box_compare(a, b, 2);
}

__device__ bvh_node::bvh_node(
    hitable** src_objects,
    size_t start, size_t end, double time0, double time1
) {
    auto objects = src_objects; // Create a modifiable array of the source scene objects

    int axis = random_int(0, 2);
    auto comparator = (axis == 0) ? box_x_compare
        : (axis == 1) ? box_y_compare
        : box_z_compare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    }
    else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        }
        else {
            left = objects[start + 1];
            right = objects[start];
        }
    }
    else {
        std::sort(objects[0] + start, objects[0] + end, comparator);

        auto mid = start + object_span / 2;
        left = std::make_shared<bvh_node>(objects, start, mid, time0, time1);
        right = std::make_shared<bvh_node>(objects, mid, end, time0, time1);
    }

    aabb box_left, box_right;

    if (!left->bounding_box(time0, time1, box_left)
        || !right->bounding_box(time0, time1, box_right)
        )
        std::cerr << "No bounding box in bvh_node constructor.\n";

    box = surrounding_box(box_left, box_right);
}

__device__ bool bvh_node::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max))
        return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
}

__device__ bool bvh_node::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = box;
    return true;
}
#endif
