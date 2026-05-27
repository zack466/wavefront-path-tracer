#pragma once
// BVH flat node definition. Kept in a standalone header (no STL) so that
// CUDA device code can include it directly.
//
// 40 bytes per node.
//   Interior: left_first = left child index, right_child = right child index,
//             prim_count = 0, split_axis = dominant split axis.
//   Leaf:     left_first = first index into prim_indices[], prim_count > 0.
struct BVHNode {
    float aabb_min[3];
    float aabb_max[3];
    int   left_first;    // interior: left child idx; leaf: first prim offset
    int   prim_count;    // 0 = interior; >0 = leaf
    int   right_child;   // interior only
    int   split_axis;    // 0/1/2 — used to order child traversal by ray sign
};
