#include "bvh.h"
#include "scene/scene.h"

#include <algorithm>
#include <cstring>

static AABB triangle_aabb_fn(const Triangle& tri) {
    AABB a = aabb_point(tri.v[0]);
    a = aabb_expand(a, tri.v[1]);
    a = aabb_expand(a, tri.v[2]);
    Vec3 pad(1e-4f);
    a.min_pt = a.min_pt - pad;
    a.max_pt = a.max_pt + pad;
    return a;
}

// ── SAH constants ──────────────────────────────────────────────────────────────
static constexpr int   MAX_LEAF_PRIMS = 4;
static constexpr int   N_BINS         = 8;
static constexpr float C_TRAV         = 1.0f;   // cost of a BVH traversal step
static constexpr float C_ISECT        = 1.0f;   // cost of a primitive intersection

// ── Primitive info (used only during build) ───────────────────────────────────
struct PrimInfo {
    AABB bounds;
    Vec3 centroid;
    int  original_index;  // index in unified prim space (< num_shapes → analytic, else triangle)
};

// ── Intermediate tree node (only lives during build) ──────────────────────────
struct BuildNode {
    AABB      bounds;
    BuildNode* children[2];
    int        first_prim_offset;
    int        n_prims;
    int        split_axis;

    void make_leaf(int first, int n, AABB b) {
        children[0] = children[1] = nullptr;
        first_prim_offset = first;
        n_prims = n;
        bounds  = b;
        split_axis = 0;
    }
    void make_interior(int axis, BuildNode* l, BuildNode* r) {
        children[0] = l; children[1] = r;
        bounds = aabb_union(l->bounds, r->bounds);
        n_prims = 0;
        split_axis = axis;
    }
};

// ── SAH binned build ──────────────────────────────────────────────────────────

static AABB centroid_aabb(const std::vector<PrimInfo>& prims, int start, int end) {
    AABB ca = aabb_empty();
    for (int i = start; i < end; ++i)
        ca = aabb_expand(ca, prims[i].centroid);
    return ca;
}

static AABB prims_aabb(const std::vector<PrimInfo>& prims, int start, int end) {
    AABB ba = aabb_empty();
    for (int i = start; i < end; ++i)
        ba = aabb_union(ba, prims[i].bounds);
    return ba;
}

// Returns a pool-allocated BuildNode.
static BuildNode* build_recursive(std::vector<PrimInfo>& prims, int start, int end,
                                   std::vector<int>& ordered_indices,
                                   std::vector<BuildNode>& pool)
{
    pool.emplace_back();
    BuildNode* node = &pool.back();
    int n = end - start;

    AABB node_bounds = prims_aabb(prims, start, end);

    // Force leaf for small primitive counts
    if (n <= MAX_LEAF_PRIMS) {
        int first = (int)ordered_indices.size();
        for (int i = start; i < end; ++i)
            ordered_indices.push_back(prims[i].original_index);
        node->make_leaf(first, n, node_bounds);
        return node;
    }

    // Compute centroid AABB to determine best split axis
    AABB caabb = centroid_aabb(prims, start, end);
    Vec3 cext  = caabb.max_pt - caabb.min_pt;
    int axis   = (cext.x > cext.y) ? (cext.x > cext.z ? 0 : 2)
                                    : (cext.y > cext.z ? 1 : 2);

    // If all centroids are coincident, make a leaf
    if (caabb.max_pt[axis] == caabb.min_pt[axis]) {
        int first = (int)ordered_indices.size();
        for (int i = start; i < end; ++i)
            ordered_indices.push_back(prims[i].original_index);
        node->make_leaf(first, n, node_bounds);
        return node;
    }

    // Binned SAH: divide centroid AABB into N_BINS bins along the chosen axis
    struct Bin { AABB bounds; int count; };
    Bin bins[N_BINS] = {};
    for (int i = 0; i < N_BINS; ++i) bins[i].bounds = aabb_empty();

    float inv_width = N_BINS / (caabb.max_pt[axis] - caabb.min_pt[axis]);
    for (int i = start; i < end; ++i) {
        int b = (int)((prims[i].centroid[axis] - caabb.min_pt[axis]) * inv_width);
        b = std::min(b, N_BINS - 1);
        bins[b].count++;
        bins[b].bounds = aabb_union(bins[b].bounds, prims[i].bounds);
    }

    // Prefix/suffix scans over bins → evaluate all N_BINS-1 split candidates in O(N_BINS)
    float parent_sa = node_bounds.surface_area();
    if (parent_sa < 1e-12f) parent_sa = 1e-12f;

    AABB left_b[N_BINS], right_b[N_BINS];
    int  lc[N_BINS],     rc[N_BINS];

    left_b[0] = bins[0].bounds; lc[0] = bins[0].count;
    for (int i = 1; i < N_BINS; ++i) {
        left_b[i] = aabb_union(left_b[i-1], bins[i].bounds);
        lc[i]     = lc[i-1] + bins[i].count;
    }
    right_b[N_BINS-1] = bins[N_BINS-1].bounds; rc[N_BINS-1] = bins[N_BINS-1].count;
    for (int i = N_BINS-2; i >= 0; --i) {
        right_b[i] = aabb_union(right_b[i+1], bins[i].bounds);
        rc[i]      = rc[i+1] + bins[i].count;
    }

    float best_cost  = 1e30f;
    int   best_split = -1;
    for (int s = 1; s < N_BINS; ++s) {
        float cost = C_TRAV + (left_b[s-1].surface_area() * lc[s-1]
                             + right_b[s].surface_area()  * rc[s]) * C_ISECT / parent_sa;
        if (cost < best_cost) { best_cost = cost; best_split = s; }
    }

    // Compare with making a leaf
    float leaf_cost = (float)n * C_ISECT;
    if (best_split < 0 || (best_cost >= leaf_cost && n <= MAX_LEAF_PRIMS)) {
        int first = (int)ordered_indices.size();
        for (int i = start; i < end; ++i)
            ordered_indices.push_back(prims[i].original_index);
        node->make_leaf(first, n, node_bounds);
        return node;
    }

    // Partition prims at the best split bin boundary
    auto mid_it = std::partition(prims.begin() + start, prims.begin() + end,
        [&](const PrimInfo& pi) {
            int b = (int)((pi.centroid[axis] - caabb.min_pt[axis]) * inv_width);
            b = std::min(b, N_BINS - 1);
            return b < best_split;
        });
    int mid = (int)(mid_it - prims.begin());

    // Fallback: if partition produced an empty half, split evenly
    if (mid == start || mid == end) mid = (start + end) / 2;

    BuildNode* left  = build_recursive(prims, start, mid, ordered_indices, pool);
    BuildNode* right = build_recursive(prims, mid,   end, ordered_indices, pool);
    node->make_interior(axis, left, right);
    return node;
}

// Flatten the build tree into the flat node array (BFS/DFS order).
// Returns the index of the flattened node.
static int flatten(const BuildNode* build_node, std::vector<BVHNode>& nodes)
{
    int idx = (int)nodes.size();
    nodes.emplace_back();
    BVHNode& n = nodes.back();

    std::memcpy(n.aabb_min, &build_node->bounds.min_pt, 12);
    std::memcpy(n.aabb_max, &build_node->bounds.max_pt, 12);
    n.split_axis = build_node->split_axis;

    if (build_node->n_prims > 0) {
        // Leaf
        n.left_first  = build_node->first_prim_offset;
        n.prim_count  = build_node->n_prims;
        n.right_child = -1;
    } else {
        // Interior — flatten children recursively
        n.prim_count = 0;
        int left_idx = flatten(build_node->children[0], nodes);
        // Note: 'n' reference may be invalidated by the recursive push_back above!
        // Re-fetch by index.
        int right_idx = flatten(build_node->children[1], nodes);
        nodes[idx].left_first  = left_idx;
        nodes[idx].right_child = right_idx;
    }
    return idx;
}

// ── Public: build ─────────────────────────────────────────────────────────────

void build_bvh(SceneData& scene)
{
    int ns = (int)scene.shapes.size();
    int nt = (int)scene.triangles.size();
    int total = ns + nt;

    scene.accel.num_shapes = ns;
    scene.accel.nodes.clear();
    scene.accel.prim_indices.clear();

    if (total == 0) return;

    // Build PrimInfo array
    std::vector<PrimInfo> prims(total);
    for (int i = 0; i < ns; ++i) {
        AABB b = shape_aabb(scene.shapes[i]);
        prims[i] = { b, b.centroid(), i };
    }
    for (int i = 0; i < nt; ++i) {
        AABB b = triangle_aabb_fn(scene.triangles[i]);
        prims[ns + i] = { b, b.centroid(), ns + i };
    }

    // Build + flatten
    std::vector<int>       ordered;  ordered.reserve(total);
    std::vector<BuildNode> pool;     pool.reserve(2 * total);

    BuildNode* root = build_recursive(prims, 0, total, ordered, pool);

    scene.accel.nodes.reserve(pool.size());
    flatten(root, scene.accel.nodes);
    scene.accel.prim_indices = std::move(ordered);
}

// ── Public: intersect ─────────────────────────────────────────────────────────

bool bvh_intersect(const Ray& ray, const SceneAccel& accel,
                   const SceneData& scene, HitRecord& hit)
{
    if (accel.nodes.empty()) return false;

    bool  found     = false;
    float t_closest = ray.tmax;

    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = 0;  // root

    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        const BVHNode& node = accel.nodes[node_idx];

        float t_near, t_far;
        Ray   test_ray = ray;
        test_ray.tmax  = t_closest;
        if (!AABB{ {node.aabb_min[0], node.aabb_min[1], node.aabb_min[2]},
                   {node.aabb_max[0], node.aabb_max[1], node.aabb_max[2]} }
                .intersect(test_ray, t_near, t_far))
            continue;

        if (node.prim_count > 0) {
            // Leaf: test all primitives
            for (int k = 0; k < node.prim_count; ++k) {
                int prim = accel.prim_indices[node.left_first + k];
                HitRecord tmp{};
                bool did_hit = false;

                if (prim < accel.num_shapes) {
                    const Shape& s = scene.shapes[prim];
                    did_hit = intersect_shape(ray, s, t_closest, tmp);
                    if (did_hit) { tmp.material_id = s.material_id; tmp.shape_id = prim; }
                } else {
                    int ti = prim - accel.num_shapes;
                    const Triangle& tri = scene.triangles[ti];
                    did_hit = intersect_triangle(ray, tri, t_closest, tmp);
                    if (did_hit) { tmp.material_id = tri.material_id; tmp.shape_id = prim; }
                }

                if (did_hit) {
                    t_closest = tmp.t;
                    hit       = tmp;
                    found     = true;
                }
            }
        } else {
            // Interior: push children. Visit closer child last (so it's popped first).
            int left  = node.left_first;
            int right = node.right_child;
            bool visit_left_first = (ray.dir[node.split_axis] >= 0.0f);
            if (visit_left_first) {
                stack[stack_top++] = right;
                stack[stack_top++] = left;
            } else {
                stack[stack_top++] = left;
                stack[stack_top++] = right;
            }
        }
    }

    return found;
}

// ── Public: shadow any-hit traversal ─────────────────────────────────────────
// Returns true as soon as any occluder is found in [ray.tmin, ray.tmax].
// Never computes normals — only vertex data is loaded for triangles.
// Mirrors gpu_shadow_blocked in gpu_bvh.cuh.

bool bvh_shadow_blocked(const Ray& ray, const SceneAccel& accel,
                        const SceneData& scene, int* hit_shape_id)
{
    if (accel.nodes.empty()) return false;

    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = 0;

    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        const BVHNode& node = accel.nodes[node_idx];

        float t_near, t_far;
        if (!AABB{ {node.aabb_min[0], node.aabb_min[1], node.aabb_min[2]},
                   {node.aabb_max[0], node.aabb_max[1], node.aabb_max[2]} }
                .intersect(ray, t_near, t_far))
            continue;

        if (node.prim_count > 0) {
            for (int k = 0; k < node.prim_count; ++k) {
                int prim = accel.prim_indices[node.left_first + k];

                if (prim < accel.num_shapes) {
                    HitRecord tmp{};
                    const Shape& s = scene.shapes[prim];
                    if (intersect_shape(ray, s, ray.tmax, tmp)) {
                        if (hit_shape_id) *hit_shape_id = prim;
                        return true;
                    }
                } else {
                    int ti = prim - accel.num_shapes;
                    const Triangle& tri = scene.triangles[ti];
                    float t_hit, u_hit, v_hit;
                    if (intersect_triangle_v(ray,
                                             tri.v[0], tri.v[1], tri.v[2],
                                             ray.tmax, t_hit, u_hit, v_hit)) {
                        if (hit_shape_id) *hit_shape_id = prim;
                        return true;
                    }
                }
            }
        } else {
            int left  = node.left_first;
            int right = node.right_child;
            bool visit_left_first = (ray.dir[node.split_axis] >= 0.0f);
            if (visit_left_first) {
                stack[stack_top++] = right;
                stack[stack_top++] = left;
            } else {
                stack[stack_top++] = left;
                stack[stack_top++] = right;
            }
        }
    }

    return false;
}
