#pragma once
// Warp-aggregated slot allocation, shared by CUDA kernels and OptiX programs.

// Returns a unique output index for every thread where `active` is true,
// using a single atomicAdd per warp.  Inactive threads receive -1.
//
// The __shfl_sync mask must be __activemask() literally rather than the
// captured ballot result: the OptiX PTX validator rejects the latter form
// even though they are semantically identical after the early return.
__device__ __forceinline__
int warp_compact_slot(int* d_count, bool active) {
    unsigned int mask = __ballot_sync(__activemask(), active);
    if (!active) return -1;
    int lane = threadIdx.x & 31;
    int rank = __popc(mask & ((1u << lane) - 1));
    int base = -1;
    if (rank == 0) base = atomicAdd(d_count, (int)__popc(mask));
    base = __shfl_sync(__activemask(), base, __ffs((int)mask) - 1);
    return base + rank;
}
