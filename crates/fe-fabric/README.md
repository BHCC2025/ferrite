# fe-fabric

TP2 collectives between two DGX Sparks cabled port to port (rank 0 on node A, rank 1 on node B, RoCE v2). The
fabric settings (rank 0's address, RDMA devices, GID index, NCCL interface) are `FERRITE_*` environment variables
(crates/fe-site, tools/site.sh; `tools/fabric-detect.sh` suggests them for a pair of nodes).

```rust
use fe_fabric::{DataType, Fabric};
let mut fab = Fabric::connect(rank, &fe_site::tp_addr())?; // TCP bootstrap (rank 0 listens) + NCCL communicator
fab.enable_roce(1 << 20)?;                                   // both ranks; FE_ROCE=0 on either keeps NCCL for both
// the TP2 reduction after a row-parallel linear: hidden = bf16(hidden + (rank 0's partial + rank 1's))
fab.reduce_to_bf16(&stream, partial_f32.ptr(), hidden.ptr(), Some(hidden.ptr()), rows * 5120)?;
fab.all_reduce_sum(&stream, buf, buf, n, DataType::F32)?;   // NCCL
fab.all_gather(&stream, send, recv, n, DataType::Bf16)?;     // NCCL: recv = [rank 0 | rank 1]
fab.check()?;                                               // after a sync: RoCE timeouts / proxy failures
```

- `reduce_to_bf16` sums in rank order then adds the residual, so both ranks hold identical bits. On the RoCE
  path (`kernels/k-fabric`, `src/roce.rs`; the design follows b12x's RoCEnante, as TensorFold's roce_gather uses
  it): one launch stages the partial into pinned memory and rings a doorbell, a proxy thread RDMA-writes it and a
  sequence flag (same RC queue pair, RDMA devices `FERRITE_ROCE_HCA`, GID index `FERRITE_ROCE_GID`), the kernel
  waits for the peer's flag and sums.
  No host sync, graph-capturable (the device epoch advances per exchange). Above the cap: NCCL all-reduce
  + a conversion kernel.
- NCCL is dlopen'd (`FERRITE_NCCL_LIB`, else `libnccl.so.2` on the loader path); it takes
  its own `NCCL_IB_GID_INDEX`, `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`, which tools/site.sh sets from the same
  settings (`$FERRITE_FAB`).

## Measured (fabric-reduce, 2000 calls back to back, exact results checked both paths)

| f32 elems (rows x 5120) | bytes a rank | RoCE us | NCCL all-reduce + cvt us | RoCE in a CUDA graph us |
|---|---|---|---|---|
| 1024 | 4096 | 13.4 | 21.6 | 10.5 |
| 5120 (1 row) | 20480 | 14.1 | 23.6 | 12.0 |
| 10240 (2 rows) | 40960 | 16.6 | 30.2 | 15.0 |
| 40960 (8 rows) | 163840 | 35.9 | 91.4 | 33.2 |
| 81920 (16 rows) | 327680 | 52.2 | 177.4 | 50.1 |
| 262144 | 1048576 | 127.0 | 217.4 | 127.0 |

NCCL alone (fabric-bench): all-reduce f32 20 KB 20.7 us, all-gather 20 KB 17.7 us.

Reproduce (both nodes, same address: rank 0's fabric IP and a free port; `FERRITE_TP_HOST` set, or
`FERRITE_FABRIC=auto` with `FERRITE_PEER`):
```
nodeA$ . tools/site.sh; env $FERRITE_FAB ./target/release/fabric-reduce 0 $FERRITE_TP_HOST:29530 2000
nodeB$ . tools/site.sh; env $FERRITE_FAB ./target/release/fabric-reduce 1 $FERRITE_TP_HOST:29530 2000
```
