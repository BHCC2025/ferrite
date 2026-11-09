#!/bin/bash
# tools/fabric-detect.sh [peer]: suggest this node's two-node fabric settings (run on rank 0's node). It compares the
# IPv4 subnets here with the peer's (over ssh). Of the shared subnets the first is the one holding FERRITE_TP_HOST
# when that is set, else the lowest address, then the nearest addresses (the other half of a split link): the first
# gives FERRITE_TP_HOST and FERRITE_NET_IF, the active RDMA devices on the first two give FERRITE_ROCE_HCA, and the
# RoCE v2 GID index of the address on them FERRITE_ROCE_GID. Prints shell assignments; changes nothing.
# FERRITE_FABRIC=auto makes tools/site.sh use them (the two-node scripts do while FERRITE_TP_HOST is unset). Two DGX
# Sparks cabled port to port can share several subnets: FERRITE_TP_HOST picks the one holding that address (and its
# device pair), without it the lowest address wins.
PEER=${1:-${FERRITE_PEER:-}}
[ -n "$PEER" ] || { echo "usage: tools/fabric-detect.sh <peer ssh host> (or set FERRITE_PEER)" >&2; exit 2; }
python3 - "$PEER" <<'PY'
import ipaddress, os, subprocess, sys
peer = sys.argv[1]
def addrs(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    return [(l.split()[1], ipaddress.ip_interface(l.split()[3])) for l in out.splitlines() if " inet " in l]
mine = addrs(["ip", "-o", "-4", "addr", "show"])
theirs = {a.network for _, a in addrs(["ssh", "-n", peer, "ip", "-o", "-4", "addr", "show"])}
shared = sorted(((a, ifn) for ifn, a in mine if a.network in theirs and not a.ip.is_loopback), key=lambda x: x[0].ip)
if not shared:
    sys.exit(f"fabric-detect: no subnet shared with {peer}")
th = os.environ.get("FERRITE_TP_HOST")
key = int(ipaddress.ip_address(th)) if th else int(shared[0][0].ip)
shared.sort(key=lambda x: (abs(int(x[0].ip) - key), int(x[0].ip)))
ib = "/sys/class/infiniband"
devs, gids = [], set()
for a, ifn in shared:
    for d in sorted(os.listdir(ib)) if os.path.isdir(ib) else []:
        port = f"{ib}/{d}/ports/1"
        if not os.path.exists(f"{ib}/{d}/device/net/{ifn}") or "ACTIVE" not in open(f"{port}/state").read():
            continue
        want = "0000:0000:0000:0000:0000:ffff:%04x:%04x" % (int(a.ip) >> 16, int(a.ip) & 0xffff)
        for i in range(len(os.listdir(f"{port}/gids"))):
            try:
                g = open(f"{port}/gids/{i}").read().strip()
                t = open(f"{port}/gid_attrs/types/{i}").read().strip()
            except OSError:
                continue
            if g == want and t == "RoCE v2":
                devs.append(d)
                gids.add(i)
                break
devs = devs[:2]
print(f"FERRITE_TP_HOST={shared[0][0].ip}")
print(f"FERRITE_NET_IF={shared[0][1]}")
if devs:
    print(f"FERRITE_ROCE_HCA={','.join(devs)}")
    print(f"FERRITE_ROCE_GID={min(gids)}")
    if len(gids) > 1:
        print(f"# fabric-detect: the devices' RoCE v2 GID indices differ ({sorted(gids)}); the engine takes one", file=sys.stderr)
else:
    print("# fabric-detect: no active RDMA device on the shared subnets (NCCL over sockets only)", file=sys.stderr)
PY
