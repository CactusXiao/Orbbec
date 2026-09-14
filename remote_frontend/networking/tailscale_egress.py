#!/usr/bin/env python3
"""Route only the selected Tailscale UDP socket through a chosen Linux NIC."""
import argparse
import fcntl
import ipaddress
import json
import os
import subprocess


def ip(*args, check=True):
    return subprocess.run(["/usr/sbin/ip", *map(str, args)], check=check,
                          capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--table", type=int, default=41641)
    parser.add_argument("--priority", type=int, default=10441)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("root is required to manage policy routing")
    if not 1024 <= args.port <= 65535 or args.table in (0, 253, 254, 255):
        parser.error("invalid UDP port or reserved routing table")
    with open("/run/orbbec-tailscale-egress.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rules = {r["priority"]: r for r in json.loads(ip("-j", "-4", "rule"))}
        for priority, table in [(args.priority, "main"), (args.priority + 1, str(args.table))]:
            r = rules.get(priority)
            if r and not (r.get("uid_start") == args.uid == r.get("uid_end")
                          and r.get("ipproto") == "udp" and r.get("sport") == args.port
                          and str(r.get("table")) == table
                          and (priority != args.priority or r.get("suppress_prefixlen") == 0)):
                raise RuntimeError(f"Refusing to overwrite unrelated rule {priority}")
        devices = json.loads(ip("-j", "-4", "address", "show", "dev", args.interface, check=False) or "[]")
        addresses = [a for d in devices if d.get("operstate") == "UP"
                     for a in d.get("addr_info", []) if a.get("scope") == "global"]
        gateways = json.loads(ip("-j", "-4", "route", "show", "default", "dev", args.interface, check=False) or "[]")
        if args.remove or not addresses or not gateways:
            for priority in [args.priority + 1, args.priority]:
                if priority in rules:
                    ip("-4", "rule", "del", "priority", priority)
            ip("-4", "route", "flush", "table", args.table, check=False)
            print("Tailscale uses normal routing (selected interface unavailable or removed)")
            return
        address = addresses[0]
        local = address["local"]
        network = str(ipaddress.ip_network(f"{local}/{address['prefixlen']}", strict=False))
        gateway = sorted(gateways, key=lambda r: r.get("metric", 0))[0]["gateway"]
        ip("-4", "route", "replace", "table", args.table, network, "dev", args.interface,
           "scope", "link", "src", local)
        ip("-4", "route", "replace", "table", args.table, "default", "via", gateway,
           "dev", args.interface, "src", local)
        for route in json.loads(ip("-j", "-4", "route", "show", "table", args.table)):
            if route.get("dst") not in (network, "default") and route.get("dev") == args.interface:
                ip("-4", "route", "del", "table", args.table, route["dst"], "dev", args.interface)
        # Preserve existing specific LAN routes; replace only the default for this socket.
        selector = ["uidrange", f"{args.uid}-{args.uid}", "ipproto", "udp", "sport", args.port]
        if args.priority not in rules:
            ip("-4", "rule", "add", "priority", args.priority, *selector,
               "lookup", "main", "suppress_prefixlength", 0)
        if args.priority + 1 not in rules:
            ip("-4", "rule", "add", "priority", args.priority + 1, *selector,
               "lookup", args.table)
        print(f"Tailscale UDP {args.port}, uid {args.uid}: {args.interface} via {gateway}")


if __name__ == "__main__":
    main()
