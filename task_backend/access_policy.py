"""Network roles use TCP peer addresses, never client-provided HTTP headers."""
import ipaddress
import os

_CAMPUS = ipaddress.ip_network("10.0.0.0/8")


def peer_address(value):
    address = ipaddress.ip_address(value)
    return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address


def is_campus_peer(value):
    address = peer_address(value)
    if address.version != 4 or address not in _CAMPUS:
        return False
    # A router can translate operator LAN clients to a campus address. Exclude
    # that site's NAT egress addresses as well as the original operator subnet.
    operator_egress = os.environ.get("ORBBEC_OPERATOR_NAT_NETWORKS", "")
    return not any(address in ipaddress.ip_network(network.strip())
                   for network in operator_egress.split(",") if network.strip())


def management_allowed(value):
    return peer_address(value).is_loopback or is_campus_peer(value)
