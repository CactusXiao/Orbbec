"""Network roles use TCP peer addresses, never client-provided HTTP headers."""
import ipaddress

_CAMPUS = ipaddress.ip_network("10.0.0.0/8")


def peer_address(value):
    address = ipaddress.ip_address(value)
    return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address


def is_campus_peer(value):
    address = peer_address(value)
    return address.version == 4 and address in _CAMPUS


def management_allowed(value):
    return peer_address(value).is_loopback or is_campus_peer(value)
