# -*- coding: utf-8 -*-
"""
Feature mapping between CICIDS-2017 and UNSW-NB15
Maps semantically similar features to enable cross-dataset evaluation
"""

# Feature mapping: UNSW column name -> CICIDS column name
# Based on semantic similarity of network flow features
FEATURE_MAPPING = {
    # Duration and timing
    'dur': 'Flow Duration',
    'rate': 'Flow Packets/s',
    
    # Source (forward) packets and bytes
    'spkts': 'Total Fwd Packets',
    'sbytes': 'Total Length of Fwd Packets',
    'smean': 'Fwd Packet Length Mean',
    'sload': 'Fwd Avg Bytes/Bulk',
    'sjit': 'Fwd IAT Mean',
    'sinpkt': 'Fwd Packets/s',
    
    # Destination (backward) packets and bytes
    'dpkts': 'Total Backward Packets',
    'dbytes': 'Total Length of Bwd Packets',
    'dmean': 'Bwd Packet Length Mean',
    'dload': 'Bwd Avg Bytes/Bulk',
    'djit': 'Bwd IAT Mean',
    'dinpkt': 'Bwd Packets/s',
    
    # TCP window and flags
    'swin': 'Init_Win_bytes_forward',
    'dwin': 'Init_Win_bytes_backward',
    'stcpb': 'Subflow Fwd Bytes',
    'dtcpb': 'Subflow Bwd Bytes',
    'synack': 'SYN Flag Count',
    'ackdat': 'ACK Flag Count',
    
    # TTL
    'sttl': 'Fwd Header Length',  # Approximate mapping
    'dttl': 'Bwd Header Length',  # Approximate mapping
    
    # Loss (no direct equivalent in CICIDS, but can approximate)
    'sloss': 'Fwd Packet Length Std',  # Std as proxy for loss/variation
    'dloss': 'Bwd Packet Length Std',
    
    # State and protocol features (UNSW specific - no direct CICIDS equivalent)
    # These are UNSW-specific categorical columns, will be dropped
    'proto': None,
    'service': None,
    'state': None,
    
    # Connection features (UNSW specific temporal features)
    # No direct CICIDS equivalents, will be dropped or set to 0
    'ct_dst_ltm': None,
    'ct_dst_sport_ltm': None,
    'ct_dst_src_ltm': None,
    'ct_flw_http_mthd': None,
    'ct_ftp_cmd': None,
    'ct_src_dport_ltm': None,
    'ct_src_ltm': None,
    'ct_srv_dst': None,
    'ct_srv_src': None,
    'ct_state_ttl': None,
    
    # UNSW specific features
    'is_ftp_login': None,
    'is_sm_ips_ports': None,
    'response_body_len': None,
    'trans_depth': None,
    'tcprtt': 'Flow IAT Mean',  # Round trip time ~ IAT
    
    # ID and labels (skip)
    'id': None,
    'label': None,
    'attack_cat': None,
}

# Reverse mapping: CICIDS -> UNSW (for future use)
REVERSE_MAPPING = {v: k for k, v in FEATURE_MAPPING.items() if v is not None}

def get_mapped_columns():
    """Get list of UNSW columns that can be mapped to CICIDS"""
    return {k: v for k, v in FEATURE_MAPPING.items() if v is not None}

def get_unmapped_columns():
    """Get list of UNSW columns that cannot be mapped"""
    return [k for k, v in FEATURE_MAPPING.items() if v is None]

if __name__ == "__main__":
    mapped = get_mapped_columns()
    unmapped = get_unmapped_columns()
    
    print(f"Mapped features: {len(mapped)}")
    print(f"Unmapped features: {len(unmapped)}")
    print(f"\nMapped:")
    for unsw, cicids in sorted(mapped.items()):
        print(f"  {unsw:20s} -> {cicids}")
    print(f"\nUnmapped (will be filled with 0 or dropped):")
    for col in sorted(unmapped):
        print(f"  {col}")
