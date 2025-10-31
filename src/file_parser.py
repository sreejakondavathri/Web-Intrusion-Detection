# src/file_parser.py

import pandas as pd
import json
import os
from scapy.all import rdpcap, TCP
import re
from io import StringIO

# ------------- Common Log Fields Template -------------
COLUMNS = ["ip", "timestamp", "method", "url", "status", "response_size"]

# ------------------------------------------------------
# 1. Parse TXT/LOG format (Common Log Format)
# ------------------------------------------------------
def parse_txt_logs(content):
    """
    Parse plain text Apache/Nginx style logs.
    """
    log_pattern = re.compile(
        r'(?P<ip>\S+) - - \[(?P<timestamp>.*?)\] "(?P<method>\S+) (?P<url>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+)'
    )
    records = []
    for line in content.splitlines():
        match = log_pattern.match(line)
        if match:
            data = match.groupdict()
            records.append({
                "ip": data["ip"],
                "timestamp": data["timestamp"],
                "method": data["method"],
                "url": data["url"],
                "status": int(data["status"]),
                "response_size": int(data["size"])
            })
    return pd.DataFrame(records)

# ------------------------------------------------------
# 2. Parse CSV format
# ------------------------------------------------------
def parse_csv_logs(content):
    try:
        df = pd.read_csv(StringIO(content))
        # Ensure expected columns exist
        required_cols = ["ip", "timestamp", "method", "url", "status", "response_size"]
        # Try to auto-detect column name variations
        df.columns = [col.strip().lower() for col in df.columns]
        rename_map = {
            "status_code": "status",
            "bytes": "response_size",
            "path": "url"
        }
        df = df.rename(columns=rename_map)
        for col in required_cols:
            if col not in df.columns:
                df[col] = None  # fill missing columns
        return df[required_cols]
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {e}")

# ------------------------------------------------------
# 3. Parse JSON format (list of records or one-per-line)
# ------------------------------------------------------
def parse_json_logs(content):
    records = []
    try:
        # Try loading as a full JSON array
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
    except:
        # Try line-delimited JSON
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    for item in data:
        rec = {
            "ip": item.get("ip") or item.get("client_ip") or "0.0.0.0",
            "timestamp": item.get("timestamp") or item.get("time") or "",
            "method": item.get("method") or item.get("verb") or "GET",
            "url": item.get("url") or item.get("path") or "/",
            "status": item.get("status") or item.get("status_code") or 200,
            "response_size": item.get("response_size") or item.get("bytes") or 0
        }
        records.append(rec)
    return pd.DataFrame(records)

# ------------------------------------------------------
# 4. Parse PCAP using Scapy (HTTP over TCP port 80)
# ------------------------------------------------------
def parse_pcap_logs(file_path):
    packets = rdpcap(file_path)
    records = []

    for pkt in packets:
        if pkt.haslayer(TCP) and pkt[TCP].dport == 80 or pkt[TCP].sport == 80:
            try:
                payload = bytes(pkt[TCP].payload).decode(errors="ignore")
                # Extract HTTP requests
                request_line = payload.split("\r\n")[0]
                match = re.match(r"(GET|POST|PUT|DELETE|HEAD) (.*?) HTTP", request_line)
                if match:
                    method, url = match.groups()
                    records.append({
                        "ip": pkt[0][1].src,
                        "timestamp": "pcap_capture",
                        "method": method,
                        "url": url,
                        "status": 200,          # PCAP does not contain response code (unless separate packet analyzed)
                        "response_size": len(payload)
                    })
            except:
                continue
    return pd.DataFrame(records)

# ------------------------------------------------------
# Master Function
# ------------------------------------------------------
def parse_uploaded_file(uploaded_file):
    """
    Detect file type by extension and parse accordingly.
    """
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue().decode(errors="ignore")

    if filename.endswith((".txt", ".log")):
        return parse_txt_logs(content)
    elif filename.endswith(".csv"):
        return parse_csv_logs(content)
    elif filename.endswith(".json"):
        return parse_json_logs(content)
    elif filename.endswith(".pcap"):
        # For PCAP, we need to write to temp file first
        temp_path = "data/temp_upload.pcap"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return parse_pcap_logs(temp_path)
    else:
        raise ValueError("Unsupported file format. Please upload TXT, CSV, JSON, or PCAP.")
