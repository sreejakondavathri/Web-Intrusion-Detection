import re
import pandas as pd
import os
from urllib.parse import urlparse

# Log format example:
# 127.0.0.1 - - [12/Jan/2025:10:10:10 +0000] "GET /home?id=1 HTTP/1.1" 200 512

LOG_PATTERN = r'(\S+) - - \[(.*?)\] "(.*?)" (\d{3}) (\d+|-)'

def parse_log_line(line):
    """Parses a single line of a web server log."""
    match = re.match(LOG_PATTERN, line)
    if not match:
        return None

    ip, timestamp, request, status, size = match.groups()
    try:
        method, url, protocol = request.split(" ")
    except ValueError:
        return None  # Avoid bad lines

    parsed_url = urlparse(url)

    return {
        "ip": ip,
        "timestamp": timestamp,
        "method": method,
        "url": url,
        "path": parsed_url.path,
        "query": parsed_url.query,
        "status": int(status),
        "size": 0 if size == "-" else int(size)
    }

def extract_features(log_file, output_csv="data/processed_data.csv"):
    """Extract features from log file and save to CSV."""
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return

    records = []
    with open(log_file, "r") as f:
        for line in f:
            data = parse_log_line(line)
            if data:
                records.append(data)

    if not records:
        print("⚠ No valid log entries found.")
        return

    df = pd.DataFrame(records)

    # Feature-1: URL Length
    df["url_length"] = df["url"].apply(len)

    # Feature-2: Query Count
    df["query_params"] = df["query"].apply(lambda q: len(q.split("&")) if q else 0)

    # Feature-3: Special Character Density
    special_chars = ["'", '"', "<", ">", ";", "--", "#", "%", "(", ")"]
    df["special_char_density"] = df["url"].apply(
        lambda x: sum(c in x for c in special_chars) / len(x) if len(x) > 0 else 0
    )

    # Feature-4: HTTP Method Encoding
    df["method_encoded"] = df["method"].apply(lambda x: 1 if x == "POST" else (0 if x == "GET" else 2))

    # Feature-5: HTTP Status Category
    df["status_category"] = df["status"].apply(lambda x: x // 100)  # 2xx → 2, 4xx → 4, etc.

    # Save processed CSV
    df.to_csv(output_csv, index=False)
    print(f"✅ Feature Extraction Completed! Saved to {output_csv}")
    print(df.head())  # Show preview

if __name__ == "__main__":
    extract_features("data/sample_logs.txt")
