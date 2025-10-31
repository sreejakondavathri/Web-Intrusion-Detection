# src/synthetic_attack_generator.py
import random
from datetime import datetime, timedelta
import os

OUT_PATH = "data/sample_logs.txt"

common_paths = [
    "/home", "/index", "/products", "/search", "/login", "/admin", "/dashboard",
    "/api/v1/users", "/api/v1/login", "/upload", "/assets/img.png"
]

malicious_payloads = [
    "/login?id=1' OR '1'='1",
    "/product?id=../../../../etc/passwd",
    "/search?q=<script>alert(1)</script>",
    "/comment?c=<img src=x onerror=alert(1)>",
    "/api/v1/users?name=admin'--",
    "/admin.php?file=../../../../etc/passwd",
    "/upload?file=../../../../../windows/win.ini",
    "/?q=%27%3B+DROP+TABLE+users%3B--",
    "/search?q=%3Csvg%2Fonload%3Dalert(1)%3E"
]

def ts(i):
    t = datetime.utcnow() + timedelta(seconds=i)
    return t.strftime("%d/%b/%Y:%H:%M:%S +0000")

def random_ip():
    return "{}.{}.{}.{}".format(random.randint(1,255), random.randint(0,255), random.randint(0,255), random.randint(0,255))

def gen_line(ip, timestamp, request, status=200, size=512):
    return f'{ip} - - [{timestamp}] "{request}" {status} {size}'

def generate(num_benign=200, num_mal=60, out_path=OUT_PATH):
    lines = []
    idx = 0
    # benign
    for _ in range(num_benign):
        ip = random_ip()
        path = random.choice(common_paths)
        if random.random() < 0.2:
            # add query benign
            path = path + "?q=hello"
        method = random.choice(["GET","GET","GET","POST"])
        request = f"{method} {path} HTTP/1.1"
        lines.append(gen_line(ip, ts(idx), request, status=200, size=random.randint(200,1500)))
        idx += 1

    # malicious
    for _ in range(num_mal):
        ip = random_ip()
        payload = random.choice(malicious_payloads)
        method = random.choice(["GET","POST"])
        request = f"{method} {payload} HTTP/1.1"
        # some triggers return 500 or 403
        status = random.choice([200,200,403,500])
        lines.append(gen_line(ip, ts(idx), request, status=status, size=random.randint(50,1200)))
        idx += 1

    # shuffle to simulate real traffic
    random.shuffle(lines)

    # append to file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Appended {len(lines)} lines to {out_path}")

if __name__ == "__main__":
    generate()
