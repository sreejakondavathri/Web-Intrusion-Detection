# src/project_report.py
import json, os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
METRICS_FILE = os.path.join(REPORTS_DIR, "metrics_summary.json")

def build_project_report(example_forensic_pdf=None):
    out_file = os.path.join(REPORTS_DIR, f"project_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    c = canvas.Canvas(out_file, pagesize=letter)
    w, h = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-60, "X-WIDS — Project Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, h-80, f"Generated: {datetime.now().isoformat()}")

    # Insert metrics summary text
    metrics = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

    y = h - 110
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Evaluation Metrics Summary")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Total requests evaluated: {metrics.get('n_total','-')}")
    y -= 14
    c.drawString(40, y, f"Malicious requests (labeled): {metrics.get('n_malicious','-')}")
    y -= 14
    c.drawString(40, y, f"Accuracy: {metrics.get('accuracy','-')}")
    y -= 14
    c.drawString(40, y, f"ROC-AUC: {metrics.get('roc_auc','-')}")
    y -= 20

    # Insert confusion matrix image
    cm_img = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_img):
        try:
            img = ImageReader(cm_img)
            c.drawImage(img, 40, y-200, width=500, height=200)
            y -= 210
        except Exception as e:
            c.drawString(40, y, f"Could not insert confusion matrix image: {e}")

    # Insert ROC if exists
    roc_img = os.path.join(REPORTS_DIR, "roc_curve.png")
    if os.path.exists(roc_img):
        try:
            img = ImageReader(roc_img)
            c.drawImage(img, 40, y-200, width=500, height=200)
            y -= 210
        except Exception as e:
            c.drawString(40, y, f"Could not insert ROC image: {e}")

    # If an example forensic pdf provided, attach as separate page (we will just note it)
    if example_forensic_pdf and os.path.exists(example_forensic_pdf):
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, h-60, "Included Forensic Report (attached separately)")
        c.drawString(40, h-80, example_forensic_pdf)

    c.showPage()
    c.save()
    print(f"✅ Project report created at: {out_file}")
    return out_file

if __name__ == "__main__":
    build_project_report()
