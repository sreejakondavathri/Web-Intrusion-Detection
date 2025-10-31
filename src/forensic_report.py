from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def create_forensic_report(record: dict, shap_img_path: str = None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REPORTS_DIR, f"forensic_{timestamp}.pdf")

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    y = height - 50  # Starting height for text

    # ✅ Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Forensic Incident Report")
    y -= 30

    # ✅ Metadata
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    # ✅ Incident Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Incident Details:")
    y -= 20

    c.setFont("Helvetica", 10)
    for key, label in [
        ('ip', 'IP Address'),
        ('timestamp', 'Timestamp'),
        ('url', 'URL'),
        ('anomaly_score', 'Anomaly Score'),
        ('stage2_label', 'Stage-2 Prediction'),
        ('suggested_action', 'Suggested Action')
    ]:
        c.drawString(60, y, f"{label}: {record.get(key, 'N/A')}")
        y -= 15

    # ✅ Feature Section
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Feature Values:")
    y -= 20
    c.setFont("Helvetica", 10)

    for key in ["url_length", "query_params", "special_char_density", "method_encoded", "status_category"]:
        c.drawString(60, y, f"{key}: {record.get(key, 'N/A')}")
        y -= 15

    # ✅ SHAP Image Section
    if shap_img_path and os.path.exists(shap_img_path):
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "SHAP Explanation Graph:")
        y -= 200  # Reserve space
        try:
            c.drawImage(shap_img_path, 50, y, width=500, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            c.drawString(50, y + 20, f"Could not insert SHAP image: {e}")

    c.save()
    return filename
