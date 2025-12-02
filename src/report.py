# src/report.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

class BayesianReport:
    def __init__(self, outdir="results"):
        self.outdir = outdir
        self.styles = getSampleStyleSheet()

    def build_report(self, filename="bayesian_report.pdf"):
        doc = SimpleDocTemplate(
            os.path.join(self.outdir, filename),
            pagesize=A4
        )

        story = []
        style = self.styles["Normal"]
        title = self.styles["Title"]

        story.append(Paragraph("Bayesian Updating Report", title))
        story.append(Spacer(1, 20))

        figs = [
            "fig_M1_timeseries.png",
            "fig_M2_timeseries.png",
            "fig_M3_timeseries.png",
            "fig_M3_validation.png",
        ]

        for f in figs:
            path = os.path.join(self.outdir, f)
            if os.path.exists(path):
                story.append(Image(path, width=400, height=300))
                story.append(Spacer(1, 20))
                story.append(Paragraph(f, style))
                story.append(Spacer(1, 20))

        doc.build(story)
