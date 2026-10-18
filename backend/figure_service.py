import base64
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage

from backend.config import settings
from backend.retrieval import normalize_arxiv_id
from backend.schemas import FigureResponse

logger = logging.getLogger("journallab.figures")

class FigureService:
    """Extract and analyze architecture and framework figures from arXiv papers."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "JournalLab-Research-Figure-Explorer/1.0"
        })
        settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def score_caption(self, caption: str) -> int:
        """Score figure captions to prioritize framework/architecture diagrams."""
        text = caption.lower()
        keywords = {
            "architecture": 8,
            "overall framework": 8,
            "framework": 6,
            "pipeline": 6,
            "overview": 5,
            "workflow": 5,
            "model": 3,
            "method": 3,
            "system": 2,
            "figure 1": 3,
            "fig. 1": 3,
            "results": -2,
            "ablation": -2,
        }
        return sum(weight for word, weight in keywords.items() if word in text)

    def to_png(self, image_bytes: bytes, content_type: str = "") -> Optional[bytes]:
        """Convert arbitrary downloaded image/svg bytes into standard PNG bytes."""
        content_type = content_type.lower()
        is_svg = "svg" in content_type or image_bytes.lstrip().startswith(b"<svg") or b"<svg" in image_bytes[:500]

        if is_svg:
            try:
                import fitz
                doc = fitz.open(stream=image_bytes, filetype="svg")
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                return pix.tobytes("png")
            except Exception:
                try:
                    import cairosvg
                    return cairosvg.svg2png(bytestring=image_bytes, output_width=1200)
                except Exception:
                    pass

        try:
            im = PILImage.open(BytesIO(image_bytes))
            im.thumbnail((1200, 1200))
            if im.mode in ("RGBA", "LA"):
                rgba = im.convert("RGBA")
                background = PILImage.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                im = background
            else:
                im = im.convert("RGB")

            buffer = BytesIO()
            im.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
        except Exception as e:
            logger.warning("Image conversion failed: %s", e)
            return None

    def fetch_html_figure(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Scrape best-captioned figure from arXiv HTML version."""
        page_url = f"https://arxiv.org/html/{paper_id}"
        try:
            resp = self.session.get(page_url, timeout=15)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            elements = soup.select("figure, .ltx_figure")
            candidates = []
            seen = set()

            for elem in elements:
                img = elem.find("img", src=True)
                if not img:
                    continue
                src = img.get("src") or img.get("data-src")
                if not src or src.startswith("data:"):
                    continue

                caption_tag = elem.find("figcaption") or elem.find(class_=re.compile("caption", re.I))
                caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""
                alt = img.get("alt", "").strip()
                caption = caption or alt or "Figure from the paper"

                key = (src, caption)
                if key in seen:
                    continue
                seen.add(key)

                candidates.append({
                    "caption": caption,
                    "image_url": urljoin(resp.url, src),
                    "paper_url": resp.url,
                    "score": self.score_caption(f"{caption} {alt}"),
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)

            for cand in candidates[:6]:
                try:
                    img_resp = self.session.get(cand["image_url"], timeout=15)
                    if img_resp.status_code != 200:
                        continue
                    png = self.to_png(img_resp.content, img_resp.headers.get("content-type", ""))
                    if png:
                        cand["png_bytes"] = png
                        cand["fetch_mode"] = "arXiv HTML figure"
                        return cand
                except requests.RequestException:
                    continue
        except requests.RequestException as e:
            logger.debug("arXiv HTML fetch error for %s: %s", paper_id, e)

        return None

    def fetch_pdf_page(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """Fallback: render PDF page likely to contain an architecture/framework figure."""
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"
        try:
            resp = self.session.get(pdf_url, timeout=25)
            if resp.status_code != 200:
                return None

            import fitz
            doc = fitz.open(stream=resp.content, filetype="pdf")
            candidates = []

            for page_no in range(min(len(doc), 10)):
                text = doc[page_no].get_text().lower()
                if not text.strip():
                    continue
                score = self.score_caption(text[:8000])
                if "architecture" in text:
                    score += 4
                if "proposed framework" in text or "pipeline" in text:
                    score += 4
                candidates.append((score, page_no))

            page_no = max(candidates)[1] if candidates else 0
            page = doc[page_no]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)

            return {
                "caption": f"PDF Page {page_no + 1} framework preview.",
                "image_url": pdf_url,
                "paper_url": f"https://arxiv.org/abs/{paper_id}",
                "png_bytes": pix.tobytes("png"),
                "fetch_mode": "PDF page preview",
            }
        except Exception as e:
            logger.warning("PDF page fallback error for %s: %s", paper_id, e)
            return None

    def get_paper_figure(self, arxiv_id: str) -> Optional[FigureResponse]:
        """Fetch figure (from disk cache or live arXiv) and return FigureResponse."""
        norm_id = normalize_arxiv_id(arxiv_id)
        cache_img = settings.FIGURES_DIR / f"{norm_id}_figure.png"
        caption = "Paper architecture diagram"
        fetch_mode = "Cached figure"

        png_bytes = None
        if cache_img.exists() and cache_img.stat().st_size > 0:
            png_bytes = cache_img.read_bytes()
        else:
            fig_data = self.fetch_html_figure(norm_id) or self.fetch_pdf_page(norm_id)
            if fig_data and "png_bytes" in fig_data:
                png_bytes = fig_data["png_bytes"]
                caption = fig_data.get("caption", caption)
                fetch_mode = fig_data.get("fetch_mode", fetch_mode)
                try:
                    cache_img.write_bytes(png_bytes)
                except Exception as e:
                    logger.warning("Could not write figure cache for %s: %s", norm_id, e)

        if not png_bytes:
            return None

        b64_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
        return FigureResponse(
            arxiv_id=norm_id,
            caption=caption,
            image_data_uri=b64_uri,
            fetch_mode=fetch_mode,
            paper_url=f"https://arxiv.org/abs/{norm_id}",
            explanation=None,
        )

    def explain_figure(self, arxiv_id: str, title: str, abstract: str) -> str:
        """Call vision LLM (if configured) or produce structured heuristic explanation."""
        api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return (
                "Vision explanation unavailable: GROQ_API_KEY is not configured. "
                "The diagram displays the system pipeline or architecture from the paper."
            )

        fig_res = self.get_paper_figure(arxiv_id)
        if not fig_res or not fig_res.image_data_uri:
            return "Figure could not be retrieved for visual explanation."

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            prompt = f"""
Explain this research-paper figure for a researcher:
Paper: {title}
arXiv ID: {arxiv_id}
Abstract: {abstract[:1200]}
Figure caption: {fig_res.caption[:400]}

Write at most 80 words using these three sections:
- What it shows:
- How to read it:
- Why it matters:
"""
            resp = client.chat.completions.create(
                model=settings.GROQ_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": fig_res.image_data_uri},
                        },
                    ],
                }],
                temperature=0.1,
                max_completion_tokens=250,
            )
            return resp.choices[0].message.content or "No explanation generated."
        except Exception as e:
            logger.warning("Vision explanation failed: %s", e)
            return f"Vision explanation unavailable: {e}"


figure_service = FigureService()
