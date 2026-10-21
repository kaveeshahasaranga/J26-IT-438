"""
Visual Hierarchy Analyzer - Element Detection & Ranking
Implements locked_decisions.txt:
- "9. Element score: mean saliency inside the box."
- Detects UI element bounding boxes and ranks them by attention share.
- Generates annotated visualization with ranked badges and UX insights.
"""

import cv2
import numpy as np


class VisualHierarchyAnalyzer:
    def __init__(self, max_elements=15):
        self.max_elements = max_elements

    def detect_elements(self, bgr_image):
        """
        Fallback UI Element Detector:
        Identifies functional UI elements (buttons, cards, banners, headings, text blocks)
        using multi-scale edge and morphological component analysis.
        """
        orig_h, orig_w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Contrast adjustment & Sobel edges for UI element boundaries
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
        
        # Morphological closing to join UI text/button fragments into bounding regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
        closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(closed, 40, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = (orig_w * orig_h) * 0.0015  # min 0.15% of page
        max_area = (orig_w * orig_h) * 0.70    # max 70% of page
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if min_area < area < max_area and w > 30 and h > 18:
                candidates.append((x, y, w, h))

        # Add grid-based key semantic sections (top nav, hero left, hero right) if contours are few
        if len(candidates) < 3:
            candidates.extend(self._generate_heuristic_regions(orig_w, orig_h))
            
        # Non-Maximum Suppression to remove heavy overlapping boxes
        boxes = self._non_max_suppression(candidates, overlap_thresh=0.45)
        return boxes[:self.max_elements]

    def _generate_heuristic_regions(self, w, h):
        """Generates standard web layout anchor regions."""
        return [
            (int(w * 0.05), int(h * 0.02), int(w * 0.90), int(h * 0.10)),  # Header / Nav
            (int(w * 0.08), int(h * 0.18), int(w * 0.48), int(h * 0.35)),  # Hero Left / Title
            (int(w * 0.58), int(h * 0.18), int(w * 0.35), int(h * 0.40)),  # Hero Media Right
            (int(w * 0.08), int(h * 0.45), int(w * 0.25), int(h * 0.10)),  # Primary CTA Button
        ]

    def _non_max_suppression(self, boxes, overlap_thresh=0.45):
        if not boxes:
            return []
        
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]
        
        order = areas.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= overlap_thresh)[0]
            order = order[inds + 1]
            
        return [tuple(boxes[k]) for k in keep]

    def analyze_hierarchy(self, bgr_image, saliency_map, explicit_boxes=None):
        """
        Ranks UI elements by their mean saliency score inside the bounding box.
        Returns:
            elements: list of dicts with rank, score, bbox, type, and UX insights.
            annotated_image: BGR image with visual hierarchy bounding boxes and badges.
        """
        orig_h, orig_w = bgr_image.shape[:2]
        boxes = explicit_boxes if explicit_boxes else self.detect_elements(bgr_image)
        
        # Compute mean saliency score for each box (per locked_decisions.txt rule 9)
        element_scores = []
        for (x, y, w, h) in boxes:
            # Ensure boundaries are inside image
            x0 = max(0, min(x, orig_w - 1))
            y0 = max(0, min(y, orig_h - 1))
            x1 = max(x0 + 1, min(x + w, orig_w))
            y1 = max(y0 + 1, min(y + h, orig_h))
            
            roi = saliency_map[y0:y1, x0:x1]
            mean_score = float(np.mean(roi)) if roi.size > 0 else 0.0
            peak_score = float(np.max(roi)) if roi.size > 0 else 0.0
            
            # Element category heuristic
            aspect = w / max(1, h)
            area_ratio = (w * h) / (orig_w * orig_h)
            if aspect > 2.5 and area_ratio < 0.04 and y0 < orig_h * 0.7:
                category = "Button / CTA"
            elif aspect > 2.0 and y0 < orig_h * 0.35:
                category = "Header / Title"
            elif 0.7 < aspect < 1.6 and area_ratio > 0.06:
                category = "Media / Visual"
            elif area_ratio > 0.10:
                category = "Content Card"
            else:
                category = "Interactive Element"

            element_scores.append({
                "x": int(x0),
                "y": int(y0),
                "w": int(x1 - x0),
                "h": int(y1 - y0),
                "mean_score": mean_score,
                "peak_score": peak_score,
                "category": category,
            })

        # Sort elements by mean saliency in descending order
        element_scores.sort(key=lambda item: item["mean_score"], reverse=True)
        
        max_score = element_scores[0]["mean_score"] if element_scores else 1.0
        if max_score <= 1e-6:
            max_score = 1.0

        # Assign ranks and UX feedback
        elements = []
        for rank, item in enumerate(element_scores, start=1):
            normalized_percent = int(round((item["mean_score"] / max_score) * 100))
            
            if rank == 1:
                feedback = "Primary Focus: Immediate eye-catcher within first 500ms of viewing."
                level = "Dominant"
            elif rank in (2, 3):
                feedback = "Secondary Anchor: Strong secondary fixation point. Ideal for key supporting information."
                level = "High"
            elif rank in (4, 5, 6):
                feedback = "Tertiary Flow: Noticeable during conscious scanning. Suitable for detail blocks."
                level = "Moderate"
            else:
                feedback = "Subtle / Peripheral: Receives peripheral attention. Needs higher contrast if critical."
                level = "Low"

            elements.append({
                "id": f"elem-{rank}",
                "rank": rank,
                "category": item["category"],
                "level": level,
                "mean_score": round(item["mean_score"], 4),
                "peak_score": round(item["peak_score"], 4),
                "attention_percentage": normalized_percent,
                "bbox": [item["x"], item["y"], item["w"], item["h"]],
                "feedback": feedback
            })

        # Draw annotated image
        annotated_image = self.draw_hierarchy_overlays(bgr_image, elements)
        return elements, annotated_image

    def draw_hierarchy_overlays(self, bgr_image, elements):
        """Draws aesthetic ranked bounding boxes and badges on the image."""
        annotated = bgr_image.copy()
        
        # Color palette by rank tier (BGR format)
        colors = {
            1: (0, 180, 255),    # Vibrant Amber / Gold for #1
            2: (255, 170, 0),    # Cyan for #2
            3: (255, 75, 150),   # Purple/Pink for #3
            "default": (120, 200, 80) # Lime Green for lower ranks
        }

        for el in elements:
            x, y, w, h = el["bbox"]
            rank = el["rank"]
            color = colors.get(rank, colors["default"])
            
            # Draw semi-transparent rectangle border
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
            
            # Badge text (e.g., "#1 · 98%")
            badge_text = f"#{rank} ({el['attention_percentage']}%)"
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Badge background pill
            badge_y1 = max(0, y - text_h - 8)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = x + text_w + 12
            
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(annotated, badge_text, (badge_x1 + 6, badge_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (15, 15, 15), thickness, cv2.LINE_AA)

        return annotated
