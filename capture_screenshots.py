#!/usr/bin/env python3
"""
Dynamic Streamlit UI Screenshot Capture Automation
==================================================
Runs the Streamlit DIP & CV app on localhost:8501, navigates through the modules,
and captures high-resolution screenshots for academic presentation slides.
"""

import os
import sys
import time
import subprocess
import urllib.request
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"

def wait_for_server(url, timeout=30):
    print(f"Waiting for Streamlit server at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"Streamlit server is ready ({time.time() - start_time:.1f}s)")
                    return True
        except Exception:
            time.sleep(1)
    return False

def run_capture():
    # 1. Start Streamlit headless
    print("Launching Streamlit application...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        if not wait_for_server(STREAMLIT_URL, timeout=30):
            print("ERROR: Streamlit server failed to start in time.")
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
            return False

        # 2. Launch Playwright with Chrome
        print("Connecting Playwright headless browser...")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 820},
                device_scale_factor=2  # High-DPI Retina capture
            )
            page = context.new_page()

            print(f"Navigating to {STREAMLIT_URL}...")
            page.goto(STREAMLIT_URL, wait_until="networkidle")
            time.sleep(3)

            # Switch to Synthetic Demo Generator to have active, moving, deterministic frames
            print("Configuring Camera Ingestion to Synthetic Demo Generator...")
            synthetic_radio = page.locator("text=Synthetic Demo Generator")
            if synthetic_radio.count() > 0:
                synthetic_radio.first.click()
                time.sleep(2)

            # Screenshot 1: Main UI Overview (Side-by-Side Face / Overview)
            print("Capturing 1: screenshot_main_ui.png...")
            main_path = os.path.join(OUTPUT_DIR, "screenshot_main_ui.png")
            page.screenshot(path=main_path, full_page=False)
            print(f"Saved: {main_path}")

            # Screenshot 2: DIP - Canny Edge Detection & Gradients
            print("Navigating to Module 3 (Edge Detection & Gradients)...")
            select_box = page.locator("div[data-baseweb='select']").first
            if select_box.count() > 0:
                select_box.click()
                time.sleep(1)
                edge_option = page.locator("li:has-text('3. Edge Detection')")
                if edge_option.count() > 0:
                    edge_option.first.click()
                else:
                    page.keyboard.type("Edge Detection")
                    page.keyboard.press("Enter")
                time.sleep(3)

            # Open the mathematical theory expander for academic rigor
            theory_expander = page.locator("text=DIP & CV Mathematical Foundations")
            if theory_expander.count() > 0:
                theory_expander.first.click()
                time.sleep(1)

            print("Capturing 2: screenshot_dip_edges.png...")
            dip_path = os.path.join(OUTPUT_DIR, "screenshot_dip_edges.png")
            page.screenshot(path=dip_path, full_page=False)
            print(f"Saved: {dip_path}")

            # Screenshot 3: CV - Real-Time Motion Detection & Contours
            print("Navigating to Module 7 (Real-Time Motion Detection)...")
            if select_box.count() > 0:
                select_box.click()
                time.sleep(1)
                motion_option = page.locator("li:has-text('7. Real-Time Motion Detection')")
                if motion_option.count() > 0:
                    motion_option.first.click()
                else:
                    page.keyboard.type("Motion Detection")
                    page.keyboard.press("Enter")
                time.sleep(3)

            print("Capturing 3: screenshot_cv_motion.png...")
            cv_path = os.path.join(OUTPUT_DIR, "screenshot_cv_motion.png")
            page.screenshot(path=cv_path, full_page=False)
            print(f"Saved: {cv_path}")

            # Additional Screenshot: Contour & Geometric Shape Analysis
            print("Navigating to Module 9 (Contour & Geometric Shape Analysis)...")
            if select_box.count() > 0:
                select_box.click()
                time.sleep(1)
                contour_option = page.locator("li:has-text('9. Contour')")
                if contour_option.count() > 0:
                    contour_option.first.click()
                else:
                    page.keyboard.type("Contour")
                    page.keyboard.press("Enter")
                time.sleep(3)

            print("Capturing 4: screenshot_cv_contours.png...")
            contour_path = os.path.join(OUTPUT_DIR, "screenshot_cv_contours.png")
            page.screenshot(path=contour_path, full_page=False)
            print(f"Saved: {contour_path}")

            browser.close()
            print("Browser closed successfully.")
            return True

    finally:
        print("Terminating Streamlit background server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit process stopped.")

if __name__ == "__main__":
    success = run_capture()
    sys.exit(0 if success else 1)
