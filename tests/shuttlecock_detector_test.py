"""file tests the detector."""

import os
import sys

from loguru import logger

# 1. THE BRIDGE: Tell Python to look inside the 'src' folder.
# This finds the folder where this script is, goes up one level, and adds 'src'
current_folder = os.path.dirname(__file__)
parent_folder = os.path.abspath(os.path.join(current_folder, "..", "src"))
sys.path.append(parent_folder)

# 2. THE IMPORT: Match your folder structure exactly.
# This means: "Inside the badminton_robot folder, find shuttlecock_detector.py"
try:
    from badminton_robot.shuttlecock_detector import ShuttlecockDetector

    logger.success("✅ System: Successfully linked to ShuttlecockDetector class.")
except ImportError as e:
    logger.error(f"❌ System: Could not find the code. Error: {e}")
    sys.exit(1)


def run_test():
    """Test the shuttlecock detector."""
    # 3. INITIALIZE: Wake up the AI
    detector = ShuttlecockDetector()

    # 4. PATH: The exact location you found with the 'find' command
    image_path = "data/data/test_image.png"

    if not os.path.exists(image_path):
        logger.error(f"❌ Error: Cannot find the image file at: {image_path}")
        return

    logger.info(f"🔍 Analyzing: {image_path}...")

    # 5. RUN: Ask the detector for the X and Y coordinates
    try:
        results = detector.get_location(image_path)

        if results:
            logger.sucess(f"🎯 SUCCESS! Found {len(results)} shuttlecock(s):")
            for i, (x, y) in enumerate(results):
                logger.info(f"   Detection {i + 1}: Pixel X={x}, Pixel Y={y}")
        else:
            logger.warning("❓ The AI finished scanning, but didn't see a shuttlecock.")

    except Exception as e:
        logger.error(f"❌ An error occurred during detection: {e}")


if __name__ == "__main__":
    run_test()
