# debug_thermal.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.esp import esp_standard, esp_thermal

print("Debug thermal tool initialized.")
