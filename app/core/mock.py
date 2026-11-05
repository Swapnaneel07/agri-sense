"""
AgriSense AI - Mock Fallback Engine
Provides realistic, domain-accurate fallback predictions tagged with mock=True.
Guarantees the dashboard functions end-to-end even before any teammate delivers model weights.
"""

import time
import numpy as np
from PIL import Image, ImageDraw


# FAO typical crop yields in hectograms per hectare (hg/ha)
CROP_BASE_YIELDS_HG_HA = {
    "Potatoes": 220000.0,
    "Sweet potatoes": 145000.0,
    "Cassava": 115000.0,
    "Plantains and others": 105000.0,
    "Yams": 92000.0,
    "Maize": 44000.0,
    "Rice, paddy": 41000.0,
    "Wheat": 33000.0,
    "Soybeans": 22000.0,
    "Sorghum": 14000.0
}


def mock_predict_yield(feature_dict):
    """
    Simulates agronomic crop yield response based on FAO indicators:
    Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes.
    Returns: yield in hg/ha (and notes).
    """
    crop = str(feature_dict.get("Item", "Maize"))
    area = str(feature_dict.get("Area", "India"))
    year = int(feature_dict.get("Year", 2013))
    rain = float(feature_dict.get("average_rain_fall_mm_per_year", 1120.0))
    pesticides = float(feature_dict.get("pesticides_tonnes", 1200.0))

    base_yield = CROP_BASE_YIELDS_HG_HA.get(crop, 40000.0)

    # Climate & technological trend modifiers
    year_factor = 1.0 + (year - 1990) * 0.008  # ~0.8% annual productivity growth
    rain_factor = 0.85 + min(rain / 2000.0, 0.35)
    pest_factor = 0.95 + min(pesticides / 50000.0, 0.15)

    predicted_hg_ha = round(base_yield * year_factor * rain_factor * pest_factor, 1)

    return {
        "predicted_yield": predicted_hg_ha,
        "units": "hg/ha",
        "mock": True,
        "model_name": "Calibrated FAO Empirical Regressor",
        "advisory": f"Estimated yield for {crop} in {area} ({year}) under {rain} mm rainfall."
    }


def mock_predict_disease(pil_image):
    """
    Simulates MobileNetV2 leaf disease diagnosis with latency and class probabilities.
    """
    start_t = time.perf_counter()
    img = pil_image.convert("RGB").resize((224, 224))
    np_img = np.array(img, dtype=np.float32) / 255.0

    r = float(np.mean(np_img[:, :, 0]))
    g = float(np.mean(np_img[:, :, 1]))
    b = float(np.mean(np_img[:, :, 2]))

    # If green is dominant -> healthy; if reddish/brown -> necrotic blight
    if g > r + 0.12:
        top_class = "Apple___healthy"
        prob = 0.942
        cat = "Healthy Foliage"
        second = ("Tomato___healthy", 0.038)
        third = ("Grape___healthy", 0.012)
    elif r > g:
        top_class = "Potato___Late_blight"
        prob = 0.887
        cat = "Fungal / Oomycete"
        second = ("Tomato___Early_blight", 0.072)
        third = ("Apple___Apple_scab", 0.024)
    else:
        top_class = "Tomato___Early_blight"
        prob = 0.795
        cat = "Fungal Foliar"
        second = ("Tomato___Late_blight", 0.124)
        third = ("Bell_Pepper___Bacterial_spot", 0.051)

    latency_ms = round((time.perf_counter() - start_t) * 1000 + 18.2, 1)

    return {
        "primary_class": top_class,
        "confidence": round(prob * 100.0, 1),
        "category": cat,
        "is_healthy": "healthy" in top_class.lower(),
        "top_3": [
            {"class": top_class, "confidence": round(prob * 100.0, 1)},
            {"class": second[0], "confidence": round(second[1] * 100.0, 1)},
            {"class": third[0], "confidence": round(third[1] * 100.0, 1)}
        ],
        "latency_ms": latency_ms,
        "mock": True
    }


def mock_predict_irrigation(readings_array):
    """
    Simulates sequential soil moisture forecast (T+10 min) and recommendation.
    """
    arr = np.asarray(readings_array, dtype=np.float32)
    last_moisture = float(arr[-1, 0]) if arr.ndim > 1 and arr.shape[1] > 0 else 28.5
    forecast = round(max(5.0, min(95.0, last_moisture - 0.25)), 2)

    if forecast < 20:
        advice = "🚨 Very low moisture (<20%). Irrigation required immediately."
    elif forecast < 35:
        advice = "⚠️ Low moisture (20-35%). Monitor soil and schedule drip cycle."
    elif forecast < 70:
        advice = "✅ Optimal moisture (35-70%). No irrigation needed."
    else:
        advice = "💧 High moisture (>70%). Over-saturation risk detected."

    return {
        "predicted_moisture": forecast,
        "horizon": "10 minutes",
        "advice": advice,
        "winner_model": "BiLSTM (Academic Benchmark)",
        "mock": True
    }


def mock_generate_synthetic(class_name, num_samples=4):
    """
    Generates procedural synthetic leaf pathology images for Module 5 showcase.
    """
    images = []
    for i in range(num_samples):
        img = Image.new("RGB", (128, 128), color=(35, 120 + (i * 15) % 30, 35))
        draw = ImageDraw.Draw(img)
        draw.ellipse([(14, 10), (114, 118)], fill=(45, 138 + (i * 9) % 25, 42))
        draw.line([(64, 15), (64, 115)], fill=(65, 175, 60), width=2)
        # Synthetic lesions
        np.random.seed(101 + i * 23)
        for _ in range(5):
            lx = np.random.randint(35, 95)
            ly = np.random.randint(30, 95)
            rad = np.random.randint(5, 12)
            draw.ellipse([(lx - rad - 2, ly - rad - 2), (lx + rad + 2, ly + rad + 2)], fill=(185, 165, 40))
            draw.ellipse([(lx - rad, ly - rad), (lx + rad, ly + rad)], fill=(90, 45, 20))
        images.append(img.resize((180, 180), Image.Resampling.LANCZOS))
    return images
