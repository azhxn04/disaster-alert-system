import os
import sys
import unittest
import pandas as pd
import numpy as np

import data_engine
import dl_engine
import security

class TestDisasterPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_engine.init_database()

    def test_01_database_population(self):
        """Test real SQLite database exists and is populated across 36 districts."""
        db_path = os.path.join(os.path.dirname(os.path.abspath(data_engine.__file__)), "disaster_data.db")
        self.assertTrue(os.path.exists(db_path), "disaster_data.db must exist on disk.")

        df = data_engine.get_historical_records()
        self.assertFalse(df.empty, "Database must not be empty.")
        self.assertEqual(df['district'].nunique(), 36, "Must cover all 36 Maharashtra districts.")

    def test_02_cyclone_and_earthquake_historical_coverage(self):
        """Test that inland districts (Pune) have realistic non-zero historical data for Cyclone and Earthquake."""
        df_cyc = data_engine.get_district_loss_by_year("Pune", "Cyclone")
        self.assertGreater(float(df_cyc['economic_loss_cr'].sum()), 0.0, "Pune must have recorded peripheral cyclone damage.")
        
        df_eq = data_engine.get_district_loss_by_year("Pune", "Earthquake")
        self.assertGreater(float(df_eq['economic_loss_cr'].sum()), 0.0, "Pune must have recorded micro-seismic damage.")

        dates, r_pune, lb, ub, _ = dl_engine.get_xgboost_forecast("Pune", "Cyclone", "Next 24 Hours")
        self.assertGreater(max(r_pune), 0.0, "Cyclone 24H trajectory in Pune must not be flat zero.")
        self.assertNotEqual(min(r_pune), max(r_pune), "Prediction trajectory must fluctuate dynamically.")

        dates, r_mumbai, _, _, _ = dl_engine.get_xgboost_forecast("Mumbai City", "Cyclone", "6 Months")
        for dt, val in zip(dates, r_mumbai):
            if dt.month == 2:
                self.assertEqual(val, 0.0, "February coastal cyclone risk must be 0.0%.")

    def test_03_historical_loss_and_deaths_categorical_axis(self):
        """Test historical loss and deaths bar chart return all years 2021-2026 as categorical strings."""
        df_loss = data_engine.get_district_loss_by_year("Pune", "Flood")
        self.assertEqual(len(df_loss), 6, "Must contain all 6 years (2021-2026).")
        self.assertTrue(all(isinstance(y, str) for y in df_loss['year']), "Years must be formatted as strings.")
        self.assertIn('economic_loss_cr', df_loss.columns)

        df_deaths = data_engine.get_district_deaths_by_year("Pune", "Flood")
        self.assertEqual(len(df_deaths), 6, "Deaths data must contain all 6 years.")
        self.assertIn('fatalities', df_deaths.columns)

    def test_04_drought_october_down_and_april_up(self):
        """Test drought risk goes down in October (post-monsoon dams full) and goes up in April (scorching summer)."""
        # 1 Year horizon covering all seasons
        dates, r_drought = data_engine.get_seasonal_risk_trajectory("Beed", "Drought", "1 Year")
        month_to_risk = {dt.month: val for dt, val in zip(dates, r_drought)}

        if 10 in month_to_risk and 4 in month_to_risk:
            self.assertLess(month_to_risk[10], 5.0, "October drought risk should be at or near minimum (~1%).")
            self.assertGreater(month_to_risk[4], 50.0, "April drought risk should be elevated (>50%).")
            self.assertGreater(month_to_risk[4], month_to_risk[10], "April drought risk must exceed October drought risk.")

    def test_05_heatwave_winter_zero_risk(self):
        """Test heatwave risk is strictly 0.0% in winter and autumn (Oct, Nov, Dec, Jan, Feb)."""
        dates, r_heat = data_engine.get_seasonal_risk_trajectory("Nagpur", "Heatwave", "1 Year")
        for dt, val in zip(dates, r_heat):
            if dt.month in [10, 11, 12, 1, 2]:
                self.assertEqual(val, 0.0, f"Month {dt.month} heatwave risk in Nagpur must be strictly 0.0%.")

    def test_06_dsp_vault_security_and_hashing(self):
        """Test PBKDF2-HMAC-SHA256 salted hashing and AES-256 encrypted storage in dsp_vault."""
        test_user = "test_officer_unit"
        test_pass = "TestSecretKey2026!"
        registered = security.register_user(test_user, test_pass, "Field Officer")
        self.assertTrue(registered or security.update_user_password(test_user, test_pass))

        auth_ok, role = security.verify_user(test_user, test_pass)
        self.assertTrue(auth_ok, "Valid password must authenticate against encrypted vault.")
        self.assertEqual(role, "Field Officer")

        auth_fail, _ = security.verify_user(test_user, "WrongPassword123")
        self.assertFalse(auth_fail, "Invalid password must fail verification.")

        # Test pipeline diagnostic
        pipeline = security.explain_dsp_pipeline("MyPass")
        self.assertEqual(pipeline["iterations"], 100000)
        self.assertIn("PBKDF2-HMAC-SHA256", pipeline["hash_algorithm"])
        self.assertTrue(os.path.exists(security.ENCRYPTED_VAULT_FILE), "Encrypted vault file must exist on disk.")

    def test_07_live_api_telemetry_and_fallback(self):
        """Test live Open-Meteo REST API telemetry and local database fallback."""
        temp, hum, aqi, seismic, src = data_engine.fetch_live_weather_aqi("Pune")
        self.assertIsInstance(temp, (int, float))
        self.assertIsInstance(hum, (int, float))
        self.assertIsInstance(aqi, (int, float))
        self.assertIsInstance(seismic, (int, float))
        self.assertIn(src, ["Live Telemetry (Open-Meteo API)", "Cached Offline Telemetry (Local DB)", "Local Baseline (Default)"])

    def test_08_pytorch_dl_engine(self):
        """Test PyTorch DL engine trains with SQLite historical data."""
        model, scaler, acc, f1, df_hist = dl_engine.train_dl_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)
        self.assertGreater(acc, 0.9)
        self.assertGreater(f1, 0.9)
        self.assertFalse(df_hist.empty)

    def test_09_xgboost_predictive_engine(self):
        """Test XGBoost risk regressor generates dynamic forecasts with confidence bounds."""
        dates, risks, lb, ub, metrics = dl_engine.get_xgboost_forecast("Pune", "Cyclone", "Next 24 Hours")
        self.assertEqual(len(dates), 24)
        self.assertEqual(len(risks), 24)
        self.assertEqual(len(lb), 24)
        self.assertEqual(len(ub), 24)
        self.assertTrue(all(l <= r <= u for l, r, u in zip(lb, risks, ub)), "Risk must lie within confidence bounds.")
        self.assertIn("r2_score", metrics)
        self.assertGreater(metrics["r2_score"], 0.9)

if __name__ == "__main__":
    unittest.main()
