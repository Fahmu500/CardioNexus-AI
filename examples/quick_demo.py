"""
CardioNexus AI Suite - Quick Demo
=================================
Demonstrates the novel risk predictors without requiring GPU or full dependencies.
Run: python examples/quick_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_second_heart():
    """Demonstrate the Second Heart (Calf Muscle Pump) predictor"""
    print("=" * 60)
    print("DEMO: Second Heart (Calf Muscle Pump) Risk Predictor")
    print("=" * 60)
    
    # Simulated patient data
    patients = [
        {
            "name": "Patient A - Active, Healthy",
            "data": {
                "calf_circumference_cm": 38,
                "ankle_brachial_index": 1.1,
                "venous_refill_time_sec": 25,
                "steps_per_day": 10000,
                "seated_hours_per_day": 4,
                "varicose_veins": False,
                "dvt_history": False,
                "peripheral_neuropathy": False,
            }
        },
        {
            "name": "Patient B - Sedentary, High Risk",
            "data": {
                "calf_circumference_cm": 30,
                "ankle_brachial_index": 0.7,
                "venous_refill_time_sec": 8,
                "steps_per_day": 2000,
                "seated_hours_per_day": 12,
                "varicose_veins": True,
                "dvt_history": True,
                "peripheral_neuropathy": True,
            }
        }
    ]
    
    for p in patients:
        d = p["data"]
        # Rule-based risk calculation (mirrors the ML model logic)
        abi_risk = max(0, 1 - d["ankle_brachial_index"]) if d["ankle_brachial_index"] < 0.9 else 0
        refill_risk = max(0, 1 - d["venous_refill_time_sec"] / 25)
        sedentary_risk = min(d["seated_hours_per_day"] / 14, 1.0)
        activity_protection = min(d["steps_per_day"] / 10000, 1.0)
        
        comorbidity = sum([d["varicose_veins"], d["dvt_history"], d["peripheral_neuropathy"]]) / 3
        
        risk = (0.25 * abi_risk + 0.25 * refill_risk + 0.2 * sedentary_risk + 0.15 * comorbidity) * (1 - 0.3 * activity_protection)
        
        print(f"\n  {p['name']}")
        print(f"  Calf Pump Risk Score: {risk:.2%}")
        print(f"  Risk Level: {'HIGH' if risk > 0.5 else 'MODERATE' if risk > 0.25 else 'LOW'}")
        print(f"  ABI Risk: {abi_risk:.2f} | Refill Risk: {refill_risk:.2f}")
        print(f"  Sedentary Risk: {sedentary_risk:.2f} | Activity Protection: {activity_protection:.2f}")


def demo_gut_heart():
    """Demonstrate the Gut-Heart Axis predictor"""
    print("\n" + "=" * 60)
    print("DEMO: Gut-Heart Axis (Microbiome-Cardiac) Risk Predictor")
    print("=" * 60)
    
    patients = [
        {
            "name": "Patient A - Healthy Microbiome",
            "data": {
                "tmao_umol": 3.2,
                "f_prausnitzii_pct": 8.5,
                "carnitine_mg_day": 100,
                "choline_mg_day": 200,
                "fiber_g_day": 35,
                "red_meat_servings_week": 1,
                "probiotic_use": True,
            }
        },
        {
            "name": "Patient B - Dysbiotic, High TMAO",
            "data": {
                "tmao_umol": 12.8,
                "f_prausnitzii_pct": 1.2,
                "carnitine_mg_day": 450,
                "choline_mg_day": 600,
                "fiber_g_day": 10,
                "red_meat_servings_week": 7,
                "probiotic_use": False,
            }
        }
    ]
    
    for p in patients:
        d = p["data"]
        # TMAO risk (>6.2 umol is high risk per Cleveland Clinic research)
        tmao_risk = min(d["tmao_umol"] / 12, 1.0)
        
        # F. prausnitzii is cardioprotective (higher = better)
        fp_protection = min(d["f_prausnitzii_pct"] / 10, 1.0)
        
        # Dietary substrate risk (carnitine + choline feed TMAO-producing bacteria)
        substrate_risk = min((d["carnitine_mg_day"] + d["choline_mg_day"]) / 800, 1.0)
        
        # Protective factors
        fiber_protection = min(d["fiber_g_day"] / 30, 1.0)
        
        risk = (0.4 * tmao_risk + 0.2 * substrate_risk) * (1 - 0.3 * fp_protection - 0.1 * fiber_protection)
        risk = max(0, min(risk, 1.0))
        
        print(f"\n  {p['name']}")
        print(f"  Gut-Heart Risk Score: {risk:.2%}")
        print(f"  Risk Level: {'HIGH' if risk > 0.4 else 'MODERATE' if risk > 0.2 else 'LOW'}")
        print(f"  TMAO: {d['tmao_umol']} umol/L ({'ELEVATED' if d['tmao_umol'] > 6.2 else 'Normal'})")
        print(f"  F. prausnitzii: {d['f_prausnitzii_pct']}% ({'Low' if d['f_prausnitzii_pct'] < 3 else 'Healthy'})")
        print(f"  Dietary Substrate Load: {substrate_risk:.2f}")


if __name__ == "__main__":
    print()
    print("  CardioNexus AI Suite - Community Edition Demo")
    print("  Demonstrating Novel Cardiac Risk Predictors")
    print()
    
    demo_second_heart()
    demo_gut_heart()
    
    print("\n" + "=" * 60)
    print("  These novel risk pathways are computationally modeled")
    print("  in the full CardioNexus AI engine with ML models.")
    print("  See cardionexus_core.py for the complete implementation.")
    print("=" * 60)
    print()
