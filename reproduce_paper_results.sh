#!/bin/bash
# Commands to reproduce paper results from archive/conformal/results/
# 
# Output: results/conformal_preds/*.json
# Same structure as archive for plotting compatibility

echo "Reproducing Paper Results"
echo "========================="
echo ""

# 1. Cylinder Flow (medium, with noise) - Comparison of all geometries
echo "1. Cylinder medium with noise - geometry comparison..."
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -c
# Output: *_alpha_sweep_compare.json

echo ""
echo "2. Cylinder medium with noise - adaptive CP (basic features, p=5)..."
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2
# Output: *_boosted_basic_alpha_sweep.json

echo ""
echo "3. Cylinder medium with noise - adaptive CP (full features, p=17)..."
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2 -f
# Output: *_boosted_enhanced_alpha_sweep.json

echo ""
echo "4. Flag dataset (3D) - geometry comparison..."
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -c
# Output: *_alpha_sweep_compare.json

echo ""
echo "5. Flag dataset - adaptive CP (basic features, p=5)..."
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2
# Output: *_boosted_basic_alpha_sweep.json

echo ""
echo "6. Flag dataset - adaptive CP (full features, p=17)..."
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2 -f
# Output: *_boosted_enhanced_alpha_sweep.json

echo ""
echo "========================="
echo "✅ All results saved to results/conformal_preds/"
echo "========================="

