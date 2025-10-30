# PowerShell script to reproduce paper results
# Output: results/conformal_preds/*.json
# Same structure as archive for plotting compatibility

Write-Host "Reproducing Paper Results" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# 1. Cylinder Flow (medium, with noise) - Comparison of all geometries
Write-Host "1. Cylinder medium with noise - geometry comparison..." -ForegroundColor Yellow
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -c
# Output: *_alpha_sweep_compare.json

Write-Host ""
Write-Host "2. Cylinder medium with noise - adaptive CP (basic features, p=5)..." -ForegroundColor Yellow
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2
# Output: *_boosted_basic_alpha_sweep.json

Write-Host ""
Write-Host "3. Cylinder medium with noise - adaptive CP (full features, p=17)..." -ForegroundColor Yellow
python cli.py cylinder_medium_noise -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2 -f
# Output: *_boosted_enhanced_alpha_sweep.json

Write-Host ""
Write-Host "4. Flag dataset (3D) - geometry comparison..." -ForegroundColor Yellow
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -c
# Output: *_alpha_sweep_compare.json

Write-Host ""
Write-Host "5. Flag dataset - adaptive CP (basic features, p=5)..." -ForegroundColor Yellow
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2
# Output: *_boosted_basic_alpha_sweep.json

Write-Host ""
Write-Host "6. Flag dataset - adaptive CP (full features, p=17)..." -ForegroundColor Yellow
python cli.py flag_medium -s 0.05,0.1,0.15,0.2,0.25,0.3 -ad -x 0.2 -f
# Output: *_boosted_enhanced_alpha_sweep.json

Write-Host ""
Write-Host "=========================" -ForegroundColor Green
Write-Host "✅ All results saved to results/conformal_preds/" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green

