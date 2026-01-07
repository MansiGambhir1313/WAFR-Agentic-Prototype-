#!/bin/bash
# Complete WAFR Pipeline Runner - Linux/Mac Shell Script
# Includes all latest features: HRI validation, strict quality control, AG-UI, lens detection

echo "======================================================================"
echo "WAFR Complete Pipeline Runner"
echo "======================================================================"
echo ""

# Check if client name is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_wafr_full.sh \"Client Name\""
    echo ""
    echo "Example: ./run_wafr_full.sh \"My Company\""
    echo ""
    echo "This will run the complete WAFR pipeline with:"
    echo "  - AG-UI event streaming"
    echo "  - Strict quality control (confidence >= 0.7)"
    echo "  - HRI validation (filters non-tangible HRIs)"
    echo "  - Automatic lens detection"
    echo "  - WA Tool integration and PDF generation"
    echo ""
    exit 1
fi

CLIENT_NAME="$1"

echo "Running WAFR pipeline for: $CLIENT_NAME"
echo ""
echo "Features enabled:"
echo "  - AG-UI Integration"
echo "  - Strict Quality Control (confidence >= 0.7)"
echo "  - HRI Validation (Claude-based)"
echo "  - Automatic Lens Detection"
echo "  - WA Tool Integration"
echo "  - PDF Report Generation"
echo ""
echo "Starting pipeline..."
echo ""

python run_wafr_full.py --wa-tool --client-name "$CLIENT_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Pipeline completed successfully!"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "Pipeline completed with errors. Check logs for details."
    echo "======================================================================"
    exit 1
fi
