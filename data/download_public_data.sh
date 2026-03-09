#!/bin/bash

usage() {
    echo "Usage: $0 --json | --xlsx"
    echo "  --json   Download JSON data file"
    echo "  --xlsx   Download XLSX data file"
    exit 1
}

if [[ $# -eq 0 ]]; then
    usage
fi

case "$1" in
    --json)
        echo "Downloading JSON file..."
        curl -o export-full-en.json https://dataex.ohchr.org/uhri/export-results/export-full-en.json
        echo "JSON file downloaded successfully."
        ;;
    --xlsx)
        echo "Downloading XLSX file..."
        curl -o export-full-en.xlsx https://dataex.ohchr.org/uhri/export-results/export-full-en.xlsx
        echo "XLSX file downloaded successfully."
        ;;
    *)
        echo "Error: Invalid option '$1'"
        usage
        ;;
esac