#!/bin/bash

# Run the download commands
echo "Starting download..."

cd src
s5cmd --no-sign-request run scripts/Data\ fetcher/download_5percent.s5cmd

echo "Download complete. Data saved to data/ directory."