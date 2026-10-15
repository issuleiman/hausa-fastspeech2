#!/usr/bin/env bash
# Download the pretrained HiFi-GAN generator (universal) released with ming024/FastSpeech2 (MIT).
# It was trained at 22.05 kHz / hop 256 / 80 mels / fmax 8000, which matches configs/*.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vocoder_assets
curl -L --fail -o vocoder_assets/generator_universal.pth.tar.zip \
  https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip
echo "saved vocoder_assets/generator_universal.pth.tar.zip"
