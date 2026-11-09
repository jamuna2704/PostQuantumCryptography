# Lorenz + Blum-Goldwasser Image Encryption

## Overview
An experimental image encryption and decryption system combining the 2D Lorenz map for pixel scrambling with the Blum-Goldwasser (BG) cryptosystem for encryption.

The project implements image preprocessing, pixel permutation, cryptographic encryption, decryption, and statistical analysis to evaluate encryption performance.

## Key Features
- Blum-Goldwasser public-key encryption and decryption.
- 2D Lorenz-based pixel scrambling and inverse scrambling.
- Native image resolution processing without resizing.
- RGB image conversion and BMP handling.
- SHA-256 hash verification.
- Image entropy and correlation analysis.
- NPCR and UACI sensitivity analysis.
- Automated unit tests and validation.
- Graphical visualizations and detailed experiment reports.

## Technologies Used
- Python
- NumPy
- Pillow
- Matplotlib
- Cryptographic number theory

## Project Structure
```text
Project/
│
├── bg_image_native.py
├── bg_analysis.py
├── bg_report.py
├── bg_visuals.py
├── requirements.txt
├── README.md
│
├── original/
├── input/
├── scrambled/
├── encrypted/
├── decrypted/
├── descrambled/
├── reports/
└── visualizations/
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Run

1. Place supported images (PNG, JPG, JPEG, BMP, TIFF, or WEBP) inside the `original` folder.
2. Execute:

```bash
python bg_image_native.py
```

3. The program performs encryption and decryption.
4. Results, reports, and visualizations are generated automatically.

To run only unit tests:

```bash
python bg_image_native.py --test-only
```

To generate shareable reports without exposing secret key values:

```bash
python bg_image_native.py --hide-secrets
```

## Experimental Notes
The implementation uses a default BG block size of 512 bits. This is an experimental configuration and exceeds the block-size bound associated with the standard BG security proof.

The Lorenz map provides pixel permutation and is not the source of cryptographic security.

This is an academic and experimental implementation, not a production-ready encryption system.

**Security classification:** Blum-Goldwasser is a classical cryptosystem and is not post-quantum secure.

## Author
Jamuna N

## License
For academic and educational purposes.
