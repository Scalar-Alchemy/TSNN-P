# Nanostructured Metasurface Simulation for Hydrogen Absorption and LENR

## Overview
This program simulates a nanostructured metasurface system for enhanced hydrogen absorption and low-energy nuclear reactions (LENR) on an NVIDIA Jetson platform. It models Bi-2223 Kagome lattice adsorption, Helmholtz coil magnetic fields, graphene-BIC metasurface resonances, and LENR probability.

## Requirements
- NVIDIA Jetson with JetPack r36.4
- Docker container: `dustynv/pytorch:r36.4.0`
- Python 3.8+
- Dependencies: See `requirements.txt`

## Installation
1. Pull the Jetson container:
   ```bash
   docker pull dustynv/pytorch:r36.4.0
