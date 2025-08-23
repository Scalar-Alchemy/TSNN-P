# Fabrication Protocols for Nanostructured Metasurface System

## 1. Bi-2223 Nanoparticle Synthesis

### Protocol 1.1: Sol-Gel Synthesis

**Materials Required:**
- Bi(NO₃)₃·5H₂O (99.99%, Sigma-Aldrich)
- Sr(NO₃)₂ (99.99%, Sigma-Aldrich)
- Ca(NO₃)₂·4H₂O (99.99%, Sigma-Aldrich)
- Cu(NO₃)₂·3H₂O (99.99%, Sigma-Aldrich)
- Citric acid (99.5%, Sigma-Aldrich)
- Ethylene glycol (99.8%, Sigma-Aldrich)
- Deionized water (18.2 MΩ·cm)

**Equipment:**
- Magnetic stirrer with heating
- pH meter
- Centrifuge (10,000 rpm)
- Tube furnace (1200°C max)
- Oxygen gas cylinder
- Flow controller

**Procedure:**

1. **Solution Preparation (Day 1)**
   ```
   - Dissolve 2.425 g Bi(NO₃)₃·5H₂O in 50 mL DI water
   - Dissolve 1.058 g Sr(NO₃)₂ in 30 mL DI water
   - Dissolve 0.944 g Ca(NO₃)₂·4H₂O in 30 mL DI water
   - Dissolve 1.208 g Cu(NO₃)₂·3H₂O in 30 mL DI water
   - Dissolve 4.2 g citric acid in 50 mL DI water
   ```

2. **Mixing and Complexation (Day 1)**
   ```
   - Combine all nitrate solutions in 500 mL beaker
   - Add citric acid solution dropwise with stirring
   - Adjust pH to 6.5 using NH₄OH
   - Add 10 mL ethylene glycol
   - Stir at 80°C for 2 hours
   ```

3. **Gel Formation (Day 1)**
   ```
   - Continue heating at 80°C until gel forms
   - Dry gel at 120°C for 12 hours
   - Crush dried gel to powder
   ```

4. **Calcination (Day 2)**
   ```
   - Load powder into alumina crucible
   - Place in tube furnace
   - Heat to 850°C at 5°C/min
   - Hold at 850°C for 12 hours in O₂ flow (1 L/min)
   - Cool to room temperature at 2°C/min
   ```

5. **Characterization**
   ```
   - XRD analysis (Cu Kα, 2θ = 10-80°)
   - SEM imaging for size distribution
   - BET surface area measurement
   - Magnetic susceptibility (SQUID)
   ```

### Protocol 1.2: Size Control and Optimization

**For 10-50 nm particles:**
- Add 0.1% PVP during gel formation
- Use ultrasonic agitation during calcination
- Control cooling rate to 1°C/min

**For uniform distribution:**
- Sieve through 200 mesh
- Centrifuge at 3000 rpm for size separation
- Store in desiccator under N₂

## 2. Kagome Lattice Template Fabrication

### Protocol 2.1: AAO Template Preparation

**Materials Required:**
- High-purity aluminum foil (99.999%, 0.5 mm thick)
- Oxalic acid (0.3 M)
- Phosphoric acid (5 wt%)
- Chromic acid (1.8 wt% CrO₃ + 6 wt% H₃PO₄)

**Equipment:**
- DC power supply (0-200 V)
- Platinum counter electrode
- Temperature-controlled bath
- Digital multimeter

**Procedure:**

1. **Aluminum Preparation**
   ```
   - Cut Al foil to 2 cm × 2 cm
   - Anneal at 500°C for 2 hours
   - Electropolish in 1:4 HClO₄:EtOH at 20 V for 2 min
   - Rinse with DI water
   ```

2. **First Anodization**
   ```
   - Immerse in 0.3 M oxalic acid at 0°C
   - Apply 40 V for 4 hours
   - Remove oxide layer in chromic acid (2 hours)
   ```

3. **Second Anodization**
   ```
   - Repeat anodization at 40 V for 2 hours
   - This creates ordered pore array
   ```

4. **Pore Widening**
   ```
   - Immerse in 5 wt% H₃PO₄ at 30°C
   - Etch for 30 minutes
   - Target pore diameter: 200 nm
   ```

5. **Template Characterization**
   ```
   - SEM imaging of pore structure
   - Measure pore diameter and density
   - Verify Kagome-like arrangement
   ```

### Protocol 2.2: Bi-2223 Deposition

**Equipment:**
- CVD system with temperature control
- Precursor delivery system
- Vacuum pump system
- Mass flow controllers

**Procedure:**

1. **Template Loading**
   ```
   - Mount AAO template in CVD chamber
   - Evacuate to 10⁻⁶ Torr
   - Heat to 850°C at 10°C/min
   ```

2. **Precursor Delivery**
   ```
   - Bi-2223 nanoparticles in crucible
   - Heat crucible to 950°C
   - Carrier gas: Ar (50 sccm)
   - Deposition time: 2 hours
   ```

3. **Post-Deposition Annealing**
   ```
   - Maintain 850°C for 1 hour in O₂
   - Cool to room temperature
   - Remove from template
   ```

## 3. Graphene-BIC Metasurface Fabrication

### Protocol 3.1: Graphene Growth

**Materials Required:**
- Copper foil (99.8%, 25 μm thick)
- Methane (99.99%)
- Hydrogen (99.99%)
- Argon (99.99%)

**Equipment:**
- CVD furnace (1000°C)
- Gas delivery system
- Optical microscope
- Raman spectrometer

**Procedure:**

1. **Copper Preparation**
   ```
   - Clean Cu foil with acetone, IPA, DI water
   - Anneal at 1000°C in H₂ (50 sccm) for 30 min
   - Cool to 1000°C
   ```

2. **Graphene Growth**
   ```
   - Introduce CH₄ (1 sccm) + H₂ (50 sccm)
   - Growth time: 30 minutes
   - Cool to room temperature in H₂
   ```

3. **Transfer to SiO₂/Si**
   ```
   - Spin-coat PMMA (950K, 4% in anisole)
   - Etch Cu in FeCl₃ solution
   - Transfer to SiO₂/Si substrate
   - Remove PMMA with acetone
   ```

### Protocol 3.2: Nanohole Patterning

**Equipment:**
- Electron beam lithography system
- Reactive ion etcher
- Atomic force microscope

**Procedure:**

1. **Resist Coating**
   ```
   - Spin-coat PMMA (950K, 2% in anisole)
   - Bake at 180°C for 2 minutes
   - Thickness: 100 nm
   ```

2. **EBL Patterning**
   ```
   - Write nanohole array pattern
   - Hole diameter: 100 nm
   - Periodicity: 250 nm
   - Dose: 300 μC/cm²
   ```

3. **Development**
   ```
   - Develop in 1:3 MIBK:IPA for 30 seconds
   - Rinse in IPA
   - Dry with N₂
   ```

4. **Etching**
   ```
   - RIE with O₂ plasma
   - Power: 50 W
   - Pressure: 10 mTorr
   - Time: 30 seconds
   ```

5. **Resist Removal**
   ```
   - Remove PMMA with acetone
   - Clean with IPA and DI water
   ```

## 4. System Assembly

### Protocol 4.1: Component Integration

**Materials Required:**
- Sapphire substrate (10 mm × 10 mm × 0.5 mm)
- Indium wire (99.99%, 0.5 mm diameter)
- Silver epoxy (Epo-Tek H20E)
- Kapton tape

**Procedure:**

1. **Substrate Preparation**
   ```
   - Clean sapphire with acetone, IPA, DI water
   - Dry with N₂
   - Mount on sample holder
   ```

2. **Graphene-BIC Mounting**
   ```
   - Place graphene-BIC on sapphire
   - Secure with Kapton tape
   - Verify electrical contact
   ```

3. **Bi-2223 Kagome Lattice Transfer**
   ```
   - Transfer lattice onto graphene
   - Align with nanohole array
   - Secure with silver epoxy
   ```

4. **Electrical Contacts**
   ```
   - Bond indium wires to graphene
   - Bond indium wires to Bi-2223
   - Test contact resistance (<1 mΩ)
   ```

### Protocol 4.2: Helmholtz Coil Assembly

**Materials Required:**
- Copper wire (AWG 18, 100 m)
- Coil formers (10 cm radius)
- Epoxy resin
- Liquid nitrogen Dewar

**Procedure:**

1. **Coil Winding**
   ```
   - Wind 100 turns on each former
   - Secure with epoxy
   - Measure resistance (0.5 Ω each)
   ```

2. **Coil Alignment**
   ```
   - Mount coils coaxially
   - Separation: 10 cm
   - Align with optical alignment
   ```

3. **Field Calibration**
   ```
   - Use Hall probe array
   - Map field uniformity
   - Adjust alignment if needed
   ```

## 5. Quality Control and Testing

### Protocol 5.1: Material Characterization

**XRD Analysis:**
```
- Cu Kα radiation (λ = 1.5406 Å)
- 2θ range: 10-80°
- Step size: 0.02°
- Scan rate: 2°/min
```

**SEM Imaging:**
```
- Acceleration voltage: 5-15 kV
- Working distance: 10 mm
- Magnification: 10k-100k×
- EDS for composition
```

**Raman Spectroscopy:**
```
- Laser wavelength: 532 nm
- Power: 1 mW
- Integration time: 10 s
- Measure I_D/I_G ratio
```

### Protocol 5.2: Performance Testing

**Hydrogen Absorption:**
```
- Load sample in Sieverts apparatus
- Expose to H₂ at 1-10 bar
- Monitor pressure change
- Calculate absorption capacity
```

**Magnetic Field Measurement:**
```
- Use Hall probe array
- Map field in 5 cm³ volume
- Verify <1% uniformity
- Calibrate with NMR probe
```

**BIC Resonance Testing:**
```
- Use tunable laser (1500-1600 nm)
- Measure transmission spectrum
- Determine Q-factor
- Map field enhancement
```

## 6. Safety Protocols

### Protocol 6.1: Hydrogen Handling

**Before Experiment:**
```
- Check all gas lines for leaks
- Verify ventilation system
- Calibrate hydrogen sensors
- Prepare emergency shutdown
```

**During Experiment:**
```
- Monitor hydrogen concentration
- Keep below 1% LEL
- Have fire extinguisher ready
- Maintain communication
```

**After Experiment:**
```
- Purge with N₂ for 30 minutes
- Ventilate room for 1 hour
- Check for residual H₂
- Document all procedures
```

### Protocol 6.2: Radiation Safety

**Detection Setup:**
```
- Calibrate neutron detectors
- Set up gamma detectors
- Establish baseline measurements
- Prepare shielding
```

**Monitoring:**
```
- Continuous monitoring during runs
- Record all radiation events
- Maintain dosimeter readings
- Follow ALARA principles
```

**Emergency Response:**
```
- Immediate shutdown if >1 mSv/h
- Evacuate personnel
- Contact radiation safety officer
- Document incident
```

## 7. Documentation and Record Keeping

### Protocol 7.1: Data Management

**Fabrication Records:**
```
- Document all process parameters
- Record material batch numbers
- Note any deviations from protocol
- Maintain calibration records
```

**Performance Data:**
```
- Store all measurement data
- Include uncertainty analysis
- Cross-reference with predictions
- Archive for future reference
```

**Quality Assurance:**
```
- Regular equipment calibration
- Periodic performance reviews
- Update protocols as needed
- Maintain training records
``` 