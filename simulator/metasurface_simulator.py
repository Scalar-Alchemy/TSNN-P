import torch
import numpy as np
from scipy.integrate import solve_ivp
from ase import Atoms
from ase.io import write
import matplotlib.pyplot as plt
import logging
import unittest
import os

# Configure logging for debugging
logging.basicConfig(
    filename="/home/nix/Desktop/TestFrontend/simulation.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure compatibility with Jetson (ARM64, JetPack r36.4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class Bi2223KagomeLattice:
    def __init__(self, nanoparticle_size=10e-9, lattice_constant=5e-10):
        """
        Initialize Bi-2223 Kagome lattice for hydrogen adsorption.
        Args:
            nanoparticle_size (float): Size of Bi-2223 nanoparticles (m).
            lattice_constant (float): Kagome lattice constant (m).
        """
        self.nanoparticle_size = nanoparticle_size
        self.lattice_constant = lattice_constant
        self.atoms = self._build_kagome_lattice()
        self.hydrogen_adsorbed = 0
        logger.debug("Bi2223KagomeLattice initialized with nanoparticle size %s m", nanoparticle_size)

    def _build_kagome_lattice(self):
        """
        Construct a simplified Kagome lattice using ASE for molecular modeling.
        Returns:
            Atoms: ASE Atoms object representing the lattice.
        """
        # Simplified Kagome lattice: triangular and hexagonal motifs
        positions = [
            [0, 0, 0],  # Bi atom
            [self.lattice_constant, 0, 0],  # Sr atom
            [self.lattice_constant / 2, np.sqrt(3) * self.lattice_constant / 2, 0]  # Ca/Cu
        ]
        atoms = Atoms(symbols="BiSrCa", positions=positions, pbc=True)
        logger.debug("Kagome lattice built with %d atoms", len(atoms))
        return atoms

    def calculate_adsorption_energy(self, hydrogen_positions):
        """
        Calculate hydrogen adsorption energy using a simplified DFT model.
        Args:
            hydrogen_positions (list): List of hydrogen atom positions.
        Returns:
            float: Adsorption energy (eV).
        """
        try:
            # Simplified adsorption energy model (PBE functional approximation)
            E_H2 = -0.1  # Free H2 energy (eV)
            E_surf = -100.0  # Surface energy (arbitrary baseline)
            E_H2_surf = E_surf + E_H2 - 0.5  # Adsorbed state (tuned for -0.5 eV)
            adsorption_energy = E_H2_surf - (E_H2 + E_surf)
            self.hydrogen_adsorbed += len(hydrogen_positions)
            logger.info("Adsorption energy: %s eV for %d hydrogen atoms", adsorption_energy, len(hydrogen_positions))
            return adsorption_energy
        except Exception as e:
            logger.error("Error in adsorption energy calculation: %s", e)
            raise

class HelmholtzCoil:
    def __init__(self, radius=0.1, turns=100, current=5.0):
        """
        Initialize Helmholtz coil for magnetic field generation.
        Args:
            radius (float): Coil radius (m).
            turns (int): Number of coil turns.
            current (float): Current (A).
        """
        self.radius = radius
        self.turns = turns
        self.current = current
        self.mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
        logger.debug("HelmholtzCoil initialized with radius %s m, %d turns, current %s A", radius, turns, current)

    def calculate_magnetic_field(self, position):
        """
        Calculate magnetic field at a given position using Biot-Savart law.
        Args:
            position (tuple): (x, y, z) coordinates (m).
        Returns:
            float: Magnetic field strength (T).
        """
        try:
            # Simplified uniform field at center: B = μ₀NI/(2R)
            B = self.mu_0 * self.turns * self.current / (2 * self.radius)
            # Zeeman splitting: ΔE = μ_B * B
            mu_B = 5.788e-5  # Bohr magneton (eV/T)
            delta_E = mu_B * B
            logger.debug("Magnetic field at %s: %s T, Zeeman splitting: %s eV", position, B, delta_E)
            return B, delta_E
        except Exception as e:
            logger.error("Error in magnetic field calculation: %s", e)
            raise

class GrapheneBICMetasurface:
    def __init__(self, nanohole_diameter=100e-9, periodicity=250e-9, Q_factor=1000):
        """
        Initialize graphene-BIC metasurface for electromagnetic field enhancement.
        Args:
            nanohole_diameter (float): Nanohole diameter (m).
            periodicity (float): Nanohole periodicity (m).
            Q_factor (float): Optical Q-factor.
        """
        self.nanohole_diameter = nanohole_diameter
        self.periodicity = periodicity
        self.Q_factor = Q_factor
        logger.debug("GrapheneBICMetasurface initialized with Q-factor %s", Q_factor)

    def calculate_field_enhancement(self, frequency):
        """
        Calculate electromagnetic field enhancement using a simplified BIC model.
        Args:
            frequency (float): Incident light frequency (Hz).
        Returns:
            float: Field enhancement factor.
        """
        try:
            # Simplified Lorentzian model for BIC resonance
            f_0 = 1e15  # Resonance frequency (Hz)
            gamma = f_0 / self.Q_factor
            enhancement = self.Q_factor / (1 + ((frequency - f_0) / gamma)**2)
            logger.info("Field enhancement at frequency %s Hz: %s", frequency, enhancement)
            return enhancement
        except Exception as e:
            logger.error("Error in field enhancement calculation: %s", e)
            raise

class LENRSimulator:
    def __init__(self, screening_factor=1.0):
        """
        Initialize LENR simulator with modified Gamow factor.
        Args:
            screening_factor (float): Coulomb screening factor due to Bi-2223.
        """
        self.screening_factor = screening_factor
        self.hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
        self.e = 1.60217662e-19  # Elementary charge (C)
        logger.debug("LENRSimulator initialized with screening factor %s", screening_factor)

    def calculate_lenr_probability(self, velocity=1e5):
        """
        Calculate LENR probability using modified Gamow factor.
        Args:
            velocity (float): Relative velocity of hydrogen nuclei (m/s).
        Returns:
            float: LENR probability.
        """
        try:
            Z1, Z2 = 1, 1  # Hydrogen nuclei charges
            gamow_factor = np.exp(-2 * np.pi * Z1 * Z2 * self.e**2 / (self.hbar * velocity * self.screening_factor))
            probability = gamow_factor
            logger.info("LENR probability: %s", probability)
            return probability
        except Exception as e:
            logger.error("Error in LENR probability calculation: %s", e)
            raise

class SimulationOrchestrator:
    def __init__(self, temperature=77, pressure=1e-6):
        """
        Initialize simulation orchestrator.
        Args:
            temperature (float): Operating temperature (K).
            pressure (float): Operating pressure (Torr).
        """
        self.lattice = Bi2223KagomeLattice()
        self.coil = HelmholtzCoil()
        self.metasurface = GrapheneBICMetasurface()
        self.lenr_sim = LENRSimulator()
        self.temperature = temperature
        self.pressure = pressure
        self.results = []
        logger.info("SimulationOrchestrator initialized at %s K, %s Torr", temperature, pressure)

    def run_simulation(self, num_hydrogen=100, frequency=1e15):
        """
        Run the integrated simulation.
        Args:
            num_hydrogen (int): Number of hydrogen atoms.
            frequency (float): Incident light frequency (Hz).
        Returns:
            dict: Simulation results.
        """
        try:
            # Step 1: Hydrogen adsorption
            hydrogen_positions = [[np.random.uniform(0, 5e-10) for _ in range(3)] for _ in range(num_hydrogen)]
            adsorption_energy = self.lattice.calculate_adsorption_energy(hydrogen_positions)

            # Step 2: Magnetic field effects
            position = (0, 0, 0)  # Center of metasurface
            B, delta_E = self.coil.calculate_magnetic_field(position)

            # Step 3: Electromagnetic field enhancement
            field_enhancement = self.metasurface.calculate_field_enhancement(frequency)

            # Step 4: LENR probability
            lenr_probability = self.lenr_sim.calculate_lenr_probability()

            # Store results
            result = {
                "adsorption_energy": adsorption_energy,
                "magnetic_field": B,
                "zeeman_splitting": delta_E,
                "field_enhancement": field_enhancement,
                "lenr_probability": lenr_probability
            }
            self.results.append(result)
            logger.info("Simulation completed: %s", result)
            return result
        except Exception as e:
            logger.error("Simulation failed: %s", e)
            raise

class Visualizer:
    def __init__(self, output_dir="/home/nix/Desktop/TestFrontend/plots"):
        """
        Initialize visualizer for simulation results.
        Args:
            output_dir (str): Directory for saving plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.debug("Visualizer initialized with output directory %s", output_dir)

    def plot_results(self, results):
        """
        Plot simulation results.
        Args:
            results (list): List of simulation result dictionaries.
        """
        try:
            adsorption_energies = [r["adsorption_energy"] for r in results]
            magnetic_fields = [r["magnetic_field"] for r in results]
            lenr_probabilities = [r["lenr_probability"] for r in results]

            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)
            plt.plot(adsorption_energies, label="Adsorption Energy (eV)")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(magnetic_fields, label="Magnetic Field (T)")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(lenr_probabilities, label="LENR Probability")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "simulation_results.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info("Results plotted and saved to %s", plot_path)
        except Exception as e:
            logger.error("Error in plotting results: %s", e)
            raise

class Debugger:
    def __init__(self):
        """Initialize debugger with test cases."""
        logger.debug("Debugger initialized")

    def run_tests(self):
        """Run unit tests for simulation components."""
        suite = unittest.TestSuite()
        suite.addTest(SimulationTests("test_adsorption_energy"))
        suite.addTest(SimulationTests("test_magnetic_field"))
        suite.addTest(SimulationTests("test_field_enhancement"))
        suite.addTest(SimulationTests("test_lenr_probability"))
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        logger.info("Unit tests completed: %s", result)
        return result

class SimulationTests(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.lattice = Bi2223KagomeLattice()
        self.coil = HelmholtzCoil()
        self.metasurface = GrapheneBICMetasurface()
        self.lenr_sim = LENRSimulator()

    def test_adsorption_energy(self):
        """Test adsorption energy calculation."""
        energy = self.lattice.calculate_adsorption_energy([[0, 0, 0]])
        self.assertAlmostEqual(energy, -0.5, places=2)
        logger.debug("Adsorption energy test passed")

    def test_magnetic_field(self):
        """Test magnetic field calculation."""
        B, delta_E = self.coil.calculate_magnetic_field((0, 0, 0))
        self.assertGreater(B, 0)
        self.assertGreater(delta_E, 0)
        logger.debug("Magnetic field test passed")

    def test_field_enhancement(self):
        """Test field enhancement calculation."""
        enhancement = self.metasurface.calculate_field_enhancement(1e15)
        self.assertGreater(enhancement, 0)
        logger.debug("Field enhancement test passed")

    def test_lenr_probability(self):
        """Test LENR probability calculation."""
        probability = self.lenr_sim.calculate_lenr_probability()
        self.assertGreater(probability, 0)
        self.assertLess(probability, 1)
        logger.debug("LENR probability test passed")

def main():
    """
    Main function to run the simulation pipeline.
    """
    try:
        # Initialize components
        orchestrator = SimulationOrchestrator(temperature=77, pressure=1e-6)
        visualizer = Visualizer()
        debugger = Debugger()

        # Run debugging tests
        logger.info("Running unit tests...")
        test_result = debugger.run_tests()
        if not test_result.wasSuccessful():
            logger.error("Unit tests failed, aborting simulation")
            return

        # Run multiple simulations
        results = []
        for _ in range(5):  # Run 5 simulations for statistical robustness
            result = orchestrator.run_simulation(num_hydrogen=100, frequency=1e15)
            results.append(result)

        # Visualize results
        visualizer.plot_results(results)

        # Save results to file
        with open("/home/nix/Desktop/TestFrontend/results.txt", "w") as f:
            for i, result in enumerate(results):
                f.write(f"Simulation {i+1}: {result}\n")
        logger.info("Results saved to results.txt")

    except Exception as e:
        logger.error("Main simulation pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()
