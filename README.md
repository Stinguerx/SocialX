# Slashdot Graph Analysis
This script analyzes the structural balance of the Slashdot social network, focusing on triad analysis and spectral balancing.
## Requirements

- Python 3.7+
- NetworkX
- NumPy
- SciPy
- Matplotlib
- Seaborn
  
## Usage

Ensure you have the Slashdot dataset file (soc-sign-Slashdot081106.txt) in the same directory as the script.

Run the script:

`python main.py`

The script will perform the following operations:

- Load and transform the Slashdot dataset into a graph
- Create and save visualizations of the graph's characteristics.
- Analyze triads in the graph
- Modify unstable triads
- Perform spectral balancing
- Create and save a visualization of the spectral balancing using a subset of the graph
- Characterize the resulting groups from the partitioning

Results will be printed to the console and to an output log file. Visualizations will be saved to a ./plots folder
