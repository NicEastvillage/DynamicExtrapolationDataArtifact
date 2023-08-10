# Dynamic Extrapolation Bench Results

This is a package containing the results related to the study 'Dynamic Extrapolation in Extended Timed Automata' by Nicolaj Ø. Jensen, Peter G. Jensen, and Kim G. Larsen (2023), in: ICFEM'23.

**Abstract:**

Abstractions, such as extrapolation, ensure termination of timed automata model checking.
However, such methods are normally only defined for classical timed automata, whereas modern tools like Uppaal take as input timed automata extended with discrete data and C-like language constructs (XTA) making classical extrapolation excessively overapproximating if even applicable.
In this paper, we propose a new dynamic extrapolation technique for XTAs which utilizes information from the immediate state of the search to find more precise extrapolation values.
We determine which code snippets are relevant to obtain the extrapolation values ahead of verification using static analysis and then execute these dynamically during verification.
We implement our novel extrapolation technique in Uppaal and find that it reduces the zone graph sizes by 34.7% overall compared to a classic location-clock-based extrapolation.
The best case is an 82.7% reduction and the worst case is a surprising 8.2% increase.

## Content

- `models-*/` contain the extended timed automata Uppaal models used for the experiments.
  The models have dynamic time constraints, i.e. constraints where clocks are compared to variable expressions.
  Some of them have come in two variants, with and without user-defined integer ranges for variables.
- `results/` contain the measurements obtained in the benchmark. The data is separated in three setups.
  We measure time spent, size of zone graph (stored), number of successors computer (explored), the residual memory used, and the virtual memory used.
- `scripts/graphs_and_tables.py` is a Python script that process the data and produces a series of graphs and tables which will be output to `graphs/`. 
  The packages required to run this script is found in `requirements.txt` (though it may differ for Linux users).

## How to use

1. Install Python.
2. Run `pip install -r requirements.txt`.
3. Run `python scripts/graphs_and_tables.py`.
4. See output graphs and tables in `graphs/`.
