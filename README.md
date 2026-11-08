# Bridging Quantum and City Planning

A hybrid quantum/classical approach to optimizing traffic at urban intersections, built for the Qiskit Fall Fest hackathon.

## Team

- Jonah
- Ihan
- Yoav

## The Problem

Urban intersections have become a place of frustration, mismanagement, and delay in the daily lives of millions of people. Our aim is to optimize the contentment of everyone involved, namely **cars and pedestrians**, rather than favoring one group at the other's expense.

## Our Approach

<!-- TODO: fill in from slides 7-9 (Brainstorm / Solution), which are image-only -->

1. **Model the intersection.** Describe cars and pedestrians as competing demands on a shared intersection.
2. **Encode the problem for Qiskit.** Translate the optimization into a form a quantum circuit can work on (e.g. a cost function / Hamiltonian over signal states).
3. **Solve with a hybrid loop.** The quantum circuit is run with Qiskit, and classical code handles setup, scoring, and interpreting the results.
4. **Evaluate.** Compare outcomes for cars and pedestrians to check that both groups benefit.


<img width="626" height="356" alt="image" src="docs/screenshot.png" />


## Why Quantum?

Intersection scheduling is a combinatorial optimization problem: the number of possible signal configurations grows quickly as more lanes, crosswalks, and agents are added. That structure is a natural fit for quantum optimization approaches.

## Results

Have we found a quantum advantage? Frankly, no. But we showed that our algorithm succeeds at optimizing outcomes for both cars and pedestrians.

<!-- TODO: add a results figure or table (wait times, satisfaction scores, etc.) -->

## Getting Started

### Requirements

```
numpy>=2.3.2
qiskit==2.5.1
matplotlib==3.11.1
pylatexenc==2.10
```

### Install

```bash
git clone https://github.com/J-SmithPosner/Qiskit-Quantum-Hackatthon.git
cd Qiskit-Quantum-Hackatthon

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

Open the project notebook in VS Code or Jupyter and run the cells in order.

<!-- TODO: replace with the actual notebook / script filename -->

## Repository Structure

<!-- TODO: update to match the repo -->

```
.
├── README.md
├── requirements.txt
└── <notebook>.ipynb
```

## Acknowledgments

Built for the Qiskit Fall Fest. Thanks to the organizers and mentors.
