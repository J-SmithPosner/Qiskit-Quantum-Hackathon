#!/usr/bin/env python3
"""Quantum traffic-light scheduler (QAOA, Qiskit).

    pip install qiskit qiskit-aer numpy
    python quantum_traffic.py 1010

INPUT   4 qubits, |0000> = all weight on cars ... |1111> = all weight on pedestrians.
OUTPUT  16 qubits, one per 6.25 s slot (100 s / 16):
        0 = N/S light green for the slot, 1 = E/W light green for the slot.

The circuit has 20 qubits. The priority register coherently controls the cost
Hamiltonian
        H(w) = (1 - w) * H_cars + w * H_peds,    w = input / 15
by adding one controlled rotation per priority qubit, weighted by 2^j / 15.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

HORIZON_S = 100.0
N_SLOTS = 16
SLOT_S = HORIZON_S / N_SLOTS  # the whiteboard's "n sec" = 6.25 s

# Model assumptions: the whiteboard names the inputs but not these values.
CAR_HEADWAY_S = 2.0  # time for one queued car to clear the stop line...
CAR_SPACING_M = 7.0  # ...plus one car length of queue at the average speed
SWITCH_LOST_S = 4.0  # green time lost per light change (yellow + start-up)
MIN_SPEED = 0.5  # m/s, floor applied to sensor speeds
MIN_SHARE = 0.15  # a direction with anyone waiting is never starved
EARLY_SERVICE = 0.15  # how much sooner-is-better matters; also breaks ties
LONG_RED = 3.0  # extra penalty for red slots that sit close together
PED_PATTERN = 0.4  # strength of the walk-pattern term relative to car switching

# Linear-ramp QAOA angles. No classical optimizer loop is needed at run time.
DEFAULT_LAYERS = 6
DEFAULT_GAMMA = 8.0
DEFAULT_BETA = 0.7


@dataclass
class Traffic:
    """The whiteboard's non-user inputs, from intersection sensors.

    Pedestrians labelled N/S walk parallel to N/S traffic, so they may cross
    while the N/S light is green. Speeds are in m/s.
    """

    cars_ns: float = 12
    car_speed_ns: float = 11.0
    cars_ew: float = 4
    car_speed_ew: float = 8.0
    peds_ns: float = 5
    ped_speed_ns: float = 1.4
    peds_ew: float = 20
    ped_speed_ew: float = 1.1
    crossing_width_m: float = 12.0

    def __post_init__(self) -> None:
        # A stopped queue reports speed 0; keep the travel-time formulas finite.
        for name in ("car_speed_ns", "car_speed_ew", "ped_speed_ns", "ped_speed_ew"):
            setattr(self, name, max(getattr(self, name), MIN_SPEED))

    def car_loads(self) -> tuple[float, float]:
        """Seconds of green needed to clear each direction's cars."""
        return (
            self.cars_ns * (CAR_HEADWAY_S + CAR_SPACING_M / self.car_speed_ns),
            self.cars_ew * (CAR_HEADWAY_S + CAR_SPACING_M / self.car_speed_ew),
        )

    def crossing_times(self) -> tuple[float, float]:
        """Seconds a pedestrian needs to cross in each direction."""
        return (
            self.crossing_width_m / self.ped_speed_ns,
            self.crossing_width_m / self.ped_speed_ew,
        )

    def ped_loads(self) -> tuple[float, float]:
        cross_ns, cross_ew = self.crossing_times()
        return self.peds_ns * cross_ns, self.peds_ew * cross_ew


# ---------------------------------------------------------------- Hamiltonians
# Each is an Ising model E(z) = sum h[i] z[i] + sum_{i<j} J[i,j] z[i] z[j],
# with z = +1 for N/S green and z = -1 for E/W green. Lower energy = happier.


def _shares(loads: tuple[float, float]) -> tuple[float, float]:
    """Each direction's share of the load. Anyone waiting gets at least MIN_SHARE."""
    total = loads[0] + loads[1]
    if total <= 0:
        return 0.5, 0.5
    ns = loads[0] / total
    if loads[0] > 0 and loads[1] > 0:
        ns = float(np.clip(ns, MIN_SHARE, 1 - MIN_SHARE))
    return ns, 1 - ns


def _happiness_terms(loads: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (1.1) + (1.2): each direction's unhappiness comes from its red time.

    Each direction pays load * K for every pair of red slots. K is larger for
    nearby slots, because one long red hurts more than the same red split up.
    The minimum gives each direction green roughly in proportion to its load,
    so 100% car weight still serves the quieter direction, as the whiteboard's
    red note requires. A second term prefers serving the larger queue first.
    """
    rho_ns, rho_ew = _shares(loads)
    T = N_SLOTS
    distance = np.abs(np.arange(T)[:, None] - np.arange(T))
    K = 1 + LONG_RED * (1 - distance / T)
    np.fill_diagonal(K, 0)
    J = np.triu(K, k=1) / (2 * T * T)
    earliness = (T - np.arange(T)) / T
    h = (rho_ew - rho_ns) * ((K.sum(axis=1) + 1) / (2 * T * T) + EARLY_SERVICE * earliness / (2 * T))
    return h, J


def _unit_scale(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale to unit energy spread, so w trades cars against peds fairly."""
    scale = np.sqrt(np.sum(h**2) + np.sum(J**2))
    return h / scale, J / scale


def car_hamiltonian(traffic: Traffic) -> tuple[np.ndarray, np.ndarray]:
    """Cars want green in proportion to load, and few light changes."""
    if traffic.cars_ns + traffic.cars_ew == 0 and traffic.peds_ns + traffic.peds_ew > 0:
        return ped_hamiltonian(traffic)  # no cars to weigh: serve whoever is there
    h, J = _happiness_terms(traffic.car_loads())
    J = J.copy()
    # Each change costs lost time: -k z_t z_(t+1) favours long, unbroken greens.
    k = 0.5 * SWITCH_LOST_S / HORIZON_S
    for t in range(N_SLOTS - 1):
        J[t, t + 1] -= k
    return _unit_scale(h, J)


def walk_slots(traffic: Traffic) -> int:
    """Slots a green must last for the slowest waiting pedestrian to cross."""
    times = [
        time
        for time, count in zip(
            traffic.crossing_times(), (traffic.peds_ns, traffic.peds_ew)
        )
        if count > 0
    ]
    return max(1, int(np.ceil(max(times, default=0.0) / SLOT_S)))


def ped_hamiltonian(traffic: Traffic) -> tuple[np.ndarray, np.ndarray]:
    """Peds want green in proportion to load, long enough to cross, and often.

    With L = walk_slots: slots closer than L apart prefer the same colour, so a
    green lasts long enough to cross, and slots exactly L apart prefer opposite
    colours, so nobody waits much longer than one crossing.
    """
    if traffic.peds_ns + traffic.peds_ew == 0:
        return car_hamiltonian(traffic)  # no peds to weigh: serve whoever is there
    h, J = _happiness_terms(traffic.ped_loads())
    J = J.copy()
    L = walk_slots(traffic)
    k = PED_PATTERN * 0.5 * SWITCH_LOST_S / HORIZON_S
    for t in range(N_SLOTS):
        for delta in range(1, L + 1):
            if t + delta < N_SLOTS:
                J[t, t + delta] += k if delta == L else -k
    return _unit_scale(h, J)


def energies(h: np.ndarray, J: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Energy of each row of a binary schedule array."""
    z = 1 - 2 * np.atleast_2d(x)
    return z @ h + np.sum((z @ J) * z, axis=1)


# -------------------------------------------------------------------- Circuit


def build_circuit(
    priority: str,
    traffic: Traffic,
    layers: int = DEFAULT_LAYERS,
    gamma: float = DEFAULT_GAMMA,
    beta: float = DEFAULT_BETA,
) -> QuantumCircuit:
    """Build the 20-qubit QAOA circuit for a 4-bit priority such as '1010'."""
    if len(priority) != 4 or set(priority) - {"0", "1"}:
        raise ValueError("priority must be four bits, from '0000' to '1111'")

    h_car, J_car = car_hamiltonian(traffic)
    h_ped, J_ped = ped_hamiltonian(traffic)

    inp = QuantumRegister(4, "priority")
    out = QuantumRegister(N_SLOTS, "slot")
    creg = ClassicalRegister(N_SLOTS, "schedule")
    qc = QuantumCircuit(inp, out, creg, name="traffic_qaoa")

    # priority[0] is the rightmost (least significant) character of the input.
    for j, bit in enumerate(reversed(priority)):
        if bit == "1":
            qc.x(inp[j])
    qc.h(out)

    def z_rotation(target, g: float, car: float, ped: float) -> None:
        """exp(-i g [(1-w) car + w ped] Z), with w read from the priority qubits."""
        qc.rz(2 * g * car, target)
        for j in range(4):
            qc.crz(2 * g * (ped - car) * 2**j / 15, inp[j], target)

    for layer in range(layers):
        ramp = (layer + 0.5) / layers
        g, b = gamma * ramp, beta * (1 - ramp)

        for t in range(N_SLOTS):  # single-slot terms
            z_rotation(out[t], g, h_car[t], h_ped[t])

        for s in range(N_SLOTS):  # slot-pair terms
            for t in range(s + 1, N_SLOTS):
                if J_car[s, t] or J_ped[s, t]:
                    qc.cx(out[s], out[t])
                    z_rotation(out[t], g, J_car[s, t], J_ped[s, t])
                    qc.cx(out[s], out[t])

        # Mixer exp(+i b X). The sign matters: |+> is the ground state of -X, so
        # this direction lowers the energy, while RX(+2b) would raise it.
        qc.rx(-2 * b, out)

    qc.measure(out, creg)
    return qc


# --------------------------------------------------------------------- Results


def schedule_to_changes(bits: str) -> list[dict]:
    """Merge a 16-character schedule (slot 0 first) into light phases."""
    changes: list[dict] = []
    for t, bit in enumerate(bits):
        green = "N/S" if bit == "0" else "E/W"
        if changes and changes[-1]["green"] == green:
            changes[-1]["end_s"] = (t + 1) * SLOT_S
        else:
            changes.append({"start_s": t * SLOT_S, "end_s": (t + 1) * SLOT_S, "green": green})
    return changes


def next_100_seconds(
    priority: str,
    traffic: Traffic | None = None,
    shots: int = 4096,
    seed: int | None = 7,
    layers: int = DEFAULT_LAYERS,
) -> dict:
    """Run the algorithm and return the next 100 seconds of light changes."""
    traffic = traffic or Traffic()
    qc = build_circuit(priority, traffic, layers=layers)
    backend = AerSimulator(method="statevector")
    compiled = transpile(qc, backend, optimization_level=1)
    counts = backend.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()

    # Qiskit prints bit 0 on the right; reverse so that slot 0 comes first.
    samples = {key.replace(" ", "")[::-1]: n for key, n in counts.items()}
    candidates = sorted(samples)
    x = np.array([[int(c) for c in s] for s in candidates])

    w = int(priority, 2) / 15
    h_car, J_car = car_hamiltonian(traffic)
    h_ped, J_ped = ped_hamiltonian(traffic)
    energy = (1 - w) * energies(h_car, J_car, x) + w * energies(h_ped, J_ped, x)
    best = candidates[int(np.argmin(energy))]

    return {
        "priority": priority,
        "ped_weight": w,
        "bits": best,
        "energy": float(np.min(energy)),
        "changes": schedule_to_changes(best),
        "simulation": simulate(best, traffic),
    }


def exact_best(priority: str, traffic: Traffic | None = None) -> tuple[str, float]:
    """Classical brute force over all 65,536 schedules, used as a benchmark."""
    traffic = traffic or Traffic()
    w = int(priority, 2) / 15
    x = (np.arange(2**N_SLOTS)[:, None] >> np.arange(N_SLOTS)) & 1
    h_car, J_car = car_hamiltonian(traffic)
    h_ped, J_ped = ped_hamiltonian(traffic)
    energy = (1 - w) * energies(h_car, J_car, x) + w * energies(h_ped, J_ped, x)
    i = int(np.argmin(energy))
    return "".join(map(str, x[i])), float(energy[i])


def simulate(bits: str, traffic: Traffic, dt: float = 0.25) -> dict:
    """The whiteboard's "Simulation with results": a simple 100 s queue model.

    Each direction starts with its sensor counts, and the same number arrive
    evenly over the 100 s. Cars discharge one at a time during green, after the
    lost time at the start of each phase. A pedestrian starts crossing only if
    the remaining green is long enough to finish.
    """
    changes = schedule_to_changes(bits)
    cars = {"N/S": float(traffic.cars_ns), "E/W": float(traffic.cars_ew)}
    peds = {"N/S": float(traffic.peds_ns), "E/W": float(traffic.peds_ew)}
    car_rate = {d: cars[d] / HORIZON_S for d in cars}
    ped_rate = {d: peds[d] / HORIZON_S for d in peds}
    car_load = dict(zip(("N/S", "E/W"), traffic.car_loads()))
    cross = dict(zip(("N/S", "E/W"), traffic.crossing_times()))
    service = {
        d: (cars[d] / car_load[d] if car_load[d] > 0 else 0.0) for d in cars
    }
    total_cars = 2 * sum(cars.values())
    total_peds = 2 * sum(peds.values())
    car_wait = ped_wait = 0.0

    for step in range(int(round(HORIZON_S / dt))):
        now = step * dt
        phase = next(c for c in changes if c["start_s"] <= now < c["end_s"])
        green = phase["green"]
        for d in cars:
            cars[d] += car_rate[d] * dt
            peds[d] += ped_rate[d] * dt
        if now - phase["start_s"] >= SWITCH_LOST_S:
            cars[green] = max(0.0, cars[green] - service[green] * dt)
        if phase["end_s"] - now >= cross[green]:
            peds[green] = 0.0
        car_wait += sum(cars.values()) * dt
        ped_wait += sum(peds.values()) * dt

    return {
        "avg_car_wait_s": car_wait / total_cars if total_cars else 0.0,
        "avg_ped_wait_s": ped_wait / total_peds if total_peds else 0.0,
        "cars_left": sum(cars.values()),
        "peds_left": sum(peds.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("priority", help="pedestrian priority, '0000' to '1111'")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    defaults = Traffic()
    for name, value in vars(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=value)
    args = parser.parse_args()

    traffic = Traffic(**{name: getattr(args, name) for name in vars(defaults)})
    result = next_100_seconds(args.priority, traffic, args.shots, args.seed)

    print(f"Pedestrian priority |{result['priority']}> = {result['ped_weight']:.0%}")
    print(f"Schedule bits (first slot on the left): {result['bits']}\n")
    for phase in result["changes"]:
        print(f"{phase['start_s']:6.2f} s to {phase['end_s']:6.2f} s   {phase['green']} green")
    sim = result["simulation"]
    print(
        f"\nSimulation: average wait {sim['avg_car_wait_s']:.1f} s per car, "
        f"{sim['avg_ped_wait_s']:.1f} s per pedestrian; "
        f"still waiting at 100 s: {sim['cars_left']:.0f} cars, {sim['peds_left']:.0f} pedestrians"
    )


if __name__ == "__main__":
    main()
