import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definition --------------------
def make_bitpattern_problem():
    """Fitness reaches 80 when number of ones = 50"""
    def fitness_fn(x: np.ndarray) -> float:
        num_ones = np.sum(x)
        return 80 - abs(50 - num_ones)
    return {
        "name": "Bit Pattern (max when 50 ones)",
        "dim": 80,
        "fitness_fn": fitness_fn
    }

# -------------------- GA Operators --------------------
def init_population(pop_size, gene_length, rng):
    return rng.integers(0, 2, size=(pop_size, gene_length), dtype=np.int8)

def tournament_selection(fitness, k, rng):
    idxs = rng.integers(0, fitness.size, size=k)
    best = idxs[np.argmax(fitness[idxs])]
    return int(best)

def one_point_crossover(a, b, rng):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x, mut_rate, rng):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop, fitness_fn):
    return np.array([fitness_fn(ind) for ind in pop], dtype=float)

# -------------------- GA Core --------------------
def run_ga(pop_size, generations, crossover_rate, mutation_rate, tournament_k, elitism, seed, live):
    rng = np.random.default_rng(seed)
    problem = make_bitpattern_problem()
    gene_length = problem["dim"]
    fitness_fn = problem["fitness_fn"]

    pop = init_population(pop_size, gene_length, rng)
    fit = evaluate(pop, fitness_fn)

    history_best, history_avg, history_worst = [], [], []
    chart_area = st.empty()
    info_area = st.empty()

    for gen in range(generations):
        # Record metrics
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            info_area.markdown(f"Generation {gen+1}/{generations} — Best fitness: **{best_fit:.2f}**")

        # Elitism
        E = max(0, min(elitism, pop_size))
        elite_idx = np.argpartition(fit, -E)[-E:] if E > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if E > 0 else np.empty((0, pop.shape[1]))

        # Create next generation
        next_pop = []
        while len(next_pop) < pop_size - E:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - E:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop), elites]) if E > 0 else np.array(next_pop)
        fit = evaluate(pop, fitness_fn)

    best_idx = int(np.argmax(fit))
    best = pop[best_idx]
    best_fit = fit[best_idx]
    df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})

    return best, best_fit, df, pop, fit

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Bit Pattern Genetic Algorithm", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm — Bit Pattern Optimization")
st.caption("Maximizes fitness (80 − |50 − ones|). Best solution has 50 ones among 80 bits.")

with st.sidebar:
    st.header("GA Parameters")
    pop_size = st.number_input("Population size", min_value=50, max_value=1000, value=300, step=50)
    generations = st.number_input("Generations", min_value=1, max_value=500, value=50, step=10)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate", 0.0, 0.1, 0.01, 0.005)
    tournament_k = st.slider("Tournament size", 2, 10, 3)
    elitism = st.slider("Elites per generation", 0, 50, 2)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    live = st.checkbox("Show live chart while running", value=True)

if st.button("Run Genetic Algorithm", type="primary"):
    with st.spinner("Running Genetic Algorithm..."):
        best, best_fit, history_df, pop, fit = run_ga(
            int(pop_size),
            int(generations),
            float(crossover_rate),
            float(mutation_rate),
            int(tournament_k),
            int(elitism),
            int(seed),
            bool(live),
        )

    st.success("GA Completed!")
    st.subheader("Fitness Progress")
    st.line_chart(history_df)

    st.subheader("Best Solution Found")
    bitstring = ''.join(map(str, best.astype(int).tolist()))
    st.code(bitstring, language="text")
    st.write(f"Number of ones: **{int(np.sum(best))} / 80**")
    st.write(f"Fitness value: **{best_fit:.2f}**")

    st.caption("Fitness increases as the number of ones approaches 50 (max 80).")

    # Show final population table (first 20)
    st.subheader("Population Snapshot (Final Generation)")
    nshow = min(20, pop.shape[0])
    df = pd.DataFrame(pop[:nshow])
    df["fitness"] = fit[:nshow]
    st.dataframe(df, use_container_width=True)

