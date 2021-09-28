import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import imageio
from glob import glob

GRID_SIZE = 10


class RandomCustomers:
    def __init__(self, m=10, demand=(0.05, 0.2), growth=0.05, gs=10):
        self.num_customers = m
        self.gs = gs
        self.locations = [(round(np.random.uniform(-gs, gs), 5),
                           round(np.random.uniform(-gs, gs), 5)) for _ in range(m)]
        self.demands = [round(np.random.uniform(demand[0], demand[1]), 3) for _ in range(m)]
        self.grid_size = gs
        self.growth = growth


class RandomDepots:
    def __init__(self, n=2, capacity=(0.2, 0.8), f=0, gs=10):
        self.num_depots = n
        self.locations = [(round(np.random.uniform(-gs, gs), 5),
                           round(np.random.uniform(-gs, gs), 5)) for _ in range(n)]
        self.capacities = [round(np.random.uniform(capacity[0], capacity[1]), 3) for _ in range(n)]
        self.fixed_costs = [f] * self.num_depots
        self.grid_size = gs


class RunModel:
    def __init__(self, customers, depots, iterations=1):
        self.customers = customers
        self.depots = depots
        self.costs = calc_costs(customers, depots)
        self.iterations = iterations

    def solve(self, mode="strict"):
        if sum(self.depots.capacities) < sum(self.customers.demands):
            print(f"Total demand of {sum(self.customers.demands)} exceeds capacity of {sum(self.depots.capacities)}")
            print("Exiting")
            exit(0)
        m = gp.Model()
        depots = m.addVars(range(self.depots.num_depots),
                           vtype=GRB.BINARY,
                           obj=self.depots.fixed_costs,
                           name="dep")

        transport = m.addVars(self.customers.num_customers,
                              self.depots.num_depots,
                              obj=self.costs,
                              name="trans")

        m.modelSense = GRB.MINIMIZE

        m.addConstrs(
            (transport.sum('*', i) <= self.depots.capacities[i] * depots[i] for i in range(self.depots.num_depots)),
            "Capacity")

        m.addConstrs(
            (transport.sum(j) == self.customers.demands[j] for j in range(self.customers.num_customers)),
            "Demand")

        m.write('models/facilityPY.lp')
        m.optimize()
        model_history = {self.iterations: m}
        if m.status == GRB.OPTIMAL:
            plot_results(self.customers, self.depots, depots, transport, self.iterations, mode)

        else:
            print("Model is infeasible, try increasing capacity/depots, or decreasing customers/demand")
            exit(0)

        for i in range(self.iterations - 1, 0, -1):
            # Downgrade demands
            self.customers.demands = [round(x/(1 + self.customers.growth), 3) for x in self.customers.demands]
            if mode == "strict":
                for de in depots:
                    if depots[de].X == 0:
                        depots[de].ub = 0

            # TODO - make this more readable
            m.remove(m.getConstrs()[self.depots.num_depots:])

            m.addConstrs(
                (transport.sum(j) == self.customers.demands[j] for j in range(self.customers.num_customers)),
                "Demand")
            m.update()
            m.printStats()

            m.optimize()

            model_history[i] = m

            if m.status == GRB.OPTIMAL:
                plot_results(self.customers, self.depots, depots, transport, i, mode)

            else:
                print("Model is infeasible, try increasing capacity/depots, or decreasing customers/demand")
                exit(0)


def pythag(cus, dep):
    return np.sqrt(((cus[0] - dep[0])**2 + (cus[1] - dep[1])**2))


def calc_costs(customers, depots):
    costs = np.zeros((customers.num_customers, depots.num_depots))
    for i, cus in enumerate(customers.locations):
        for j, dep in enumerate(depots.locations):
            costs[i, j] = pythag(cus, dep)
    return costs


def plot_results(customers, depots, depot_soln, transport, i, mode):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    plt.xlim((-customers.gs, customers.gs))
    plt.ylim((-customers.gs, customers.gs))

    for p in range(depots.num_depots):
        if depot_soln[p].x > 0.99:
            capacity_filled = 0
            circle = plt.Circle(depots.locations[p], depots.capacities[p], color='blue', fill=False, lw=2, alpha=0.5)
            ax.add_artist(circle)

            for w in range(customers.num_customers):
                demand_satisfied = transport[w, p].x
                if demand_satisfied > 0:
                    circle = plt.Circle(customers.locations[w], customers.demands[w], color='red', fill=False, lw=2)
                    ax.add_artist(circle)
                    line = plt.Line2D((customers.locations[w][0], depots.locations[p][0]),
                                      (customers.locations[w][1], depots.locations[p][1]),
                                      linewidth=demand_satisfied * 20,
                                      color="black")
                    ax.add_artist(line)
                    capacity_filled += demand_satisfied

            circle = plt.Circle(depots.locations[p], capacity_filled, color='blue', fill=True, lw=0, alpha=0.2)
            ax.add_artist(circle)
        else:
            circle = plt.Circle(depots.locations[p], depots.capacities[p], color='blue', fill=False, lw=2, alpha=0.2)
            ax.add_artist(circle)
    plt.title(f"Iteration {i}")
    if mode == "strict":
        plt.savefig(f"plots/solution_{i}_strict.png")
    else:
        plt.savefig(f"plots/solution_{i}_lax.png")
    # plt.show()


c = RandomCustomers(m=50, demand=(0.05, 0.2), growth=0.1, gs=GRID_SIZE)
d = RandomDepots(n=30, capacity=(0.2, 1.0), f=20, gs=GRID_SIZE)
mo = RunModel(c, d, iterations=9)
mo.solve()
mo.solve("lax")

images = []

for filename in sorted(glob("plots/solution*strict.png")):
    images.append(imageio.imread(filename))
imageio.mimsave('plots/omg.gif', images, duration=2)

for filename in sorted(glob("plots/solution*lax.png")):
    images.append(imageio.imread(filename))
imageio.mimsave('plots/omg2.gif', images, duration=2)
