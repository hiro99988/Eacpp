// import numpy as np

// from src.mutations.i_mutation import IMutation

// class PolynomialMutation(IMutation):
//     def __init__(
//         self,
//         mutation_rate: float,
//         distribution_index: float,
//         variable_bounds: list[list[int | float]],
//         seed: int = None,
//     ):
//         self.mutation_rate = mutation_rate
//         self.distribution_index = distribution_index
//         self.variable_bounds = variable_bounds

//         self._rng = np.random.default_rng(seed)
//         self._length = len(variable_bounds)
//         self._last_bound_index = len(variable_bounds) - 1

//     def mutate(self, individual: list[int | float]):
//         rnd = self._rng.random(len(individual))
//         mutated_index = np.where(rnd < self.mutation_rate)[0]
//         if self._length == 1:
//             for i in mutated_index:
//                 individual[i] += self._sigma() * (
//                     self.variable_bounds[0, 1] - self.variable_bounds[0, 0]
//                 )
//         else:
//             for i in mutated_index:
//                 if i < self._last_bound_index:
//                     individual[i] += self._sigma() * (
//                         self.variable_bounds[i, 1] - self.variable_bounds[i, 0]
//                     )
//                 else:
//                     individual[i] += self._sigma() * (
//                         self.variable_bounds[-1, 1] - self.variable_bounds[-1, 0]
//                     )

//     def _sigma(self):
//         if self._rng.random() < 0.5:
//             return (
//                 np.power(2 * self._rng.random(), 1 / (self.distribution_index + 1)) - 1
//             )
//         else:
//             return 1 - np.power(
//                 2 - 2 * self._rng.random(), 1 / (self.distribution_index + 1)
//             )
