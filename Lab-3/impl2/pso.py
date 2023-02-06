import numpy as np
import logging
from abc import ABC, abstractmethod
from numpy.random import default_rng

from typing import Tuple, Union

# pylint: disable=too-many-instance-attributes

LOGGER = logging.getLogger(__name__)


class Coordinate:
    def __init__(self, **kwargs) -> None:
        """
        Initializes a new random coordinate.
        """
        self.__lower_boundary = kwargs.get('lower_boundary', 0.)
        self.__upper_boundary = kwargs.get('upper_boundary', 4.)
        self._random = kwargs['bit_generator']
        self._function = kwargs['function']

        self.__value:float = 0.0
        self.__position:np.ndarray = np.array([])
        self._initialize()

    def _initialize(self) -> None:
        """
        Initialize a new random position and its value
        """
        self._position = self._random.uniform(
            self.__lower_boundary, self.__upper_boundary, 2)

    @property
    def position(self) -> np.ndarray:
        """
        Get the coordinate's position

        Returns:
            numpy.ndarray: the Position
        """
        return self._position

    # Internal Getter
    @property
    def _position(self) -> np.ndarray:
        return self.__position

    # Internal Setter for automatic position clipping and value update
    @_position.setter
    def _position(self, new_pos: np.ndarray) -> None:
        """
        Set the coordinate's new position.
        Also updates checks whether the position is within the set boundaries
        and updates the coordinate's value.

        Args:
            new_pos (numpy.ndarray): The new coordinate position
        """
        self.__position = np.clip(
            new_pos, a_min=self.__lower_boundary, a_max=self.__upper_boundary)
        self.__value = self._function(self.__position)

    @property
    def value(self) -> float:
        return self.__value

    def __eq__(self, other) -> bool:
        return self.__value == other.value

    def __ne__(self, other) -> bool:
        return self.__value != other.value

    def __lt__(self, other) -> bool:
        return self.__value < other.value

    def __le__(self, other) -> bool:
        return self.__value <= other.value

    def __gt__(self, other) -> bool:
        return self.__value > other.value

    def __ge__(self, other) -> bool:
        return self.__value >= other.value


class ProblemBase(ABC):
    def __init__(self, **kwargs) -> None:
        self._random = default_rng(kwargs.get('seed', None))

    @abstractmethod
    def solve(self) -> Coordinate:
        pass

class Particle(Coordinate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__w = kwargs.get('weight', .5)
        self.__c_1 = kwargs.get('c_1', 2)
        self.__c_2 = kwargs.get('c_2', 2)
        self.__max_velocity = kwargs.get('maximum_velocity', 2)

        # Randomly create a new particle properties
        self.__velocity = self._random.uniform(-1, 1, size=2)
        self.__clip_velocity()

        # Local best
        self.__best_position = self._position
        self.__best_value = self.value

    @property
    def velocity(self) -> float:
        return self.__velocity

    def step(self, global_best_pos: Union[Tuple[float, float], np.ndarray]) -> None:
        """
        Execute a particle step.
        Update the particle's velocity, position and value.

        Arguments:
            global_best_pos {Tuple[float, float]} -- The global best position
        """

        # Calculate velocity
        cognitive_velocity = self.__c_1 * \
            self._random.random(size=2) * \
            (self.__best_position - self._position)
        social_velocity = self.__c_2 * \
            self._random.random(size=2) * (global_best_pos - self._position)
        self.__velocity = self.__w * self.__velocity + \
            cognitive_velocity + social_velocity

        # Clip velocity
        self.__clip_velocity()

        # Update position and clip it to boundaries
        self._position = self._position + self.__velocity

        # Update local best
        if self.value < self.__best_value:
            self.__best_position = self._position
            self.__best_value = self.value

    def __clip_velocity(self):
        norm = np.linalg.norm(self.__velocity)
        if norm > self.__max_velocity:
            self.__velocity *= self.__max_velocity/norm


class PSOProblem(ProblemBase):
    def __init__(self, **kwargs):
        """
        Initialize a new particle swarm optimization problem.
        """
        super().__init__(**kwargs)
        self.__iteration_number = kwargs['iteration_number']
        self.__particles = [
            Particle(**kwargs, bit_generator=self._random)
            for _ in range(kwargs['particles'])
        ]

    def solve(self) -> Particle:
        # And also update global_best_particle
        for _ in range(self.__iteration_number):

            # Update global best
            global_best_particle = min(self.__particles)

            for particle in self.__particles:
                particle.step(global_best_particle.position)

        LOGGER.info('Last best solution="%s" at position="%s"',
                    global_best_particle.value, global_best_particle.position)
        return global_best_particle

def _run_pso(args):
    LOGGER.info('Start particle swarm optimization with parameters="%s"', args)
    args['function'] = FUNCTIONS[args['function']]

    problem = PSOProblem(**args)
    problem.solve()
