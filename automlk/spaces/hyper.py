import random
from abc import ABCMeta, abstractmethod


class AbstractHyper():
    """
    abstract class for hyper optimization
    """

    @abstractmethod
    def get_rand(self):
        """
        generate random value
        :return: random value in the space
        """
        return random.choice(self.vals)


class HyperWeights(AbstractHyper):
    """
    class for choices with weighted probabilities, entered as a weighted dictionary, eg: {'key': proba, ...}
    """

    def __init__(self, dict_vals):
        """

        :param dict_vals: dictionary value:probability
        """
        # values
        self.vals = [x for x in dict_vals.keys()]
        # probabilities
        self.weights = [dict_vals[x] for x in self.vals]

    def get_rand(self):
        return random.choices(self.vals, weights=self.weights)[0]


class HyperChoice(AbstractHyper):
    """
    class for choice between various values in a list
    """

    def __init__(self, vals):
        self.vals = vals


class HyperRangeInt(AbstractHyper):
    # class for choices in a integer range (start, end, step)

    def __init__(self, start, end, step=1):
        self.vals = [x for x in range(start, end, step)]


class HyperRangeFloat(AbstractHyper):
    # class for choices in a float range (start, end, number of values)
    def __init__(self, start, end, n=100):
        self.vals = [start + (end - start) * i / n for i in range(n)]


def eval_space_key(sk):
    # evaluate the space key
    if isinstance(sk, AbstractHyper):
        result = sk.get_rand()
        if isinstance(result, AbstractHyper):
            # if the result is an Hyper object, we will recursively evaluate
            # print('recursive', result)
            return eval_space_key(result)
        else:
            if isinstance(result, list):
                # choice within a list
                return random.choice(result)
            else:
                # unique value
                return result
    else:
        return sk


def get_random_params(space):
    # generate a parameter list with random from space definition
    param = {}
    for key in space.keys():
        param[key] = eval_space_key(space[key])
    return param
