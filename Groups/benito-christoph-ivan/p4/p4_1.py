from sympy.stats import ContinuousMarkovChain
from sympy import Matrix, S


def main():
    G = Matrix([
        [-S(3)/S(2), S(3)/S(2), S(0), S(0)],
        [S(3), -S(9)/S(2), S(3)/S(2), S(0)],
        [S(0), S(3), -S(9)/S(2), S(3)/S(2)],
        [S(0), S(0), S(3), -S(3)]
    ])

    print(limiting_distribution(C))


def limiting_distribution(G):
    C = ContinuousMarkovChain('C', state_space=list(range(G.shape[0])), gen_mat=G)

    return C.limiting_distribution()
