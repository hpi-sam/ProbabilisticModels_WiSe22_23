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

    # TODO question: should we use proabilities from p1 instead of values from slide 6?

    #lambda = l_1,....,l_n
    #transition matrix of some MC as initial transitions
    #denote P
    #l_i = 1 - p_ii -- probability of jump at the begining
    #it means: to change the state to another

    #gij = l_i * pij, i!=j
    #g_ij = -l_i, i == j
    #so, if i!=j:
    #G[i] = l_i * P[i] = (1 - P[i][i])*P[i]
    #P is start transition matrix

    #TO TASK-2
    #P = (transition matrix from p3)
    #G = [[(1 - P[i][i])*P[i]] for i in range(len(P))]


def limiting_distribution(G):
    C = ContinuousMarkovChain('C', state_space=list(range(G.shape[0])), gen_mat=G)

    return C.limiting_distribution()
