import numpy as np
import pandas as pd


def main():
    profile_results = pd.read_csv('../p3/profile.csv', index_col=False)

    profile_results = profile_results.drop(columns=['episode'])
    profile_results = profile_results.groupby(['state', 'next_state', 'time_index']).mean().reset_index()

    profile_results = profile_results.groupby(['state', 'next_state']).apply(
        lambda x: fit(x['time_index'], x['time'])
    )

    print(profile_results)

    # TODO NEXT: add remaining states as NaN, use code from p4_1 to compute limiting distribution. should we set a high epsilon in p3 for covering more states?


def fit(X, Y):
    print(X, Y)
    try:
        [λ, b] = np.polyfit(X, np.log(Y), 1, w=np.sqrt(Y))
    except:
        λ = b = np.nan
    print(λ, b)
    print()
    # TODO: what to do with b?
    return λ


if __name__ == '__main__':
    main()
