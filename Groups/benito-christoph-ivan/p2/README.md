# P2 - HMM (Benito, Christoph, Ivan)

## Instructions

> 1. Apply the Baum-Welch algorithm to learn a model that allows to predict the traces of observations produced by your DTMC
> 
>    Use the learned model to answer the following questions.
> 2. What is the probability that your model has generated the following sequence of observations:
> 
>    1. Intermittent failure in component-1 $O_1^∗→O_1^d→O_1^∗$
>    2. Intermittent failure in component-2 $O_2^∗→O_2^d→O_2^∗$
>    3. Failure cascade $O_1^d→O_2^d→O_2^u$
>    4. Failure cascade $O_1^d→O_1^u→O_2^u$
>    5. Failure masking $O_1^u→O_2^u→O_1^∗$
> 
> 3. What is the most probable sequence of states for each of the five above observations?

## Solution

> 1. Apply the Baum-Welch algorithm to learn a model that allows to predict the traces of observations produced by your DTMC

First, we reuse the original transition matrix constructed in P1 and convert them into an equivalent emission matrix (`generate_emissions()`). Just for the sake of distinguishing both matrices from each other, the emission matrix uses the reverse order of each state tuple (e.g., $u_1^O$ instead of $O_1^u$). We also add noise to the emission matrix to make learning non-trivial. We generate observations from the emission matrix by converting walkthroughs created by the method from P1 to the relevant emission for each visited state (`generate_observations()`).

For learning (`learn_hmm()`), we apply the Baum-Welch algorithm to a `CategoricalHMM` from `hmmlearn`. This requires us to quantize all observations into a numeric representation first and convert the resulting model back to the meaningful states afterward. Since Baum-Welch is not stable, we run it multiple times with different random seeds, compute the score for each trained model, and choose the model with the highest score. For computing the score, we split up the observations into two random disjunct subsets for training and testing.

> 2. What is the probability that your model has generated the following sequence of observations

Analogously to P1T5, we use intuitive list comprehension/numpy queries for each requested probability (`answer_questions()`).

> 3. What is the most probable sequence of states for each of the five above observations?

We use the `predict()` method from `hmmlearn` to compute the most probable state sequence for each observation (`predict_states()`).

## Implementation notes

- Depending on the specified arguments, we cache different intermediate results using `pickle` to speed up the feedback loop.
