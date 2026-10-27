# E-L* on random minimal DFAs

800 runs. Learned means accuracy at least 0.99; right_size means the learned DFA has the target's number of states.

| states | eta | runs | learned | exact | right_size | accuracy_min | accuracy_median | queries_median | queries_p90 | seconds_median | seconds_max | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.05 | 100 | 1 | 1 | 1 | 1 | 1 | 9.19e+04 | 1.7e+05 | 4.31 | 28.4 | 0 |
| 6 | 0.1 | 100 | 1 | 1 | 1 | 1 | 1 | 1.67e+05 | 2.58e+05 | 7.43 | 368 | 0 |
| 6 | 0.2 | 100 | 1 | 1 | 1 | 1 | 1 | 4.15e+05 | 8.7e+05 | 18.4 | 211 | 0 |
| 6 | 0.3 | 100 | 1 | 1 | 1 | 1 | 1 | 1.81e+06 | 4.68e+06 | 86.3 | 795 | 0 |
| 12 | 0.05 | 100 | 1 | 0.99 | 0.99 | 0.999 | 1 | 1.99e+05 | 4.44e+05 | 10.7 | 173 | 0 |
| 12 | 0.1 | 100 | 1 | 0.99 | 0.99 | 0.996 | 1 | 3.13e+05 | 8.45e+05 | 15.8 | 466 | 0 |
| 12 | 0.2 | 100 | 0.99 | 0.98 | 0.99 | 0.984 | 1 | 1.02e+06 | 2.18e+06 | 52.6 | 700 | 0 |
| 12 | 0.3 | 100 | 1 | 1 | 1 | 1 | 1 | 4.1e+06 | 1.05e+07 | 194 | 1.97e+03 | 0 |

## Runs not learned (1)

| target | eta | seed | accuracy | learned_states | error |
|---|---|---|---|---|---|
| n12_072 | 0.2 | 0 | 0.984 | 12 | - |
