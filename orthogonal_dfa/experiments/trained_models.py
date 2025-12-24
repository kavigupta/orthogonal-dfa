from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, Tuple

from orthogonal_dfa.experiments.gate_experiments import (
    train_psam_linear,
    train_psam_linear_on_others,
    train_psam_linear_with_alternates,
    train_rnn_direct,
    train_rnn_psams,
)
from orthogonal_dfa.experiments.training_curves import pdfa_results, process_results
from orthogonal_dfa.module.rnn import LSTMProcessor
from orthogonal_dfa.utils.pdfa import PDFAHyberbolicParameterization


@dataclass
class TrainedModels:
    name: str
    get_results: Callable[[], Tuple]

    @cached_property
    def results(self):
        return self.get_results()


@lru_cache(None)
def gates_psams_orig():
    gates, _, _ = train_psam_linear(11)
    return gates


@lru_cache(None)
def gates_psams():
    gates, _, _ = train_psam_linear_with_alternates(10)
    return gates


r_pdfa = TrainedModels("PDFA", lambda: pdfa_results(10, []))
r_pdfa_cp = TrainedModels(
    "PDFA | PSAMs [orig]", lambda: pdfa_results(10, gates_psams_orig())
)
r_pdfa_10s = TrainedModels(
    "PDFA [10 states]", lambda: pdfa_results(10, [], num_states=10)
)
r_pdfa_cp_10s = TrainedModels(
    "PDFA [10 states] | PSAMs [orig]",
    lambda: pdfa_results(10, gates_psams_orig(), num_states=10),
)
r_pdfa_hyp = TrainedModels(
    "PDFA [hyp]", lambda: pdfa_results(10, [], pdfa_typ=PDFAHyberbolicParameterization)
)
r_pdfa_cp_hyp = TrainedModels(
    "PDFA [hyp] | PSAMs [orig]",
    lambda: pdfa_results(
        10, gates_psams_orig(), pdfa_typ=PDFAHyberbolicParameterization
    ),
)
r_psam_lin = TrainedModels(
    "PSAM Linear | PSAMs [orig]",
    lambda: process_results(
        [train_psam_linear_on_others(11, seed) for seed in range(10)]
    ),
)
r_rnn = TrainedModels(
    "RNN Direct",
    lambda: process_results([train_rnn_direct(seed) for seed in range(10)]),
)
r_rnn_500 = TrainedModels(
    "RNN Direct [500]",
    lambda: process_results(
        [train_rnn_direct(seed, hidden_size=500) for seed in range(6)]
    ),
)
r_rnn_500_1l = TrainedModels(
    "RNN Direct [500, 1 layer]",
    lambda: process_results(
        [train_rnn_direct(seed, hidden_size=500, layers=1) for seed in range(10)]
    ),
)
r_rnn_500_1l_cp = TrainedModels(
    "RNN Direct [500, 1 layer] | PSAMs",
    lambda: process_results(
        [train_rnn_direct(seed, hidden_size=500, layers=1, starting_gates=gates_psams()) for seed in range(4)]
    ),
)
r_rnn_psams_3 = TrainedModels(
    "RNN PSAMs [noise=3]",
    lambda: process_results([train_rnn_psams(seed, 3) for seed in range(10)]),
)
r_rnn_psams_4 = TrainedModels(
    "RNN PSAMs [noise=4]",
    lambda: process_results([train_rnn_psams(seed, 4) for seed in range(10)]),
)
r_rnn_500_1l_psams_3 = TrainedModels(
    "RNN PSAMs [noise=3] [500, 1 layer]",
    lambda: process_results(
        [train_rnn_psams(seed, 3, hidden_size=500, layers=1) for seed in range(10)]
    ),
)
r_rnn_500_1l_psams_4 = TrainedModels(
    "RNN PSAMs [noise=4] [500, 1 layer]",
    lambda: process_results(
        [train_rnn_psams(seed, 4, hidden_size=500, layers=1) for seed in range(10)]
    ),
)
r_rnn_500_1l_psams_3_cp = TrainedModels(
    "RNN PSAMs [noise=3] [500, 1 layer] | PSAMs",
    lambda: process_results(
        [
            train_rnn_psams(
                seed, 3, hidden_size=500, layers=1, starting_gates=gates_psams()
            )
            for seed in range(5)
        ]
    ),
)
r_rnn_500_1l_psams_4_cp = TrainedModels(
    "RNN PSAMs [noise=4] [500, 1 layer] | PSAMs",
    lambda: process_results(
        [
            train_rnn_psams(
                seed, 4, hidden_size=500, layers=1, starting_gates=gates_psams()
            )
            for seed in range(5)
        ]
    ),
)
r_lstm = TrainedModels(
    "LSTM Direct",
    lambda: process_results(
        [train_rnn_direct(seed, constructor=LSTMProcessor) for seed in range(6)]
    ),
)
r_lstm_500 = TrainedModels(
    "LSTM Direct [500]",
    lambda: process_results(
        [
            train_rnn_direct(seed, constructor=LSTMProcessor, hidden_size=500)
            for seed in range(3)
        ]
    ),
)

pdfa_models_basic = [r_pdfa, r_pdfa_cp, r_pdfa_10s]
pdfa_models_rest = [r_pdfa_cp_10s, r_pdfa_hyp, r_pdfa_cp_hyp]

pdfa_models = pdfa_models_basic + pdfa_models_rest

psam_models = [r_psam_lin]

rnn_models_hyperparam_search = [
    r_rnn,
    r_rnn_psams_3,
    r_rnn_psams_4,
    r_lstm,
    r_lstm_500,
]

rnn_models_main = [
    r_rnn_500,
    r_rnn_500_1l,
    r_rnn_500_1l_psams_3,
    r_rnn_500_1l_psams_4,
    r_rnn_500_1l_cp,
    r_rnn_500_1l_psams_3_cp,
    r_rnn_500_1l_psams_4_cp,
]

all_models = pdfa_models + psam_models + rnn_models_hyperparam_search + rnn_models_main
