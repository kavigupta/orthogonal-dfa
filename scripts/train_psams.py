from orthogonal_dfa.spliceai.best_psams_for_lssi import train_and_evaluate_lssi_psams

don = train_and_evaluate_lssi_psams("donor", num_batches=100_000, lr=3e-4)
acc = train_and_evaluate_lssi_psams("acceptor", num_batches=100_000, lr=3e-4)
