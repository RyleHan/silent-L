# Main Figure Caption

**From decodable world state to closed-loop control.** (a) Paired-prompt
probing reveals instruction-conditioned target geometry and task phase in both
OpenVLA and pi0.5, including the pi0.5 action expert. Values are locked-test
R2 or macro-F1. (b) Replacing pi0.5 action-expert block 13 with the natural
same-state counterfactual residual recovers about 22% of the prompt-induced
action change, but the supervised goal-probe subspace explains almost none of
that effect. (c) The single-block intervention creates a path-incompatible
hybrid computation and fails closed-loop target switching, whereas replaying
all 18 prompt-conditioned action-expert blocks exactly reproduces the source
policy and serves as an operator positive control. (d) A paper-guided COAST
conceptor reimplementation yields a positive but underpowered 13.3 percentage
point held-out gain on LIBERO-10 KS3; the paper's reported 53% to 93% result is
shown separately because simulator seeds and implementation are not matched.
Together, the results separate decodability, local causal influence, pathway
compatibility, and useful closed-loop control.
