# Canonical HH→4b BDT train-only model choice

Status: `hh4b_bdt_train_only_model_choice_frozen`.

The primary nominal model is `global_v1_mass_aware`. Its weighted train-only OOF AUC is
0.7737142628735474; at the signal
efficiency matched to the optimized $R_{HH}<34$ cut, its weighted background
efficiency is
0.184305471151.
It provides substantially stronger background rejection than the optimized
cut while retaining the simpler global strategy.

The predeclared secondary model is `categorized_cms_inspired_mass_aware`, an advanced CMS-inspired
categorized alternative. Its weighted train-only OOF AUC values are
0.753744498261
in low $m_{HH}$ and
0.805324916529
in high $m_{HH}$. Its cut-matched combined weighted background efficiency is
0.183893493127.
The source-member bootstrap median v2-minus-global-v1 difference is
-0.000479674979924, with a 2.5--97.5 percentile
interval [-0.0134913838606,
0.0133359939886] and a v2-favoring fraction of
0.533. The negligible
nominal-point improvement is therefore not stable under this diagnostic.

The three requested ablations remain non-nominal diagnostics. The optimized
`r_hh_125_125 < 34` selection remains the cut baseline.

The next gate may fit the frozen primary, secondary, and global mass-plane-blind
diagnostic on the complete frozen train population and evaluate validation
exactly once, together with the optimized cut. Validation model selection and
post-validation hyperparameter tuning are not authorized. Test access is not
authorized.

This freeze gate opened no validation or test candidate file, trained no
model, wrote no prediction, ran no hyperparameter trial, used no physical event
weight, and calculated no physics yield or significance.
