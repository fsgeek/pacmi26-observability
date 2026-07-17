.PHONY: figures traces clean help

help:
	@echo "make figures  - regenerate all three paper figures from committed data (no model weights)"
	@echo "make traces   - re-run the model to regenerate the entropy trace (needs weights/GPU; pip install .[traces])"
	@echo "make clean    - remove regenerated figure outputs"

figures:
	python scripts/fig1_entropy_trace.py
	python scripts/fig2_confidence_distributions.py
	python scripts/fig3_budget_curve.py

traces:
	python scripts/generate_trace.py

clean:
	rm -f figures/fig1_entropy_trace.tex \
	      figures/fig2_confidence_distributions.pdf figures/fig2_confidence_distributions.png \
	      figures/fig3_budget_curve.pdf figures/fig3_budget_curve.png
