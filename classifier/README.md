## Dependencies

```bash
conda create -n phd-ml
conda activate phd-ml
conda install -c pytorch -c conda-forge pytorch=1.6.0 cudatoolkit=10.2 
conda install -c pytorch -c conda-forge xlrd=1.2.0 pandas matplotlib jupyterlab ipykernel tensorboard
pip3 install scikit-learn transformers==3.5.0
```
## Training with leave-one-out cross-validation
`python3 training_strategy.py`

## Predict with fine-tuned model
1. Put the test set in the same format as the files in `data` under the folder `testset/set` (one dialog per file).
2. Run `python3 predict_strategy.py`
