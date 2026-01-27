# DNA Model Evaluation Script

This directory contains a comprehensive evaluation script for finetuned DNA generation models. The script evaluates models using five key metrics that are commonly used in DNA sequence generation research.

## Overview

The evaluation script (`evaluate_model.py`) provides a unified interface to run all five evaluation metrics:

1. **Pred-Activity based on eval oracle** - Evaluates the predicted activity of generated sequences using a pre-trained oracle model
2. **ATAC-Acc** - Measures the accuracy of ATAC-seq predictions for generated sequences
3. **3-mer Pearson Correlation** - Calculates the correlation between k-mer frequencies in generated sequences and high-expression training sequences
4. **JASPER Motif Analysis** - Analyzes the presence of transcription factor binding motifs using the JASPAR database
5. **Likelihood** - Computes the likelihood of generated sequences under the pretrained model

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- SciPy
- tqdm
- matplotlib
- Hydra
- grelu (for JASPER motif analysis)

## Usage

### Command Line Interface

The simplest way to run the evaluation is using the command line interface:

```bash
python evaluate_model.py --model_path /path/to/your/finetuned.ckpt
```

#### Command Line Options

- `--model_path`: **Required**. Path to the finetuned model checkpoint file
- `--num_batches`: Number of sample batches to generate (default: 10)
- `--samples_per_batch`: Number of samples per batch (default: 64)
- `--output`: Output file path for results (default: evaluation_results.npz)

#### Examples

```bash
# Basic evaluation with default parameters
python evaluate_model.py --model_path ./models/finetuned.ckpt

# Custom number of samples
python evaluate_model.py --model_path ./models/finetuned.ckpt --num_batches 20 --samples_per_batch 32

# Custom output file
python evaluate_model.py --model_path ./models/finetuned.ckpt --output my_results.npz
```

### Programmatic Usage

You can also use the evaluator programmatically:

```python
from evaluate_model import DNAModelEvaluator

# Create evaluator
evaluator = DNAModelEvaluator(
    model_path="/path/to/finetuned.ckpt",
    num_sample_batches=10,
    num_samples_per_batch=64
)

# Run all evaluations
results = evaluator.run_all_evaluations()

# Access specific results
pred_activity = results['pred_activity']['pred_activity_median']
atac_acc = results['atac_acc']['atac_accuracy']
kmer_corr = results['kmer_correlation']['kmer_correlation']

# Save results
evaluator.save_results(results, "my_results.npz")
```

## Output

The script provides:

1. **Console output** with progress bars and real-time results
2. **Summary table** with all five metrics
3. **Saved results** in NPZ format containing all raw data

### Console Output Example

```
============================================================
Starting DNA Model Evaluation
============================================================
Loading models...
Models loaded successfully!
Loading high-expression k-mers...
Generating 640 samples...
Generating samples: 100%|██████████| 10/10 [00:15<00:00,  1.52s/it]
Generated 640 samples
Evaluating pred-activity based on eval oracle...
Pred-Activity (median): 4.2345
Evaluating ATAC-Acc...
ATAC-Acc: 0.6781
Evaluating 3-mer Pearson correlation...
3-mer Pearson Correlation: 0.8234 (p-value: 1.23e-45)
Evaluating JASPER motif analysis...
JASPER Motif Spearman Correlation: 0.7456
Evaluating likelihood...
Likelihood (median): -2.3456

============================================================
EVALUATION SUMMARY
============================================================
1. Pred-Activity (median): 4.2345
2. ATAC-Acc: 0.6781
3. 3-mer Pearson Correlation: 0.8234
4. JASPER Motif Spearman Correlation: 0.7456
5. Likelihood (median): -2.3456
============================================================
```

### Saved Results

The NPZ file contains:
- `pred_activity_median`: Median pred-activity score
- `pred_activity_all`: All pred-activity scores
- `atac_accuracy`: ATAC accuracy score
- `atac_predictions`: All ATAC predictions
- `kmer_correlation`: 3-mer Pearson correlation
- `kmer_p_value`: P-value for correlation
- `likelihood_median`: Median likelihood score
- `likelihood_all`: All likelihood scores
- `jasper_spearman_corr`: JASPER motif correlation (if available)

## Configuration

The script automatically loads the configuration from `configs_gosai/config_gosai.yaml`. Make sure this file exists and contains the necessary model configuration.

## Dependencies

The script expects the following directory structure:
```
drakes_dna/
├── evaluate_model.py          # Main evaluation script
├── oracle.py                  # Oracle functions for evaluation
├── diffusion_gosai_update.py # Diffusion model implementation
├── dataloader_gosai.py       # Data loading utilities
├── utils.py                   # Utility functions
├── configs_gosai/            # Configuration files
│   └── config_gosai.yaml
└── models/                    # Model checkpoints
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--num_batches` or `--samples_per_batch`
2. **Model not found**: Check the `--model_path` argument
3. **JASPER analysis unavailable**: Install the `grelu` package
4. **Configuration errors**: Ensure `configs_gosai/config_gosai.yaml` exists

### Performance Tips

- Use smaller batch sizes if memory is limited
- The script automatically uses GPU if available
- Results are cached and can be reused

## Example Workflow

1. **Train your model** and save the checkpoint
2. **Run evaluation**:
   ```bash
   python evaluate_model.py --model_path ./checkpoints/finetuned.ckpt
   ```
3. **Analyze results** from the console output and saved NPZ file
4. **Compare models** by running the script on different checkpoints

## Citation

If you use this evaluation script in your research, please cite the original DRAKES paper and include a reference to this evaluation framework.
