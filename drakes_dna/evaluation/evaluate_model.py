#!/usr/bin/env python3
"""
DNA Model Evaluation Script

This script evaluates a finetuned DNA generation model using five metrics:
1. Pred-Activity based on eval oracle
2. ATAC-Acc
3. 3-mer Pearson Correlation
4. JASPER Motif Analysis
5. Likelihood

Usage:
    python evaluate_model.py --model_path /path/to/finetuned.ckpt [--use_autoregressive]
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import json

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import dataloader_gosai
import oracle
from utils import set_seed

# Import JASPER motif analysis
try:
    from grelu.interpret.motifs import scan_sequences
    JASPER_AVAILABLE = True
except ImportError:
    print("Warning: JASPER motif analysis not available. Install grelu package.")
    JASPER_AVAILABLE = False

# Set random seed for reproducibility
set_seed(0, use_cuda=True)

# Set matplotlib parameters
plt.rcParams['figure.dpi'] = 200

class DNAModelEvaluator:
    """Evaluates DNA generation models using multiple metrics."""
    
    def __init__(self, model_path, num_sample_batches=10, num_samples_per_batch=64, num_steps=128, use_autoregressive=False):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the finetuned model checkpoint
            num_sample_batches (int): Number of batches to sample
            num_samples_per_batch (int): Number of samples per batch
            num_steps (int): Number of steps for sampling and likelihood calculation
            use_autoregressive (bool): Whether to use autoregressive sampler
        """
        self.model_path = model_path
        self.num_sample_batches = num_sample_batches
        self.num_samples_per_batch = num_samples_per_batch
        self.num_steps = num_steps
        self.use_autoregressive = use_autoregressive
        
        # Load models
        self._load_models()
        
        # Load high-expression k-mers for comparison
        self._load_high_exp_kmers()
        
    def _load_models(self):
        """Load the finetuned model and reference models."""
        print("Loading models...")
        
        # Reinitialize Hydra
        GlobalHydra.instance().clear()
        
        # Get the directory where this script is located and change to the drakes_dna directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        drakes_dna_dir = os.path.join(script_dir, "..")
        
        # Verify the drakes_dna directory and configs exist
        if not os.path.exists(drakes_dna_dir):
            raise FileNotFoundError(f"Drakes DNA directory not found: {drakes_dna_dir}")
        
        config_dir = os.path.join(drakes_dna_dir, "configs_gosai")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        print(f"Script directory: {script_dir}")
        print(f"Drakes DNA directory: {drakes_dna_dir}")
        print(f"Config directory: {config_dir}")
        print(f"Current working directory: {os.getcwd()}")
        
        # Change to the drakes_dna directory to ensure relative paths work correctly
        original_cwd = os.getcwd()
        os.chdir(drakes_dna_dir)
        print(f"Changed to working directory: {os.getcwd()}")
        
        try:
            # Initialize Hydra and compose the configuration using relative path
            initialize(config_path="configs_gosai", job_name="load_model")
            self.cfg = compose(config_name="config_gosai.yaml")
            self.cfg.eval.checkpoint_path = self.model_path
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
            print(f"Restored working directory: {os.getcwd()}")
        
        # Load finetuned model
        self.model = diffusion_gosai_update.Diffusion(self.cfg, eval=False).cuda()
        self.model.load_state_dict(torch.load(self.cfg.eval.checkpoint_path))
        self.model.eval()
        
        # Override parameterization if autoregressive sampling is requested
        if self.use_autoregressive:
            self.model.parameterization = 'ar'
        
        # Load pretrained model for likelihood comparison
        base_path = '/n/netscratch/nali_lab_seas/Lab/haitongma/dna_data/data_and_model/'
        old_path = os.path.join(base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
        self.old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(old_path, config=self.cfg)
        self.old_model.eval()
        
        print("Models loaded successfully!")
        
    def _load_high_exp_kmers(self):
        """Load high-expression k-mers for comparison."""
        print("Loading high-expression k-mers...")
        self.highexp_kmers_999, self.n_highexp_kmers_999, _, _, _, _, _ = oracle.cal_highexp_kmers(return_clss=True)
        
    def generate_samples(self):
        """Generate samples from the finetuned model."""
        print(f"Generating {self.num_sample_batches * self.num_samples_per_batch} samples...")
        
        all_detokenized_samples = []
        all_raw_samples = []
        
        for _ in tqdm(range(self.num_sample_batches), desc="Generating samples"):
            samples = self.model._sample(eval_sp_size=self.num_samples_per_batch, num_steps=self.num_steps)
            all_raw_samples.append(samples)
            detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy())
            all_detokenized_samples.extend(detokenized_samples)
            
        self.all_raw_samples = torch.concat(all_raw_samples)
        self.all_detokenized_samples = all_detokenized_samples
        
        print(f"Generated {len(self.all_detokenized_samples)} samples")
        
    def evaluate_pred_activity(self):
        """Evaluate pred-activity based on eval oracle."""
        print("Evaluating pred-activity based on eval oracle...")
        
        # Calculate predictions using the eval oracle
        generated_preds = oracle.cal_gosai_pred_new(self.all_detokenized_samples, mode='eval')
        
        # Calculate median prediction
        median_pred = np.median(generated_preds[:, 0])
        
        print(f"Pred-Activity (median): {median_pred:.4f}")
        return {
            'pred_activity_median': median_pred,
            'pred_activity_all': generated_preds[:, 0]
        }
        
    def evaluate_atac_acc(self):
        """Evaluate ATAC-Acc."""
        print("Evaluating ATAC-Acc...")
        
        # Calculate ATAC predictions
        generated_preds_atac = oracle.cal_atac_pred_new(self.all_detokenized_samples)
        
        # Calculate accuracy (percentage of sequences with ATAC > 0.5)
        atac_accuracy = (generated_preds_atac[:, 1] > 0.5).sum() / len(generated_preds_atac)
        
        print(f"ATAC-Acc: {atac_accuracy:.4f}")
        return {
            'atac_accuracy': atac_accuracy,
            'atac_predictions': generated_preds_atac[:, 1]
        }
        
    def evaluate_kmer_correlation(self):
        """Evaluate 3-mer Pearson correlation."""
        print("Evaluating 3-mer Pearson correlation...")
        
        # Count k-mers in generated sequences
        generated_kmer = oracle.count_kmers(self.all_detokenized_samples)
        
        # Calculate correlation with high-expression k-mers
        kmer_set = set(self.highexp_kmers_999.keys()) | set(generated_kmer.keys())
        counts = np.zeros((len(kmer_set), 2))
        
        for i, kmer in enumerate(kmer_set):
            if kmer in self.highexp_kmers_999:
                counts[i][1] = self.highexp_kmers_999[kmer] * len(generated_kmer) / self.n_highexp_kmers_999
            if kmer in generated_kmer:
                counts[i][0] = generated_kmer[kmer]
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(counts[:, 0], counts[:, 1])
        
        print(f"3-mer Pearson Correlation: {correlation:.4f} (p-value: {p_value:.2e})")
        return {
            'kmer_correlation': correlation,
            'kmer_p_value': p_value,
            'kmer_counts': counts
        }
        
    def evaluate_jasper_motifs(self):
        """Evaluate JASPER motif analysis."""
        if not JASPER_AVAILABLE:
            print("Skipping JASPER motif analysis (grelu not available)")
            return None
            
        print("Evaluating JASPER motif analysis...")
        
        # Scan sequences for motifs
        motif_count = scan_sequences(self.all_detokenized_samples, 'jaspar')
        motif_count_sum = motif_count['motif'].value_counts()
        
        # Also scan high-expression sequences for comparison
        _, _, _, _, _, _, highexp_seqs_999 = oracle.cal_highexp_kmers(return_clss=True)
        motif_count_top = scan_sequences(highexp_seqs_999, 'jaspar')
        motif_count_top_sum = motif_count_top['motif'].value_counts()
        
        # Calculate Spearman correlation between motif counts
        motifs_summary = pd.concat([motif_count_top_sum, motif_count_sum], axis=1)
        motifs_summary.columns = ['top_data', 'finetuned']
        motifs_summary = motifs_summary.fillna(0)
        
        # Calculate correlation
        spearman_corr = motifs_summary.corr(method='spearman').iloc[0, 1]
        
        print(f"JASPER Motif Spearman Correlation: {spearman_corr:.4f}")
        return {
            'jasper_spearman_corr': spearman_corr,
            'motif_counts': motif_count_sum,
            'motif_summary': motifs_summary
        }
        
    def evaluate_likelihood(self):
        """Evaluate likelihood."""
        print("Evaluating likelihood...")
        
        # Calculate likelihood using the pretrained model
        model_logl = self.old_model.get_likelihood(self.all_raw_samples, num_steps=self.num_steps, n_samples=1)
        
        # Calculate median likelihood
        median_likelihood = torch.median(model_logl).item()
        
        print(f"Likelihood (median): {median_likelihood:.4f}")
        return {
            'likelihood_median': median_likelihood,
            'likelihood_all': model_logl.detach().cpu().numpy()
        }
        
    def run_all_evaluations(self):
        """Run all evaluation metrics."""
        print("=" * 60)
        print("Starting DNA Model Evaluation")
        print("=" * 60)
        
        # Generate samples
        self.generate_samples()
        
        # Run all evaluations
        results = {}
        
        # 1. Pred-Activity
        results['pred_activity'] = self.evaluate_pred_activity()
        
        # 2. ATAC-Acc
        results['atac_acc'] = self.evaluate_atac_acc()
        
        # 3. 3-mer Pearson Correlation
        results['kmer_correlation'] = self.evaluate_kmer_correlation()
        
        # 4. JASPER Motif Analysis
        results['jasper_motifs'] = self.evaluate_jasper_motifs()
        
        # 5. Likelihood
        results['likelihood'] = self.evaluate_likelihood()
        
        # Print summary
        self._print_summary(results)
        
        return results
        
    def _print_summary(self, results):
        """Print a summary of all evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"1. Pred-Activity (median): {results['pred_activity']['pred_activity_median']:.4f}")
        print(f"2. ATAC-Acc: {results['atac_acc']['atac_accuracy']:.4f}")
        print(f"3. 3-mer Pearson Correlation: {results['kmer_correlation']['kmer_correlation']:.4f}")
        print(f"4. JASPER Motif Spearman Correlation: {results['jasper_motifs']['jasper_spearman_corr']:.4f}" if results['jasper_motifs'] else "4. JASPER Motif Analysis: Not available")
        print(f"5. Likelihood (median): {results['likelihood']['likelihood_median']:.4f}")
        print("=" * 60)
        
    def save_results(self, results, output_path="evaluation_results.npz"):
        """Save evaluation results to both NPZ and CSV files."""
        print(f"Saving results to {output_path}...")
        
        # Create results folder in the checkpoint directory with checkpoint name
        checkpoint_dir = os.path.dirname(self.model_path)
        checkpoint_name = os.path.splitext(os.path.basename(self.model_path))[0]
        results_dir = os.path.join(checkpoint_dir, f"evaluation_{checkpoint_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'pred_activity_median': results['pred_activity']['pred_activity_median'],
            'pred_activity_all': results['pred_activity']['pred_activity_all'],
            'atac_accuracy': results['atac_acc']['atac_accuracy'],
            'atac_predictions': results['atac_acc']['atac_predictions'],
            'kmer_correlation': results['kmer_correlation']['kmer_correlation'],
            'kmer_p_value': results['kmer_correlation']['kmer_p_value'],
            'likelihood_median': results['likelihood']['likelihood_median'],
            'likelihood_all': results['likelihood']['likelihood_all']
        }
        
        # Add JASPER results if available
        if results['jasper_motifs']:
            save_data['jasper_spearman_corr'] = results['jasper_motifs']['jasper_spearman_corr']
        
        # Save NPZ file
        npz_path = os.path.join(results_dir, output_path)
        np.savez(npz_path, **save_data)
        print(f"NPZ results saved to {npz_path}")
        
        # Save summary CSV with metadata
        self._save_summary_csv(results, results_dir)
        
        # Save generated sequences
        self._save_sequences_csv(results_dir)
        
    def _save_summary_csv(self, results, results_dir):
        """Save summary results as CSV with metadata."""
        # Get current timestamp
        evaluation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create summary data
        summary_data = {
            'metric': [
                'pred_activity_median',
                'atac_accuracy', 
                'kmer_correlation',
                'kmer_p_value',
                'likelihood_median'
            ],
            'value': [
                results['pred_activity']['pred_activity_median'],
                results['atac_acc']['atac_accuracy'],
                results['kmer_correlation']['kmer_correlation'],
                results['kmer_correlation']['kmer_p_value'],
                results['likelihood']['likelihood_median']
            ]
        }
        
        # Add JASPER result if available
        if results['jasper_motifs']:
            summary_data['metric'].append('jasper_spearman_corr')
            summary_data['value'].append(results['jasper_motifs']['jasper_spearman_corr'])
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Add metadata
        metadata = {
            'evaluation_time': [evaluation_time],
            'model_path': [self.model_path],
            'model_steps': [self.num_steps],
            'sample_steps': [self.num_steps],  # Same as model_steps for this implementation
            'num_sample_batches': [self.num_sample_batches],
            'num_samples_per_batch': [self.num_samples_per_batch],
            'total_samples': [self.num_sample_batches * self.num_samples_per_batch]
        }
        
        metadata_df = pd.DataFrame(metadata)
        
        # Save summary CSV
        summary_csv_path = os.path.join(results_dir, 'evaluation_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary CSV saved to {summary_csv_path}")
        
        # Save metadata CSV
        metadata_csv_path = os.path.join(results_dir, 'evaluation_metadata.csv')
        metadata_df.to_csv(metadata_csv_path, index=False)
        print(f"Metadata CSV saved to {metadata_csv_path}")
        
        # Save combined summary with metadata
        combined_summary = pd.concat([metadata_df, summary_df], axis=1)
        combined_csv_path = os.path.join(results_dir, 'evaluation_results_combined.csv')
        combined_summary.to_csv(combined_csv_path, index=False)
        print(f"Combined results CSV saved to {combined_csv_path}")
        
    def _save_sequences_csv(self, results_dir):
        """Save generated DNA sequences as CSV file."""
        if hasattr(self, 'all_detokenized_samples'):
            sequences_df = pd.DataFrame({
                'sample_id': range(len(self.all_detokenized_samples)),
                'dna_sequence': self.all_detokenized_samples
            })
            sequences_csv_path = os.path.join(results_dir, 'generated_sequences.csv')
            sequences_df.to_csv(sequences_csv_path, index=False)
            print(f"Generated sequences saved to {sequences_csv_path}")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a finetuned DNA generation model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the finetuned model checkpoint')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of sample batches (default: 10)')
    parser.add_argument('--samples_per_batch', type=int, default=64,
                       help='Number of samples per batch (default: 64)')
    parser.add_argument('--num_steps', type=int, default=128,
                       help='Number of steps for sampling and likelihood calculation (default: 128)')
    parser.add_argument('--output', type=str, default='evaluation_results.npz',
                       help='Output file path for results (default: evaluation_results.npz)')
    parser.add_argument('--use_autoregressive', action='store_true',
                       help='Use autoregressive sampler instead of diffusion sampler')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    try:
        # Create evaluator and run evaluations
        evaluator = DNAModelEvaluator(
            model_path=args.model_path,
            num_sample_batches=args.num_batches,
            num_samples_per_batch=args.samples_per_batch,
            num_steps=args.num_steps,
            use_autoregressive=args.use_autoregressive
        )
        
        # Run all evaluations
        results = evaluator.run_all_evaluations()
        
        # Save results
        evaluator.save_results(results, args.output)
        
        print("\nEvaluation completed successfully!")
        print(f"Results saved to the 'evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}' folder in the checkpoint directory:")
        print(f"  - NPZ file: {os.path.dirname(args.model_path)}/evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}/{args.output}")
        print(f"  - Summary CSV: {os.path.dirname(args.model_path)}/evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}/evaluation_summary.csv")
        print(f"  - Metadata CSV: {os.path.dirname(args.model_path)}/evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}/evaluation_metadata.csv")
        print(f"  - Combined results: {os.path.dirname(args.model_path)}/evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}/evaluation_results_combined.csv")
        print(f"  - Generated sequences: {os.path.dirname(args.model_path)}/evaluation_{os.path.splitext(os.path.basename(args.model_path))[0]}/generated_sequences.csv")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
