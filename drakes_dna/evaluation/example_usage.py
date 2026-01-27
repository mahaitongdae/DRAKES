#!/usr/bin/env python3
"""
Example usage of the DNA Model Evaluator

This script demonstrates how to use the DNAModelEvaluator class
to evaluate a finetuned DNA generation model.
"""

from evaluate_model import DNAModelEvaluator

def example_evaluation():
    """Example of how to use the DNAModelEvaluator."""
    
    # Example model path (replace with your actual model path)
    model_path = "/path/to/your/finetuned.ckpt"
    
    # Create evaluator
    evaluator = DNAModelEvaluator(
        model_path=model_path,
        num_sample_batches=5,  # Generate 5 batches
        num_samples_per_batch=32  # 32 samples per batch
    )
    
    # Run all evaluations
    results = evaluator.run_all_evaluations()
    
    # Save results
    evaluator.save_results(results, "my_model_evaluation_results.npz")
    
    # Access specific results
    print(f"\nDetailed Results:")
    print(f"Pred-Activity median: {results['pred_activity']['pred_activity_median']:.4f}")
    print(f"ATAC Accuracy: {results['atac_acc']['atac_accuracy']:.4f}")
    print(f"3-mer Correlation: {results['kmer_correlation']['kmer_correlation']:.4f}")
    
    if results['jasper_motifs']:
        print(f"JASPER Motif Correlation: {results['jasper_motifs']['jasper_spearman_corr']:.4f}")
    
    print(f"Likelihood median: {results['likelihood']['likelihood_median']:.4f}")

if __name__ == "__main__":
    print("This is an example script. Please modify the model_path variable")
    print("to point to your actual finetuned model checkpoint.")
    print("\nTo run the evaluation directly, use:")
    print("python evaluate_model.py --model_path /path/to/your/finetuned.ckpt")
