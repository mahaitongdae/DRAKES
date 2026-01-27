# advantage-weighted behavior cloning for DNA sequence generation
import diffusion_gosai_update
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
from utils import str2bool, set_seed
import dataloader_gosai


def fine_tune_weighted_bc(new_model, new_model_y, new_model_y_eval, old_model, args, dataset_loader=None, eps=1e-5):
    """
    Advantage-weighted behavior cloning for DNA sequence generation.
    Uses the reward function as advantages to weight the behavior cloning loss.
    """
    
    with open(log_path, 'w') as f:
        f.write(args.__repr__() + '\n')

    # new_model.config.finetuning.truncate_steps = args.truncate_steps
    # new_model.config.finetuning.gumbel_softmax_temp = args.gumbel_temp
    dt = (1 - eps) / args.total_num_steps
    new_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.Adam(new_model.parameters(), lr=args.learning_rate)
    batch_losses = []
    batch_rewards = []
    
    
    # Initialize dataset iterator if using dataset samples
    dataset_iter = None
    if args.use_dataset_samples and dataset_loader is not None:
        dataset_iter = iter(dataset_loader)

    for epoch_num in range(args.num_epochs):
        rewards = []
        rewards_eval = []
        losses = []
        diffusion_losses = []
        # weighted_diffusion_losses = []
        batch_weights = []
        tot_grad_norm = 0.0
        new_model.train()
        
        for _step in range(args.num_accum_steps):
            # Sample sequences - either from dataset or from old model
            if args.use_dataset_samples and dataset_loader is not None:
                try:
                    batch = next(dataset_iter)
                except StopIteration:
                    # Restart iterator when we reach the end
                    dataset_iter = iter(dataset_loader)
                    batch = next(dataset_iter)
                
                # Get sequences from dataset
                sample_tokens = batch['seqs'].to(new_model.device)  # [bsz, seqlen]
                
                # Convert to one-hot format for reward computation
                # Keep the same format as the original: [bsz, seqlen, 4]
                sample = F.one_hot(sample_tokens, num_classes=4).float()  # [bsz, seqlen, 4]
                
                # For dataset samples, we don't have the diffusion process intermediates
                # So we'll use the tokens directly for loss computation
                sample_for_loss = sample_tokens
                
            else:
                # Sample from either old_model or new_model based on the parameter
                if args.bootstrap_from_new_model:
                    # Sample from new model (more aggressive bootstrapping)
                    sample_tokens = new_model._sample(eval_sp_size=args.batch_size)  # [bsz, seqlen] - discrete tokens
                else:
                    # Sample from old model (behavior policy) - no gradient accumulation needed
                    sample_tokens = old_model._sample(eval_sp_size=args.batch_size)  # [bsz, seqlen] - discrete tokens
                
                # Convert to one-hot format for reward computation
                sample = F.one_hot(sample_tokens, num_classes=4).float()  # [bsz, seqlen, 4]
                
                # Use tokens directly for loss computation
                sample_for_loss = sample_tokens
            
            sample2 = torch.transpose(sample, 1, 2)
            preds = new_model_y(sample2).squeeze(-1)  # [bsz, 3]
            reward = preds[:, 0]  # Use reward as advantage

            # Convert to argmax for evaluation
            sample_argmax = torch.argmax(sample, 2)
            sample_argmax = 1.0 * F.one_hot(sample_argmax, num_classes=4)
            sample_argmax = torch.transpose(sample_argmax, 1, 2)

            preds_argmax = new_model_y(sample_argmax).squeeze(-1)
            reward_argmax = preds_argmax[:, 0]
            rewards.append(reward_argmax.detach().cpu().numpy())
            
            preds_eval = new_model_y_eval(sample_argmax).squeeze(-1)
            reward_argmax_eval = preds_eval[:, 0]
            rewards_eval.append(reward_argmax_eval.detach().cpu().numpy())
            
            # Calculate reweighted diffusion loss
            # Create attention mask (assuming all positions are valid)
            attention_mask = torch.ones(sample_for_loss.shape, device=sample_for_loss.device)
            
            # Compute diffusion loss using the new model
            loss_result = new_model._loss(sample_for_loss, attention_mask)
            # diffusion_loss = loss_result.loss  # This is the token-level NLL
            diffusion_loss = loss_result.nlls.sum(axis=-1) / attention_mask.sum(axis=-1)  # [bsz]
            
            # Compute advantages and convert to weights
            advantages = reward.detach()  # [bsz]
            
            # Apply temperature scaling to advantages
            if args.advantage_temp > 0:
                advantages = advantages / args.advantage_temp
            
            # # Clip advantages for stability
            # if args.advantage_clip > 0:
            #     advantages = torch.clamp(advantages, -args.advantage_clip, args.advantage_clip)
            
            # Convert advantages to weights (positive advantages get higher weights)
            # Use softmax to normalize weights across batch
            weights = F.softmax(advantages, dim=0)  # [bsz]
            
            # Apply weights to diffusion loss
            # Since diffusion_loss is already averaged, we need to scale it by the weights
            weighted_diffusion_loss = torch.sum(diffusion_loss * weights)
            
            # For logging purposes, also compute unweighted diffusion loss
            unweighted_diffusion_loss = diffusion_loss
            
            # Final loss is the weighted diffusion loss
            loss = weighted_diffusion_loss
            
            # # Add optional KL regularization to prevent too much deviation from old model
            # if args.kl_weight > 0:
            #     # Compute KL divergence between new and old model on the same samples
            #     with torch.no_grad():
            #         old_loss_result = old_model._loss(sample_for_loss, attention_mask)
            #         old_diffusion_loss = old_loss_result.loss
            #     kl_loss = diffusion_loss - old_diffusion_loss  # Approximation of KL divergence
            #     loss = loss + args.kl_weight * kl_loss
            
            loss = loss / args.num_accum_steps
            
            loss.backward()
            if (_step + 1) % args.num_accum_steps == 0:  # Gradient accumulation
                norm = torch.nn.utils.clip_grad_norm_(new_model.parameters(), args.gradnorm_clip)
                tot_grad_norm += norm
                optim.step()
                optim.zero_grad()

            batch_losses.append(loss.cpu().detach().numpy())
            batch_rewards.append(torch.mean(reward).cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy() * args.num_accum_steps)
            diffusion_losses.append(unweighted_diffusion_loss.cpu().detach().numpy())
            # weighted_diffusion_losses.append(weighted_diffusion_loss.cpu().detach().numpy())
            batch_weights.append(weights.cpu().detach().numpy())

        rewards = np.array(rewards)
        rewards_eval = np.array(rewards_eval)
        losses = np.array(losses)
        diffusion_losses = np.array(diffusion_losses)
        # weighted_diffusion_losses = np.array(weighted_diffusion_losses)
        batch_weights = np.array(batch_weights)
        print("Epoch %d" % epoch_num, 
              "Mean reward %f" % np.mean(rewards), 
              "Mean reward eval %f" % np.mean(rewards_eval),
              "Mean grad norm %f" % tot_grad_norm, 
              "Mean loss %f" % np.mean(losses), 
              "Mean diffusion loss %f" % np.mean(diffusion_losses), 
            #   "Mean weighted diffusion loss %f" % np.mean(weighted_diffusion_losses),
              "Mean batch weight %f" % np.mean(batch_weights),
              "Max batch weight %f" % np.mean(np.max(batch_weights, axis=1))
        )
        
        if args.name != 'debug':
            wandb.log({
                "epoch": epoch_num, 
                "mean_reward": np.mean(rewards), 
                "mean_reward_eval": np.mean(rewards_eval),
                "mean_grad_norm": tot_grad_norm, 
                "mean_loss": np.mean(losses), 
                "mean_diffusion_loss": np.mean(diffusion_losses), 
                # "mean_weighted_diffusion_loss": np.mean(weighted_diffusion_losses),
                "mean_batch_weight": np.mean(batch_weights),
                "max_mean_batch_weight": np.mean(np.max(batch_weights, axis=1)),
                "min_mean_batch_weight": np.mean(np.min(batch_weights, axis=1)),
                "max_max_batch_weight": np.max(np.max(batch_weights, axis=1)),
                "min_min_batch_weight": np.min(np.min(batch_weights, axis=1))
            })
        
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch_num} Mean reward {np.mean(rewards)} Mean reward eval {np.mean(rewards_eval)} "
                   f"Mean grad norm {tot_grad_norm} Mean loss {np.mean(losses)} "
                   f"Mean diffusion loss {np.mean(diffusion_losses)}\n")
        
        if (epoch_num + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(save_path, f'model_{epoch_num}.ckpt')
            torch.save(new_model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch_num}")
    
    if args.name != 'debug':
        wandb.finish()

    return batch_losses


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='/n/netscratch/nali_lab_seas/Lab/haitongma/dna_data/data_and_model/')
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=100)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gradnorm_clip', type=float, default=10.0)
argparser.add_argument('--batch_size', type=int, default=128)
argparser.add_argument('--name', type=str, default='test')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=50)
argparser.add_argument('--advantage_temp', type=float, default=0.1, 
                       help='Temperature for scaling advantages before converting to weights')
# argparser.add_argument('--advantage_clip', type=float, default=10.0, 
#                        help='Clip advantages to this value for stability')
# argparser.add_argument('--kl_weight', type=float, default=0.0, 
#                        help='Weight for KL regularization to prevent too much deviation from old model')
argparser.add_argument('--use_dataset_samples', type=str2bool, default=False,
                       help='Whether to use samples from the original training dataset instead of bootstrapping from the model')
argparser.add_argument('--bootstrap_from_new_model', type=str2bool, default=False,
                       help='Whether to bootstrap from new_model instead of old_model (only applies when not using dataset samples)')
argparser.add_argument("--seed", type=int, default=0)
args = argparser.parse_args()
print(args)

# pretrained model path
CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
log_base_dir = os.path.join(args.base_path, 'mdlm/weighted_bc_results_final')

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# initialize a log file
if args.name == 'debug':
    print("Debug mode")
    save_path = os.path.join(log_base_dir, args.name)
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, 'log.txt')
else:
    if args.use_dataset_samples:
        dataset_suffix = '_dataset'
    elif args.bootstrap_from_new_model:
        dataset_suffix = '_bootstrap_new'
    else:
        dataset_suffix = '_bootstrap_old'
    run_name = f'{dataset_suffix}_adv_temp{args.advantage_temp}_accum{args.num_accum_steps}_bsz{args.batch_size}_clip{args.gradnorm_clip}_{args.name}_{curr_time}' # _kl{args.kl_weight}
    save_path = os.path.join(log_base_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    wandb.init(project='weighted_bc_final', name=run_name, config=args, dir=save_path)
    log_path = os.path.join(save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# Initialize the model
new_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
old_model = diffusion_gosai_update.Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg)
reward_model = oracle.get_gosai_oracle(mode='train').to(new_model.device)
reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(new_model.device)
reward_model.eval()
reward_model_eval.eval()

# Load dataset if using dataset samples
dataset_loader = None
if args.use_dataset_samples:
    print("Loading original training dataset for sampling...")
    train_dataset = dataloader_gosai.get_datasets_gosai()
    dataset_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

fine_tune_weighted_bc(new_model, reward_model, reward_model_eval, old_model, args, dataset_loader)
