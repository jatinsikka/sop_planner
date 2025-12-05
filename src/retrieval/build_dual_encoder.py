from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from rich.console import Console
from datasets import Dataset

# Add parent directory to path to enable imports from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.schemas import SOPEntry, load_jsonl
from src.retrieval.index_utils import build_and_save_index

# Console instance for logging output with rich formatting
console = Console()


def load_config(config_path: str | Path) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        console.log("[yellow]PyYAML not installed, using default config[/yellow]")
        return {}
    except FileNotFoundError:
        console.log(f"[yellow]Config file not found: {config_path}, using defaults[/yellow]")
        return {}


# Combines SOP (Standard Operating Procedure) information into a single text representation
def synthesize_text(sop: SOPEntry) -> str:
    """
    Convert a SOPEntry into a synthesized text string for encoding.
    
    Args:
        sop: SOPEntry object containing title, condition, and steps
        
    Returns:
        A formatted string combining title, condition, and steps separated by semicolons
    """
    return f"{sop.title}. {sop.condition}. Steps: " + " ; ".join(sop.steps)


def train_dual_encoder(args: argparse.Namespace) -> None:
    """
    Train a dual-encoder model using InfoNCE loss and build a FAISS index for SOP retrieval.
    
    This function:
    1. Loads SOP data from JSONL
    2. Synthesizes text from SOP entries
    3. Trains separate query and passage encoders using contrastive learning
    4. Saves the trained encoders and tokenizer
    5. Builds a FAISS(Facebook AI Similarity Search) index for efficient retrieval 
    
    Args:
        args: Command-line arguments containing:
            - out_dir: Output directory for saving models and index
            - train_jsonl: Path to JSONL file with SOP examples
            - model_name: Pre-trained model name (default: bert-base-uncased)
            - epochs: Number of training epochs
    """
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SOP data from JSONL file
    sops = load_jsonl(args.train_jsonl, SOPEntry)
    
    # Synthesize text representations for each SOP
    texts = [synthesize_text(s) for s in sops]
    
    # Extract SOP IDs for indexing
    ids = [s.sop_id for s in sops]

    # Try real dual-encoder training; fallback to no-train and index only
    try:
        # Import deep learning libraries
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        # Model setup
        model_name = args.model_name
        console.log(f"[dim]Loading tokenizer: {model_name}[/dim]")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        console.log(f"[dim]Loading model (this may download weights on first run)...[/dim]")
        # Initialize two separate encoders: one for queries (incidents), one for passages (SOPs)
        model_q = AutoModel.from_pretrained(model_name)
        model_p = AutoModel.from_pretrained(model_name)

        # Collate function to prepare batches of text for the model
        def collate(batch):
            # batch is a list of dicts from Dataset, extract "text" field
            texts = [item["text"] for item in batch]
            # Tokenize texts with padding and truncation
            return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # Create dataset from texts
        ds = Dataset.from_dict({"text": texts})
        
        # Create data loader with configurable batch size and shuffling
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        
        # Optimizer: update parameters of both query and passage encoders
        opt = torch.optim.AdamW(list(model_q.parameters()) + list(model_p.parameters()), lr=args.lr)
        
        # Learning rate scheduler: linear warmup then decay
        total_steps = max(1, len(dl) * args.epochs)
        sched = get_linear_schedule_with_warmup(opt, 0, total_steps)
        
        # Loss function: cross-entropy for contrastive learning (InfoNCE - score contrastive tasks)
        loss_fn = nn.CrossEntropyLoss()
        
        # Set models to training mode
        model_q.train()
        model_p.train()
        
        # Move models to GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_q.to(device)
        model_p.to(device)
        
        # Training loop: iterate over epochs and batches
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in dl:
                # Move batch to device
                for k in batch:
                    batch[k] = batch[k].to(device)
                
                # Forward pass: get [CLS] token embeddings (contextualized sentence representation)
                q = model_q(**batch).last_hidden_state[:, 0, :]  # [CLS] token for queries
                p = model_p(**batch).last_hidden_state[:, 0, :]  # [CLS] token for passages
                
                # L2 normalization for embeddings
                q = q / (q.norm(dim=1, keepdim=True) + 1e-9)
                p = p / (p.norm(dim=1, keepdim=True) + 1e-9)
                
                # Compute similarity scores: dot product between all query-passage pairs
                scores = q @ p.t()
                
                # Labels: diagonal elements (positive pairs) should have highest scores
                labels = torch.arange(scores.size(0), device=device)
                
                # Compute InfoNCE loss
                loss = loss_fn(scores, labels)
                
                # Backward pass and optimization step
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                
                # Track loss for logging
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            console.log(f"[dim]Epoch {epoch+1}/{args.epochs}: avg_loss = {avg_loss:.4f}[/dim]")
        
        # Save trained encoders and tokenizer in HuggingFace format
        model_q.save_pretrained(str(out_dir / "incident_encoder"))
        model_p.save_pretrained(str(out_dir / "sop_encoder"))
        tokenizer.save_pretrained(str(out_dir / "tokenizer"))
        console.log("Saved dual encoders.")
    except Exception as e:
        # Graceful fallback if training fails (e.g., missing dependencies)
        import traceback
        console.log(f"[yellow]Training skipped due to: {type(e).__name__}: {e}[/yellow]")
        console.log(f"[dim]{traceback.format_exc()}[/dim]")

    # Always build FAISS index from SOP texts for retrieval
    build_and_save_index(texts, ids, out_dir / "index")
    console.log(f"Artifacts saved under {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for command-line interface.
    
    Returns:
        ArgumentParser with the following options:
        - config: Path to YAML config file
        - model_name: Pre-trained model to use (overrides config)
        - train_jsonl: Path to SOP training data (overrides config)
        - out_dir: Output directory for artifacts (overrides config)
        - epochs: Number of training epochs (overrides config)
        - batch_size: Batch size for training (overrides config)
        - lr: Learning rate (overrides config)
    """
    # Get the project root directory (three levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    
    p = argparse.ArgumentParser(description="Train dual-encoder (InfoNCE) and build index.")
    p.add_argument("--config", type=str, default=str(project_root / "config" / "retriever_config.yaml"),
                   help="Path to config file (default: config/retriever_config.yaml)")
    p.add_argument("--model_name", type=str, default=None, 
                   help="Pre-trained model name (overrides config)")
    p.add_argument("--train_jsonl", type=str, default=None,
                   help="Path to training data (overrides config)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (overrides config)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of training epochs (overrides config)")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Batch size for training (overrides config)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (overrides config)")
    return p


# Main entry point
if __name__ == "__main__":
    # Parse command-line arguments
    args = build_parser().parse_args()
    
    # Load config from file
    config = load_config(args.config)
    project_root = Path(args.config).parent.parent
    
    # Merge config with CLI args (CLI args take precedence)
    class Config:
        def __init__(self, cli_args, config_dict, project_root):
            # Get values from CLI args first (if specified), then config, then defaults
            self.model_name = cli_args.model_name or config_dict.get('model', {}).get('name') or "bert-base-uncased"
            self.train_jsonl = cli_args.train_jsonl or config_dict.get('data', {}).get('train_jsonl')
            if self.train_jsonl:
                self.train_jsonl = str(project_root / self.train_jsonl)
            else:
                self.train_jsonl = str(project_root / "src" / "data" / "sop_examples.jsonl")
                
            self.out_dir = cli_args.out_dir or config_dict.get('output', {}).get('out_dir')
            if self.out_dir:
                self.out_dir = str(project_root / self.out_dir)
            else:
                self.out_dir = str(project_root / "artifacts" / "retriever_bert")
                
            self.epochs = cli_args.epochs or config_dict.get('training', {}).get('epochs') or 3
            self.batch_size = cli_args.batch_size or config_dict.get('training', {}).get('batch_size') or 2
            self.lr = cli_args.lr or config_dict.get('training', {}).get('learning_rate') or 1e-4
    
    # Create merged config object
    config = Config(args, config, project_root)
    
    # Log the configuration being used
    console.log("[bold]Configuration:[/bold]")
    console.log(f"  Model: {config.model_name}")
    console.log(f"  Training data: {config.train_jsonl}")
    console.log(f"  Output dir: {config.out_dir}")
    console.log(f"  Epochs: {config.epochs}")
    console.log(f"  Batch size: {config.batch_size}")
    console.log(f"  Learning rate: {config.lr}")
    
    # Train dual-encoder model and build index
    train_dual_encoder(config)


