import os
import random
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from transformers import ViTModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model, VeraConfig, EigenLoRAConfig
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    StanfordCars,
    Flowers102,
    Food101,
    ImageFolder,
)
import torchvision.transforms as transforms


from utils import (
    argument_check,
    dataloader_from_subset,
    consolidate_loras,
    get_eigenvectors,
    calculate_initial_parameters,
)


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier with optional PEFT methods (LoRA, VeRA, EigenLoRA).

    This model wraps a pretrained ViT and adds a classification head. It supports
    training with different parameter-efficient fine-tuning methods.

    Args:
        model_name: HuggingFace model identifier (default: google/vit-base-patch16-224)
        num_classes: Number of output classes
        method: PEFT method - 'lora', 'vera', 'elora', or 'none' for full fine-tuning
        lora_rank: Rank for LoRA/VeRA adaptations
        elora_components: Number of eigenvector components for EigenLoRA
        device: Device to run the model on
    """

    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        num_classes=10,
        method="lora",
        lora_rank=8,
        elora_components=8,
        device="cuda:0",
    ):
        super().__init__()
        self.model_name = model_name
        self.method = method
        self.device = device if torch.cuda.is_available() else "cpu"
        self.lora_rank = lora_rank
        self.elora_components = elora_components

        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_classes)

        # Apply PEFT method
        self._apply_peft_method()

        # Move to device
        self.vit.to(self.device)
        self.classifier.to(self.device)

        print(f"Model initialized on {self.device}")
        self._print_trainable_parameters()

    def _apply_peft_method(self):
        """Apply the specified PEFT method to the ViT backbone."""
        target_modules = ["query", "value"]

        if self.method == "lora":
            config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank * 16,
                target_modules=target_modules,
            )
            self.vit = get_peft_model(self.vit, config)

        elif self.method == "vera":
            config = VeraConfig(
                r=self.lora_rank,
                target_modules=target_modules,
            )
            self.vit = get_peft_model(self.vit, config)

        elif self.method == "elora":
            config = EigenLoRAConfig(
                r=self.lora_rank,
                num_components=self.elora_components,
                num_rank_updates=2,
                target_modules=target_modules,
            )
            self.vit = get_peft_model(self.vit, config)
        # else: full fine-tuning (no PEFT)

    def _print_trainable_parameters(self):
        """Print the number of trainable vs total parameters."""
        trainable_params = 0
        total_params = 0

        for name, param in self.vit.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        # Include classifier parameters
        for param in self.classifier.parameters():
            total_params += param.numel()
            trainable_params += param.numel()

        print(
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, x):
        """Forward pass through ViT + classifier."""
        features = self.vit(x).pooler_output
        return self.classifier(features)

    def train_model(
        self,
        train_loader,
        test_loader,
        epochs=20,
        lr=2e-4,
        weight_decay=1e-6,
        use_wandb=True,
    ):
        """
        Train the model on the provided data.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/validation data
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            use_wandb: Whether to log metrics to Weights & Biases
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Use different schedulers for EigenLoRA vs other methods
        if self.method == "elora":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
            )

        criterion = nn.CrossEntropyLoss()
        best_test_acc = -np.inf

        # Initial evaluation
        test_loss, test_acc = self.evaluate(test_loader)
        if use_wandb:
            wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "epoch": 0})

        for epoch in range(epochs):
            self.train()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_losses = []
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc="Training"):
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.append(logits.argmax(dim=-1).cpu())
                train_labels.append(batch[1])

            # Update scheduler
            avg_loss = np.mean(train_losses)
            if self.method == "elora":
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(avg_loss)
                new_lr = optimizer.param_groups[0]["lr"]
                if old_lr != new_lr:
                    print(f"LR: {old_lr:.2e} -> {new_lr:.2e}")
            else:
                scheduler.step()

            # Calculate train accuracy
            all_preds = torch.cat(train_preds)
            all_labels = torch.cat(train_labels)
            train_acc = (all_preds == all_labels).float().mean().item() * 100

            print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

            if use_wandb:
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_accuracy": train_acc,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # Evaluate on test set
            test_loss, test_acc = self.evaluate(test_loader)

            if use_wandb:
                wandb.log(
                    {
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "epoch": epoch + 1,
                    }
                )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"★ New best: {best_test_acc:.2f}%")
                if use_wandb:
                    wandb.log(
                        {"best_test_accuracy": best_test_acc, "best_epoch": epoch + 1}
                    )

    @torch.no_grad()
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.

        Returns:
            tuple: (average_loss, accuracy_percentage)
        """
        self.eval()
        criterion = nn.CrossEntropyLoss()

        test_losses = []
        test_preds = []
        test_labels = []

        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            logits = self(images)
            loss = criterion(logits, labels)

            test_losses.append(loss.item())
            test_preds.append(logits.argmax(dim=-1).cpu())
            test_labels.append(batch[1])

        avg_loss = np.mean(test_losses)
        all_preds = torch.cat(test_preds)
        all_labels = torch.cat(test_labels)
        accuracy = (all_preds == all_labels).float().mean().item() * 100

        print(f"Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

        return avg_loss, accuracy

    def save_checkpoint(self, save_path):
        """Save model state dict to the specified path."""
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")


def get_image_transforms(processor, train=True):
    """
    Get image transforms for training or evaluation.

    Args:
        processor: HuggingFace image processor
        train: Whether to use training augmentations

    Returns:
        torchvision.transforms.Compose object
    """
    img_size = processor.size["height"]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean, std=processor.image_std
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean, std=processor.image_std
                ),
            ]
        )


def load_dataset(dataset_name, data_root, processor):
    """
    Load a dataset by name.

    Args:
        dataset_name: Name of the dataset (CIFAR10, CIFAR100, etc.)
        data_root: Root directory for dataset storage
        processor: HuggingFace image processor for transforms

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_transform = get_image_transforms(processor, train=True)
    test_transform = get_image_transforms(processor, train=False)

    dataset_loaders = {
        "CIFAR10": lambda: (
            CIFAR10(
                root=data_root, train=True, transform=train_transform, download=True
            ),
            CIFAR10(
                root=data_root, train=False, transform=test_transform, download=True
            ),
        ),
        "CIFAR100": lambda: (
            CIFAR100(
                root=data_root, train=True, transform=train_transform, download=True
            ),
            CIFAR100(
                root=data_root, train=False, transform=test_transform, download=True
            ),
        ),
        "stanford_cars": lambda: (
            StanfordCars(
                root=data_root, split="train", transform=train_transform, download=False
            ),
            StanfordCars(
                root=data_root, split="test", transform=test_transform, download=False
            ),
        ),
        "food101": lambda: (
            Food101(
                root=data_root, split="train", transform=train_transform, download=True
            ),
            Food101(
                root=data_root, split="test", transform=test_transform, download=True
            ),
        ),
        "flowers102": lambda: (
            Flowers102(
                root=data_root, split="train", transform=train_transform, download=True
            ),
            Flowers102(
                root=data_root, split="test", transform=test_transform, download=True
            ),
        ),
        "RESISC45": lambda: (
            ImageFolder(
                root=os.path.join(data_root, "RESISC45_train"),
                transform=train_transform,
            ),
            ImageFolder(
                root=os.path.join(data_root, "RESISC45_test"), transform=test_transform
            ),
        ),
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(dataset_loaders.keys())}"
        )

    return dataset_loaders[dataset_name]()


def create_class_subsets(dataset, subset_size, dataset_name=None):
    """
    Divide dataset classes into random subsets.

    Args:
        dataset: The dataset to create subsets from
        subset_size: Number of classes per subset
        dataset_name: Optional dataset name for special handling

    Returns:
        list: List of class index lists, one per subset
    """
    # Determine number of classes
    if dataset_name == "food101":
        num_classes = 101
    elif dataset_name == "flowers102":
        num_classes = 102
    elif hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    else:
        raise ValueError("Cannot determine number of classes in dataset")

    if num_classes % subset_size != 0:
        raise ValueError(
            f"subset_size ({subset_size}) must evenly divide "
            f"number of classes ({num_classes})"
        )

    all_classes = list(range(num_classes))
    random.shuffle(all_classes)

    subsets = []
    for i in range(0, num_classes, subset_size):
        subsets.append(all_classes[i : i + subset_size])

    return subsets


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ViT with LoRA/VeRA/EigenLoRA on image classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "vera", "elora", "none"],
        help="PEFT method to use (none = full fine-tuning)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--r",
        "--rank",
        type=int,
        default=8,
        dest="rank",
        help="LoRA/VeRA rank parameter",
    )
    parser.add_argument(
        "--elora_components",
        type=int,
        default=8,
        help="Number of eigenvector components for EigenLoRA",
    )

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "stanford_cars",
            "food101",
            "flowers102",
            "RESISC45",
        ],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for dataset storage",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10,
        help="Number of classes per subset",
    )

    # Paths
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--sampled_subsets_path",
        type=str,
        default=None,
        help="Path to file with predefined class subsets (required for elora/vera)",
    )
    parser.add_argument(
        "--lora_dict_directory",
        type=str,
        default=None,
        help="Directory with LoRA checkpoints for EigenLoRA initialization",
    )

    # EigenLoRA specific
    parser.add_argument(
        "--elora_holdout",
        action="store_true",
        help="Use leave-one-out evaluation for EigenLoRA",
    )

    # Logging
    parser.add_argument(
        "--use_wandb", action="store_true", help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ViT_PEFT", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity/team name"
    )

    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_args()
    print(f"\nConfiguration:\n{'-' * 40}")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"{'-' * 40}\n")

    # Validate arguments
    argument_check(args)

    # Setup experiment directory
    experiment_dir = os.path.join(args.save_path, args.method, args.dataset)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "label_mappings"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "model_checkpoints"), exist_ok=True)

    # Load image processor and dataset
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    train_set, test_set = load_dataset(args.dataset, args.data_root, processor)

    # Create or load class subsets
    if args.method in ["vera", "elora"] and args.sampled_subsets_path:
        # Load predefined subsets for consistency with LoRA experiments
        with open(os.path.join(args.save_path, args.sampled_subsets_path), "r") as f:
            subsets = [eval(line.split(": ")[1].strip()) for line in f.readlines()]
    else:
        subsets = create_class_subsets(train_set, args.subset_size, args.dataset)

    # Save subset configuration
    subset_file = os.path.join(experiment_dir, "sampled_subsets.txt")
    with open(subset_file, "w") as f:
        for i, subset in enumerate(subsets):
            f.write(f"Subset {i + 1}: {subset}\n")
    print(f"Saved {len(subsets)} class subsets to {subset_file}")

    # Train on each subset
    for subset_idx, class_subset in enumerate(subsets):
        print(f"\n{'=' * 50}")
        print(f"Training Subset {subset_idx + 1}/{len(subsets)}")
        print(f"Classes: {class_subset}")
        print(f"{'=' * 50}")

        # Initialize model
        model = ViTClassifier(
            model_name=args.model_name,
            num_classes=args.subset_size,
            method=args.method,
            lora_rank=args.rank,
            elora_components=args.elora_components,
        )

        # Initialize EigenLoRA with pre-computed components
        if args.method == "elora":
            holdout_idx = subset_idx if args.elora_holdout else None
            consolidated = consolidate_loras(
                args.lora_dict_directory, holdout_lora=holdout_idx
            )
            eigenvectors = get_eigenvectors(
                consolidated, num_components=args.elora_components
            )

            # Find the corresponding LoRA checkpoint for this subset
            lora_path = os.path.join(
                args.lora_dict_directory, f"subset_{subset_idx + 1}_model.pth"
            )
            if not os.path.exists(lora_path):
                lora_path = os.path.join(
                    args.lora_dict_directory, f"model_{subset_idx + 1}.pth"
                )

            init_params = calculate_initial_parameters(
                eigenvectors,
                torch.load(lora_path, map_location="cpu"),
                num_components=args.elora_components,
                rank=args.rank,
            )
            model.load_state_dict(init_params, strict=False)
            model.vit.model_mode("default", "inference")

        # Setup W&B logging
        if args.use_wandb:
            wandb.init(
                project=f"{args.wandb_project}_{args.dataset}",
                entity=args.wandb_entity,
                config=vars(args),
                name=f"{args.method}_subset_{subset_idx + 1}",
                reinit=True,
            )

        # Create data loaders for this subset
        train_loader, test_loader = dataloader_from_subset(
            train_set,
            test_set,
            class_subset,
            experiment_folder=experiment_dir,
            subset_index=subset_idx + 1,
            batch_size=args.batch_size,
        )

        # Train
        model.train_model(
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_wandb=args.use_wandb,
        )

        # Save checkpoint
        model.save_checkpoint(
            os.path.join(
                experiment_dir,
                "model_checkpoints",
                f"subset_{subset_idx + 1}_model.pth",
            )
        )

        if args.use_wandb:
            wandb.finish()

    print(f"\n{'=' * 50}")
    print("Training complete!")
    print(f"Checkpoints saved to: {experiment_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
