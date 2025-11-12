#!/usr/bin/env python3
"""
Script to generate loss and IoU diagrams from existing training log files.
This script creates separate plots for IoU and Loss with a dashed vertical line at epoch 10.
Epoch 10 is treated as the current epoch for training shape analysis.

Usage:
    python generate_loss_diagram.py --log_file path/to/logfile.log --output_dir path/to/output
    python generate_loss_diagram.py --log_file checkpoints/phase_1/seqtrack-seqtrack_b256.log --output_dir checkpoints/phase_1
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys

def parse_log_file(log_file_path):
    """
    Parse the training log file to extract IoU and Loss values.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary containing parsed data
    """
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")
    
    print(f"Parsing log file: {log_file_path}")
    
    # Patterns to match different log formats - prioritize final epoch values
    iou_patterns = [
        r'\[.*?\]\s*IoU\s*collected:\s*Epoch\s*(\d+),\s*Value:\s*([0-9.]+)',  # [phase_1] IoU collected: Epoch 1, Value: 0.0479 (PRIORITY)
        r'IoU:\s*([0-9.]+)',  # IoU: 0.04789 (fallback)
    ]
    
    loss_patterns = [
        r'\[.*?\]\s*Loss\s*collected:\s*Epoch\s*(\d+),\s*Value:\s*([0-9.]+)',  # [phase_1] Loss collected: Epoch 1, Value: 8.29168 (PRIORITY)
        r'Loss/total:\s*([0-9.]+)',  # Loss/total: 8.29168 (fallback)
    ]
    
    # Pattern to find the last loss value in each epoch
    epoch_end_pattern = r'\[.*?\]\s*Epoch\s*(\d+):\s*\d+\s*/\s*\d+\s*samples.*?Loss/total:\s*([0-9.]+)'
    
    epoch_pattern = r'\[.*?\]\s*Epoch\s*(\d+):'  # [phase_1] Epoch 1:
    
    data = {
        'epochs': [],
        'iou_values': [],
        'loss_values': [],
        'raw_logs': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} lines from log file...")
    
    # Track final values per epoch
    epoch_data = {}  # {epoch: {'iou': value, 'loss': value}}
    current_epoch = None
    last_loss_in_epoch = {}  # Track the last loss value for each epoch
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
            
        # Store raw logs
        data['raw_logs'].append(line)
        
        # Track current epoch
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # Extract IoU values - prioritize "collected" format
        for pattern in iou_patterns:
            iou_match = re.search(pattern, line)
            if iou_match:
                if len(iou_match.groups()) == 2:
                    # IoU collected format - this is the final value for the epoch
                    epoch_num = int(iou_match.group(1))
                    iou_value = float(iou_match.group(2))
                    if epoch_num not in epoch_data:
                        epoch_data[epoch_num] = {}
                    epoch_data[epoch_num]['iou'] = iou_value
                    print(f"Found IoU for Epoch {epoch_num}: {iou_value:.4f}")
                elif len(iou_match.groups()) == 1:
                    # Simple IoU: value format - only use if no collected value exists
                    iou_value = float(iou_match.group(1))
                    # Don't use intermediate values, only final collected values
                break
        
        # Extract Loss values - prioritize "collected" format
        for pattern in loss_patterns:
            loss_match = re.search(pattern, line)
            if loss_match:
                if len(loss_match.groups()) == 2:
                    # Loss collected format - this is the final value for the epoch
                    epoch_num = int(loss_match.group(1))
                    loss_value = float(loss_match.group(2))
                    if epoch_num not in epoch_data:
                        epoch_data[epoch_num] = {}
                    epoch_data[epoch_num]['loss'] = loss_value
                    print(f"Found Loss for Epoch {epoch_num}: {loss_value:.4f}")
                elif len(loss_match.groups()) == 1:
                    # Simple Loss: value format - track the last value for current epoch
                    loss_value = float(loss_match.group(1))
                    if current_epoch is not None:
                        last_loss_in_epoch[current_epoch] = loss_value
                break
    
    # Process collected data
    for epoch_num in sorted(epoch_data.keys()):
        epoch_info = epoch_data[epoch_num]
        
        # Use collected loss if available, otherwise use last loss in epoch
        loss_value = None
        if 'loss' in epoch_info:
            loss_value = epoch_info['loss']
        elif epoch_num in last_loss_in_epoch:
            loss_value = last_loss_in_epoch[epoch_num]
            print(f"Using last loss value for Epoch {epoch_num}: {loss_value:.4f}")
        
        if 'iou' in epoch_info and loss_value is not None:
            data['epochs'].append(epoch_num)
            data['iou_values'].append(epoch_info['iou'])
            data['loss_values'].append(loss_value)
            print(f"Epoch {epoch_num}: IoU={epoch_info['iou']:.4f}, Loss={loss_value:.4f}")
        else:
            print(f"Warning: Incomplete data for Epoch {epoch_num} - IoU: {'iou' in epoch_info}, Loss: {loss_value is not None}")
    
    print(f"Extracted data for {len(data['epochs'])} epochs")
    return data

def create_plots(data, output_dir, phase_name="training"):
    """
    Create IoU and Loss plots from parsed data.
    
    Args:
        data (dict): Parsed data from log file
        output_dir (str): Directory to save plots
        phase_name (str): Name for the phase (used in plot titles)
    """
    if not data['epochs']:
        print("No epoch data found in log file!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create IoU plot
    if data['iou_values']:
        plt.figure(figsize=(12, 8))
        plt.plot(data['epochs'], data['iou_values'], marker='o', linewidth=2, markersize=6, color='blue')
        
        # Add dashed vertical line at epoch 10
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Current Epoch: 10')
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("IoU", fontsize=12)
        plt.title(f"IoU Progress - {phase_name} (Total Epochs: {len(data['epochs'])})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        iou_path = os.path.join(output_dir, f"{phase_name}_iou.png")
        plt.savefig(iou_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"IoU plot saved: {iou_path}")
        print(f"   IoU range: {min(data['iou_values']):.4f} to {max(data['iou_values']):.4f}")
    
    # Create Loss plot
    if data['loss_values']:
        plt.figure(figsize=(12, 8))
        plt.plot(data['epochs'], data['loss_values'], marker='o', linewidth=2, markersize=6, color='red')
        
        # Add dashed vertical line at epoch 10
        plt.axvline(x=10, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Current Epoch: 10')
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Loss Progress - {phase_name} (Total Epochs: {len(data['epochs'])})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        loss_path = os.path.join(output_dir, f"{phase_name}_loss.png")
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved: {loss_path}")
        print(f"   Loss range: {min(data['loss_values']):.4f} to {max(data['loss_values']):.4f}")

def save_parsed_data(data, output_dir, phase_name="training"):
    """
    Save parsed data to files for future use.
    
    Args:
        data (dict): Parsed data from log file
        output_dir (str): Directory to save data files
        phase_name (str): Name for the phase
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save epoch data as CSV
    if data['epochs']:
        import csv
        csv_path = os.path.join(output_dir, f"{phase_name}_training_data.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'IoU', 'Loss', 'Current_Epoch'])
            for i, epoch in enumerate(data['epochs']):
                iou_val = data['iou_values'][i] if i < len(data['iou_values']) else None
                loss_val = data['loss_values'][i] if i < len(data['loss_values']) else None
                # Mark epoch 10 as current epoch
                current_epoch = "Yes" if epoch == 10 else "No"
                writer.writerow([epoch, iou_val, loss_val, current_epoch])
        print(f"Training data saved: {csv_path}")
    
    # Save raw logs
    log_path = os.path.join(output_dir, f"{phase_name}_raw_logs.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in data['raw_logs']:
            f.write(line + '\n')
    print(f"Raw logs saved: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate loss and IoU diagrams from training log files')
    parser.add_argument('--log_file', type=str, required=True, 
                       help='Path to the training log file')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save generated plots (default: ./plots)')
    parser.add_argument('--phase_name', type=str, default='training',
                       help='Name for the training phase (default: training)')
    parser.add_argument('--save_data', action='store_true',
                       help='Save parsed data as CSV and raw logs')
    
    args = parser.parse_args()
    
    try:
        print("Starting log file analysis...")
        print(f"Log file: {args.log_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Phase name: {args.phase_name}")
        
        # Parse log file
        data = parse_log_file(args.log_file)
        
        if not data['epochs']:
            print("No training data found in log file!")
            return
        
        # Create plots
        create_plots(data, args.output_dir, args.phase_name)
        
        # Save data if requested
        if args.save_data:
            save_parsed_data(data, args.output_dir, args.phase_name)
        
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
