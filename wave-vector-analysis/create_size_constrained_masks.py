#!/usr/bin/env python3
"""
Create embryo masks using size constraints - embryos are always roughly the same size,
so we can filter by expected embryo area to exclude background and noise.

This script:
1. Uses adaptive thresholding with a wider greyscale range
2. Filters contours by expected embryo size (min/max area)
3. Excludes regions that are too large (background) or too small (noise)
4. Outputs masks that capture the actual embryo body, not the whole image
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import tifffile as tiff
import matplotlib.pyplot as plt
import csv


def find_brightest_frame(tiff_path, num_samples=5):
    """
    Find the brightest frame in a TIFF stack by sampling frames and comparing
    their 99th percentile intensity.
    
    Returns:
        Tuple of (frame_index, frame_data) or (None, None) if error
    """
    try:
        with tiff.TiffFile(tiff_path) as tif:
            num_pages = len(tif.pages)
            if num_pages == 0:
                return None, None
            
            best_frame_idx = None
            best_frame_data = None
            max_intensity_score = -1
            
            # Sample frames to find the brightest one
            # Always check first, middle, and last frames, plus random samples
            frames_to_sample = sorted(list(set([0, num_pages // 2, num_pages - 1] + 
                                              [np.random.randint(0, num_pages) for _ in range(num_samples - 3)])))
            
            for frame_idx in frames_to_sample:
                if frame_idx >= num_pages:
                    continue
                img = tif.pages[frame_idx].asarray()
                
                # Convert to grayscale if multi-channel
                if img.ndim == 3:
                    if img.shape[2] == 3:  # RGB
                        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    elif img.shape[2] == 4:  # RGBA
                        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                    else:  # Assume first channel is relevant
                        img_gray = img[:, :, 0]
                else:
                    img_gray = img
                
                # Convert to 8-bit for consistent intensity comparison
                if img_gray.dtype == np.uint16:
                    # Use percentile-based conversion
                    p2 = np.percentile(img_gray, 2)
                    p98 = np.percentile(img_gray, 98)
                    if p98 > p2:
                        img_8bit = np.clip((img_gray.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                    else:
                        img_8bit = (img_gray / 65535.0 * 255).astype(np.uint8)
                else:
                    img_8bit = img_gray.astype(np.uint8)
                
                # Use 99th percentile as a robust measure of "brightness"
                current_score = np.percentile(img_8bit, 99)
                
                if current_score > max_intensity_score:
                    max_intensity_score = current_score
                    best_frame_idx = frame_idx
                    # Store the original (not 8-bit) for mask generation
                    best_frame_data = img_gray
            
            if best_frame_data is not None:
                return best_frame_idx, best_frame_data
            return None, None
            
    except Exception as e:
        print(f"    ⚠ Error finding brightest frame: {e}")
        return None, None


def create_improved_mask(gray, min_area_ratio=0.005, max_area_ratio=0.15, 
                         method='combined', expand_percent=10.0):
    """
    Create improved mask using multiple strategies to better match embryo shape.
    
    Methods:
    - 'combined': Try multiple approaches and combine best results
    - 'edge_based': Use edge detection + thresholding
    - 'adaptive': Use adaptive thresholding
    - 'watershed': Use watershed segmentation
    - 'gradient': Use gradient-based approach
    """
    h, w = gray.shape
    total_pixels = h * w
    min_area = int(min_area_ratio * total_pixels)
    max_area = int(max_area_ratio * total_pixels)
    
    # Normalize to float32
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float32) / 65535.0
        gray_8bit = (gray_float * 255).astype(np.uint8)
    elif gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32) / 255.0
        gray_8bit = gray
    else:
        gray_float = gray.astype(np.float32)
        gray_8bit = (np.clip(gray_float, 0, 255)).astype(np.uint8)
    
    # Apply Gaussian blur for noise reduction
    blur = cv2.GaussianBlur(gray_8bit, (5, 5), 0)
    blur_float = blur.astype(np.float32) / 255.0
    
    if method == 'combined' or method == 'edge_based':
        # Method 1: Edge-based approach
        # Use Canny edge detection to find embryo boundaries
        edges = cv2.Canny(blur, 30, 100)
        
        # Dilate edges slightly to connect nearby edges
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_small, iterations=2)
        
        # Fill enclosed regions
        edges_filled = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_small, iterations=3)
        
        # Combine with intensity thresholding
        flat_intensities = blur.flatten()
        p10 = np.percentile(flat_intensities, 10)
        p50 = np.percentile(flat_intensities, 50)
        threshold = max(p10 * 1.5, p50 * 0.6)
        intensity_mask = (blur >= threshold).astype(np.uint8) * 255
        
        # Combine edge and intensity masks
        combined = cv2.bitwise_and(edges_filled, intensity_mask)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                embryo_contours.append(contour)
        
        if embryo_contours:
            mask1 = np.zeros_like(blur)
            cv2.drawContours(mask1, embryo_contours, -1, 255, -1)
        else:
            mask1 = None
    
    if method == 'combined' or method == 'adaptive':
        # Method 2: Improved adaptive thresholding
        flat_intensities = blur.flatten()
        p5 = np.percentile(flat_intensities, 5)
        p15 = np.percentile(flat_intensities, 15)
        p50 = np.percentile(flat_intensities, 50)
        mean_int = flat_intensities.mean()
        
        # Use multiple thresholds and combine
        threshold1 = max(p5, p50 * 0.5, mean_int * 0.6)
        threshold2 = max(p15, p50 * 0.65)
        
        mask_a = (blur >= threshold1).astype(np.uint8) * 255
        mask_b = (blur >= threshold2).astype(np.uint8) * 255
        
        # Use the more permissive one
        mask2 = mask_a
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Filter by size
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                embryo_contours.append(contour)
        
        if embryo_contours:
            mask2_final = np.zeros_like(blur)
            cv2.drawContours(mask2_final, embryo_contours, -1, 255, -1)
            mask2 = mask2_final
        else:
            mask2 = None
    
    if method == 'combined' or method == 'gradient':
        # Method 3: Gradient-based approach
        # Compute gradients
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
        
        # Threshold gradient to find boundaries
        grad_threshold = np.percentile(gradient_magnitude, 75)
        grad_mask = (gradient_magnitude >= grad_threshold).astype(np.uint8) * 255
        
        # Combine with intensity
        flat_intensities = blur.flatten()
        p10 = np.percentile(flat_intensities, 10)
        intensity_mask = (blur >= p10 * 1.3).astype(np.uint8) * 255
        
        # Combine and fill
        combined = cv2.bitwise_and(grad_mask, intensity_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                embryo_contours.append(contour)
        
        if embryo_contours:
            mask3 = np.zeros_like(blur)
            cv2.drawContours(mask3, embryo_contours, -1, 255, -1)
        else:
            mask3 = None
    
    # Combine methods if using 'combined'
    if method == 'combined':
        masks = [m for m in [mask1, mask2, mask3] if m is not None]
        if masks:
            # Combine all masks
            final_mask = np.zeros_like(blur)
            for m in masks:
                final_mask = cv2.bitwise_or(final_mask, m)
            
            # Refine by finding largest connected components
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Sort by area and keep largest ones
                contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
                embryo_contours = []
                for contour in contours_sorted:
                    area = cv2.contourArea(contour)
                    if min_area <= area <= max_area:
                        embryo_contours.append(contour)
                    if len(embryo_contours) >= 2:  # Max 2 embryos
                        break
                
                final_mask = np.zeros_like(blur)
                if embryo_contours:
                    cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
        else:
            final_mask = np.zeros_like(blur)
    else:
        # Use single method
        if method == 'edge_based':
            final_mask = mask1 if mask1 is not None else np.zeros_like(blur)
        elif method == 'adaptive':
            final_mask = mask2 if mask2 is not None else np.zeros_like(blur)
        elif method == 'gradient':
            final_mask = mask3 if mask3 is not None else np.zeros_like(blur)
        else:
            final_mask = np.zeros_like(blur)
    
    # Fill holes
    if np.sum(final_mask > 0) > 0:
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(final_mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            final_mask = filled
    
    # Expand if requested
    if expand_percent > 0 and np.sum(final_mask > 0) > 0:
        mask_area = np.sum(final_mask > 0)
        radius_factor = np.sqrt(1 + expand_percent / 100.0) - 1
        # Get contours for radius estimation
        expand_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if expand_contours:
            avg_radius = 0
            for contour in expand_contours:
                (x, y), (w_rect, h_rect), angle = cv2.minAreaRect(contour)
                avg_radius += np.sqrt(w_rect * h_rect) / 2
            avg_radius = avg_radius / len(expand_contours)
        else:
            avg_radius = np.sqrt(mask_area / np.pi)
        dilation_radius = max(2, int(avg_radius * radius_factor))
        kernel_size = dilation_radius * 2 + 1
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        final_mask = cv2.dilate(final_mask, dilation_kernel, iterations=1)
    
    # Final cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Calculate statistics
    mask_area = np.sum(final_mask > 0)
    coverage = (mask_area / total_pixels) * 100
    
    # Get final contour count
    final_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stats = {
        'method': method,
        'mask_area': mask_area,
        'coverage': coverage,
        'num_contours': len(final_contours),
    }
    
    return final_mask, stats


def create_size_constrained_mask(gray, min_area_ratio=0.005, max_area_ratio=0.15, 
                                 percentile_low=5, background_percentile=10,
                                 min_intensity_ratio=0.3, expand_percent=10.0, 
                                 method='improved'):
    """
    Create mask using size constraints to identify embryo-sized regions.
    
    Args:
        gray: Grayscale image
        min_area_ratio: Minimum embryo area as fraction of image (default: 0.5%)
        max_area_ratio: Maximum embryo area as fraction of image (default: 15%)
        percentile_low: Lower percentile for thresholding (default: 5)
        background_percentile: Percentile to define background (default: 10)
        min_intensity_ratio: Minimum ratio relative to background
    
    Returns:
        Tuple of (mask, stats_dict)
    """
    h, w = gray.shape
    total_pixels = h * w
    min_area = int(min_area_ratio * total_pixels)
    max_area = int(max_area_ratio * total_pixels)
    
    # Normalize to float32
    if gray.dtype == np.uint16:
        gray_float = gray.astype(np.float32)
    elif gray.dtype == np.uint8:
        gray_float = gray.astype(np.float32)
    else:
        gray_float = gray.astype(np.float32)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
    flat_intensities = blur.flatten()
    
    # Calculate thresholds
    background_threshold = np.percentile(flat_intensities, background_percentile)
    median_intensity = np.median(flat_intensities)
    p25 = np.percentile(flat_intensities, 25)
    mean_intensity = flat_intensities.mean()
    
    # Use lower percentile but ensure it's above background
    percentile_threshold = np.percentile(flat_intensities, percentile_low)
    adaptive_threshold = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
    
    # Use the more permissive threshold (lower value = more inclusive)
    embryo_threshold = min(percentile_threshold, adaptive_threshold)
    embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
    
    # Create initial mask
    initial_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
    
    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours and filter by size
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            embryo_contours.append(contour)
    
    # If no contours found with strict size filter, try more lenient approach
    # This handles cases where embryos might be slightly larger or smaller than expected
    if not embryo_contours and len(contours) > 0:
        # Try with more lenient min_area (half of original)
        lenient_min_area = max(min_area * 0.5, 100)  # At least 100 pixels
        lenient_max_area = max_area * 1.5  # Allow up to 1.5x max
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if lenient_min_area <= area <= lenient_max_area:
                embryo_contours.append(contour)
        
        if embryo_contours:
            print(f"    ⚠ Using lenient size filter: {lenient_min_area:.0f} - {lenient_max_area:.0f} pixels")
    
    # Create final mask from filtered contours
    final_mask = np.zeros_like(initial_mask)
    if embryo_contours:
        cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
    
    # Fill holes within embryo contours
    if embryo_contours:
        # Create a mask for each embryo and fill holes
        for contour in embryo_contours:
            # Create individual mask
            single_mask = np.zeros_like(final_mask)
            cv2.drawContours(single_mask, [contour], -1, 255, -1)
            
            # Fill holes
            filled = single_mask.copy()
            contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled, contours_filled, -1, 255, -1)
            
            # Combine
            final_mask = cv2.bitwise_or(final_mask, filled)
    
    # Expand mask by specified percentage using dilation
    # Calculate dilation kernel size based on mask dimensions
    if np.sum(final_mask > 0) > 0 and expand_percent > 0:
        # Estimate average dimension of mask regions
        mask_area_before = np.sum(final_mask > 0)
        
        # To increase area by expand_percent%, we need to increase radius by sqrt(1 + expand_percent/100) - 1
        # For area = πr²: new_area = (1 + expand_percent/100) * old_area
        # => new_r = sqrt(1 + expand_percent/100) * old_r
        # => dilation_radius ≈ (sqrt(1 + expand_percent/100) - 1) * old_r
        radius_factor = np.sqrt(1 + expand_percent / 100.0) - 1
        
        # Estimate average radius of mask regions
        if embryo_contours:
            # Use average of contour bounding box dimensions
            avg_radius = 0
            for contour in embryo_contours:
                (x, y), (w, h), angle = cv2.minAreaRect(contour)
                avg_radius += np.sqrt(w * h) / 2
            avg_radius = avg_radius / len(embryo_contours)
        else:
            # Fallback: estimate from total area
            avg_radius = np.sqrt(mask_area_before / np.pi)
        
        dilation_radius = max(2, int(avg_radius * radius_factor))  # At least 2 pixels
        
        # Create circular kernel for dilation
        kernel_size = dilation_radius * 2 + 1
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        final_mask = cv2.dilate(final_mask, dilation_kernel, iterations=1)
    
    # Final cleanup
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Calculate statistics
    mask_area = np.sum(final_mask > 0)
    coverage = (mask_area / total_pixels) * 100
    
    stats = {
        'method': 'size_constrained',
        'background_threshold': background_threshold,
        'embryo_threshold': embryo_threshold,
        'percentile_threshold': percentile_threshold,
        'adaptive_threshold': adaptive_threshold,
        'min_area': min_area,
        'max_area': max_area,
        'mask_area': mask_area,
        'coverage': coverage,
        'num_contours_before_filter': len(contours),
        'num_contours_after_filter': len(embryo_contours),
        'total_pixels': total_pixels,
    }
    
    return final_mask, stats


def process_tiff_file(tiff_path, output_dir=None, min_area_ratio=0.005, max_area_ratio=0.15,
                      percentile_low=5, expand_percent=10.0, visualize=True, save_mask=True, 
                      frame_idx=None, use_brightest=True, skip_blank=False, method='improved'):
    """
    Process a single TIFF file and create size-constrained embryo masks.
    
    Args:
        frame_idx: Specific frame index to use (None = auto-detect)
        use_brightest: If True and frame_idx is None, use brightest frame
    """
    print(f"\n{'='*60}")
    print(f"Processing: {Path(tiff_path).name}")
    print(f"{'='*60}")
    
    # Determine which frame to use
    try:
        with tiff.TiffFile(tiff_path) as tif:
            num_frames = len(tif.pages)
            print(f"  Found {num_frames} frames")
            
            if frame_idx is not None:
                # Use specified frame
                if frame_idx >= num_frames:
                    print(f"  ⚠ Warning: frame_idx {frame_idx} >= {num_frames}, using frame 0")
                    frame_idx = 0
                raw_data = tif.pages[frame_idx].asarray()
                print(f"  Using specified frame {frame_idx}/{num_frames}")
            elif use_brightest:
                # Find and use brightest frame
                print(f"  Finding brightest frame...")
                brightest_idx, brightest_data = find_brightest_frame(tiff_path)
                if brightest_data is not None:
                    frame_idx = brightest_idx
                    gray = brightest_data
                    print(f"  → Using brightest frame {frame_idx}/{num_frames}")
                else:
                    print(f"  ⚠ Could not find brightest frame, using frame 0")
                    frame_idx = 0
                    raw_data = tif.pages[0].asarray()
            else:
                # Default to frame 0
                frame_idx = 0
                raw_data = tif.pages[0].asarray()
                print(f"  Using default frame 0/{num_frames}")
    except Exception as e:
        print(f"  ✗ Error reading TIFF: {e}")
        return None
    
    # Convert to grayscale if we didn't already get grayscale from brightest frame
    if 'gray' not in locals():
        if raw_data.ndim == 3:
            if raw_data.shape[2] == 3:
                gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)
            else:
                gray = raw_data[:, :, 0]
        else:
            gray = raw_data
    
    # Create mask using improved method
    if method == 'improved' or method in ['combined', 'edge_based', 'adaptive', 'gradient']:
        mask, stats = create_improved_mask(
            gray,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            method='combined' if method == 'improved' else method,
            expand_percent=expand_percent
        )
    else:
        # Use original method (fallback to original create_size_constrained_mask logic)
        # This is the original implementation inline
        h, w = gray.shape
        total_pixels = h * w
        min_area = int(min_area_ratio * total_pixels)
        max_area = int(max_area_ratio * total_pixels)
        
        if gray.dtype == np.uint16:
            gray_float = gray.astype(np.float32)
        else:
            gray_float = gray.astype(np.float32)
        
        blur = cv2.GaussianBlur(gray_float, (5, 5), 0)
        flat_intensities = blur.flatten()
        
        background_threshold = np.percentile(flat_intensities, background_percentile)
        median_intensity = np.median(flat_intensities)
        p25 = np.percentile(flat_intensities, 25)
        mean_intensity = flat_intensities.mean()
        
        percentile_threshold = np.percentile(flat_intensities, percentile_low)
        adaptive_threshold = max(median_intensity * 0.5, p25 * 0.7, background_threshold * 1.2)
        
        embryo_threshold = min(percentile_threshold, adaptive_threshold)
        embryo_threshold = max(embryo_threshold, background_threshold * (1 + min_intensity_ratio))
        
        initial_mask = (blur >= embryo_threshold).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        embryo_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                embryo_contours.append(contour)
        
        if not embryo_contours and len(contours) > 0:
            lenient_min_area = max(min_area * 0.5, 100)
            lenient_max_area = max_area * 1.5
            for contour in contours:
                area = cv2.contourArea(contour)
                if lenient_min_area <= area <= lenient_max_area:
                    embryo_contours.append(contour)
        
        final_mask = np.zeros_like(initial_mask)
        if embryo_contours:
            cv2.drawContours(final_mask, embryo_contours, -1, 255, -1)
            
            for contour in embryo_contours:
                single_mask = np.zeros_like(final_mask)
                cv2.drawContours(single_mask, [contour], -1, 255, -1)
                filled = single_mask.copy()
                contours_filled, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(filled, contours_filled, -1, 255, -1)
                final_mask = cv2.bitwise_or(final_mask, filled)
        
        if np.sum(final_mask > 0) > 0 and expand_percent > 0:
            mask_area_before = np.sum(final_mask > 0)
            radius_factor = np.sqrt(1 + expand_percent / 100.0) - 1
            if embryo_contours:
                avg_radius = 0
                for contour in embryo_contours:
                    (x, y), (w_rect, h_rect), angle = cv2.minAreaRect(contour)
                    avg_radius += np.sqrt(w_rect * h_rect) / 2
                avg_radius = avg_radius / len(embryo_contours)
            else:
                avg_radius = np.sqrt(mask_area_before / np.pi)
            
            dilation_radius = max(2, int(avg_radius * radius_factor))
            kernel_size = dilation_radius * 2 + 1
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            final_mask = cv2.dilate(final_mask, dilation_kernel, iterations=1)
        
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        mask_area = np.sum(final_mask > 0)
        coverage = (mask_area / total_pixels) * 100
        
        stats = {
            'method': 'original',
            'mask_area': mask_area,
            'coverage': coverage,
            'num_contours': len(embryo_contours),
        }
        
        mask = final_mask
    
    print(f"\n  Mask Statistics:")
    if 'embryo_threshold' in stats:
        print(f"    Embryo threshold: {stats['embryo_threshold']:.1f}")
    if 'min_area' in stats:
        print(f"    Area range: {stats['min_area']:,} - {stats['max_area']:,} pixels")
    if 'num_contours_before_filter' in stats:
        print(f"    Contours before filter: {stats['num_contours_before_filter']}")
    if 'num_contours_after_filter' in stats:
        print(f"    Contours after filter: {stats['num_contours_after_filter']}")
    print(f"    Method: {stats.get('method', 'unknown')}")
    print(f"    Mask area: {stats['mask_area']:,} pixels ({stats['coverage']:.2f}% of image)")
    print(f"    Contours: {stats.get('num_contours', 0)}")
    
    # Save mask (only if not blank, or if skip_blank is False)
    mask_area_check = np.sum(mask > 0)
    is_blank = mask_area_check == 0
    
    if save_mask:
        if skip_blank and is_blank:
            print(f"    ⚠ Skipping blank mask (0% coverage)")
            mask_path = None
        else:
            if output_dir is None:
                output_dir = os.path.dirname(tiff_path)
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(tiff_path).stem
            mask_path = os.path.join(output_dir, f"{base_name}_mask_frame{frame_idx}.png")
            cv2.imwrite(mask_path, mask)
            print(f"    ✓ Saved mask to: {mask_path}")
    else:
        mask_path = None
    
    # Create visualization
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original grayscale
        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')
        
        # Mask overlay
        overlay = gray.copy()
        if gray.dtype == np.uint16:
            overlay = (overlay / 65535.0 * 255).astype(np.uint8)
        overlay_colored = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        overlay_colored[mask > 0] = [0, 255, 0]  # Green overlay
        
        axes[0, 1].imshow(overlay_colored)
        axes[0, 1].set_title(f'Mask Overlay (Green = Embryo, {stats["coverage"]:.2f}%)')
        axes[0, 1].axis('off')
        
        # Mask only
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title('Binary Mask')
        axes[1, 0].axis('off')
        
        # Intensity histogram
        axes[1, 1].hist(gray.flatten(), bins=100, alpha=0.7, label='All pixels', density=True)
        axes[1, 1].hist(gray[mask > 0].flatten(), bins=100, alpha=0.7, label='Embryo pixels', color='green', density=True)
        
        bg_thresh = stats['background_threshold']
        emb_thresh = stats['embryo_threshold']
        axes[1, 1].axvline(bg_thresh, color='red', linestyle='--', label=f'Background ({bg_thresh:.1f})')
        axes[1, 1].axvline(emb_thresh, color='blue', linestyle='--', label=f'Embryo ({emb_thresh:.1f})')
        
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Intensity Histogram')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir is None:
            output_dir = os.path.dirname(tiff_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        vis_path = os.path.join(output_dir, f"{base_name}_mask_vis_frame{frame_idx}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved visualization to: {vis_path}")
        plt.close()
    
    return {
        'mask': mask,
        'stats': stats,
        'gray': gray,
        'mask_path': mask_path if save_mask else None,
        'tiff_path': str(tiff_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create size-constrained embryo masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python create_size_constrained_masks.py path/to/image.tif
  
  # Process directory
  python create_size_constrained_masks.py path/to/tiff_dir --batch
  
  # Adjust size constraints (embryo should be 1-10% of image)
  python create_size_constrained_masks.py image.tif --min-area 0.01 --max-area 0.10
        """
    )
    
    parser.add_argument('input', help='Input TIFF file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all TIFF files in directory')
    parser.add_argument('--output-dir', '-o', help='Output directory for masks')
    parser.add_argument('--min-area', type=float, default=0.005,
                       help='Minimum embryo area as fraction of image (default: 0.005 = 0.5%%)')
    parser.add_argument('--max-area', type=float, default=0.15,
                       help='Maximum embryo area as fraction of image (default: 0.15 = 15%%)')
    parser.add_argument('--percentile-low', type=float, default=5.0,
                       help='Lower percentile for thresholding (default: 5.0)')
    parser.add_argument('--expand-percent', type=float, default=10.0,
                       help='Expand mask size by this percentage (default: 10.0%%)')
    parser.add_argument('--frame-idx', type=int, default=None,
                       help='Frame index to use (default: None = use brightest frame)')
    parser.add_argument('--use-brightest', action='store_true', default=True,
                       help='Use brightest frame for mask generation when frame-idx is not specified (default: True)')
    parser.add_argument('--no-brightest', dest='use_brightest', action='store_false',
                       help='Disable brightest frame detection, use frame 0 instead')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving mask files')
    parser.add_argument('--skip-blank', action='store_true',
                       help='Skip saving masks that are blank (0%% coverage)')
    parser.add_argument('--method', choices=['improved', 'combined', 'edge_based', 'adaptive', 'gradient', 'original'],
                       default='improved',
                       help='Mask generation method (default: improved)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: Path does not exist: {input_path}")
        return 1
    
    if args.batch or input_path.is_dir():
        # Batch mode
        tiff_files = list(input_path.rglob('*.tif')) + list(input_path.rglob('*.tiff'))
        if not tiff_files:
            print(f"✗ No TIFF files found in {input_path}")
            return 1
        
        print(f"Found {len(tiff_files)} TIFF files")
        results = []
        
        for tiff_file in sorted(tiff_files):
            result = process_tiff_file(
                str(tiff_file),
                output_dir=args.output_dir,
                min_area_ratio=args.min_area,
                max_area_ratio=args.max_area,
                percentile_low=args.percentile_low,
                expand_percent=args.expand_percent,
                visualize=not args.no_visualize,
                save_mask=not args.no_save,
                frame_idx=args.frame_idx,
                use_brightest=args.use_brightest,
                skip_blank=args.skip_blank,
                method=args.method
            )
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Batch processing complete: {len(results)}/{len(tiff_files)} files processed")
        print(f"{'='*60}")
        
        # Generate summary
        if results:
            print(f"\n{'='*60}")
            print("EMBRYO BODY COVERAGE SUMMARY (Size-Constrained)")
            print(f"{'='*60}")
            print(f"{'File':<50} {'Coverage %':<12} {'Mask Area':<15} {'Contours':<10}")
            print("-" * 90)
            
            coverages = []
            for result in results:
                stats = result['stats']
                coverages.append(stats['coverage'])
                
                mask_path = result.get('mask_path')
                if mask_path and mask_path != 'N/A':
                    filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                else:
                    # Extract filename from tiff path if available
                    if 'tiff_path' in result:
                        filename = Path(result['tiff_path']).name
                    else:
                        filename = "Unknown"
                
                print(f"{filename:<50} {stats['coverage']:>10.2f}%  {stats['mask_area']:>13,}  {stats['num_contours_after_filter']:>8}")
            
            print("-" * 90)
            if coverages:
                print(f"\nStatistics:")
                print(f"  Average coverage: {np.mean(coverages):.2f}%")
                print(f"  Median coverage:  {np.median(coverages):.2f}%")
                print(f"  Min coverage:     {np.min(coverages):.2f}%")
                print(f"  Max coverage:     {np.max(coverages):.2f}%")
            
            # Save summary CSV (only include files with masks if skip_blank is True)
            if args.output_dir:
                summary_path = os.path.join(args.output_dir, 'size_constrained_coverage_summary.csv')
            else:
                summary_path = 'size_constrained_coverage_summary.csv'
            
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'File', 'Coverage_%', 'Mask_Area', 'Image_Width', 'Image_Height',
                    'Min_Area', 'Max_Area', 'Contours_Before_Filter', 'Contours_After_Filter',
                    'Embryo_Threshold', 'Background_Threshold'
                ])
                for result in results:
                    stats = result['stats']
                    
                    # Skip blank masks in CSV if skip_blank is enabled
                    if args.skip_blank and stats['coverage'] == 0.0:
                        continue
                    
                    gray_shape = result['gray'].shape
                    
                    mask_path = result.get('mask_path')
                    if mask_path and mask_path != 'N/A':
                        filename = Path(mask_path).name.replace('_mask_frame0.png', '')
                    else:
                        # Extract filename from tiff path if available
                        if 'tiff_path' in result:
                            filename = Path(result['tiff_path']).name
                        else:
                            filename = "Unknown"
                    
                    writer.writerow([
                        filename,
                        f"{stats['coverage']:.2f}",
                        stats['mask_area'],
                        gray_shape[1],
                        gray_shape[0],
                        stats['min_area'],
                        stats['max_area'],
                        stats['num_contours_before_filter'],
                        stats['num_contours_after_filter'],
                        stats['embryo_threshold'],
                        stats['background_threshold']
                    ])
            
            print(f"\n✓ Summary saved to: {summary_path}")
        
    else:
        # Single file mode
        if not input_path.suffix.lower() in ['.tif', '.tiff']:
            print(f"✗ Error: Not a TIFF file: {input_path}")
            return 1
        
        result = process_tiff_file(
            str(input_path),
            output_dir=args.output_dir,
            min_area_ratio=args.min_area,
            max_area_ratio=args.max_area,
            percentile_low=args.percentile_low,
            expand_percent=args.expand_percent,
            visualize=not args.no_visualize,
            save_mask=not args.no_save,
            frame_idx=args.frame_idx,
            use_brightest=args.use_brightest,
            skip_blank=args.skip_blank,
            method=args.method
        )
        
        if result is None:
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
