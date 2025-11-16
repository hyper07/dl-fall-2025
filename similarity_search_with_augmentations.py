#!/usr/bin/env python3
"""
Modified similarity search that shows all augmentations for the top similar images.
For each similar image found, displays all 6 augmentations (original, rotations, flips).
"""

import os
import sys
import json
import psycopg2
from pgvector.psycopg2 import register_vector

def get_similar_images_with_augmentations(query_image_id, top_k=10):
    """
    Find top-k similar images and return all their augmentations.
    """
    conn = psycopg2.connect(
        host='localhost',
        port='45432',
        database='db',
        user='admin',
        password='PassW0rd'
    )
    register_vector(conn)
    cursor = conn.cursor()

    # Get the query image details
    cursor.execute('SELECT embedding, original_image, label FROM images_features WHERE id = %s', (query_image_id,))
    query_result = cursor.fetchone()
    if not query_result:
        conn.close()
        return None

    query_embedding, query_original_image, query_label = query_result

    # Find top-k most similar images (excluding the query image itself)
    cursor.execute("""
        SELECT DISTINCT original_image, label,
               MIN(embedding <=> %s::vector) as min_distance,
               1 - MIN(embedding <=> %s::vector) as max_similarity
        FROM images_features
        WHERE original_image != %s
        GROUP BY original_image, label
        ORDER BY min_distance
        LIMIT %s
    """, (query_embedding, query_embedding, query_original_image, top_k))

    top_similar_results = cursor.fetchall()

    # For each similar image, get all its augmentations
    all_results = []
    for original_image, label, min_distance, max_similarity in top_similar_results:
        # For each similar image, get all its augmentations
        cursor.execute("""
            SELECT id, augmentation,
                   1 - (embedding <=> %s::vector) as similarity
            FROM images_features
            WHERE original_image = %s
            ORDER BY
                CASE augmentation
                    WHEN 'original' THEN 1
                    WHEN 'rotate_90' THEN 2
                    WHEN 'rotate_180' THEN 3
                    WHEN 'rotate_270' THEN 4
                    WHEN 'flip_horizontal' THEN 5
                    WHEN 'flip_vertical' THEN 6
                END
        """, (query_embedding, original_image))

        augmentations = cursor.fetchall()

        # Add to results
        for db_id, augmentation, similarity in augmentations:
            all_results.append({
                'original_image': original_image,
                'class': label,
                'augmentation': augmentation,
                'similarity_score': float(similarity) if similarity is not None else 0.0,
                'db_id': db_id
            })

    conn.close()
    return all_results

def main():
    # Selected images from each class
    selected_images = {
        'Abrasions': {'id': 1, 'filename': 'abrasions (1).jpg'},
        'Bruises': {'id': 3319, 'filename': 'bruises (16).jpg'},
        'Burns': {'id': 8941, 'filename': 'burns (33).jpg'},
        'Cut': {'id': 3889, 'filename': 'cut (13).jpg'},
        'Diabetic Wounds': {'id': 583, 'filename': '114_0.jpg'},
        'Laseration': {'id': 8479, 'filename': 'mirrored_laseration (19).jpg'},
        'Normal': {'id': 5647, 'filename': '1.jpg'},
        'Pressure Wounds': {'id': 6289, 'filename': '11_2.jpg'},
        'Surgical Wounds': {'id': 1975, 'filename': '113_0.jpg'},
        'Venous Wounds': {'id': 4165, 'filename': '100_0.jpg'}
    }

    # Results storage
    all_results = {}

    # Perform similarity search for each class
    for class_name, image_info in selected_images.items():
        print(f"\n{'='*80}")
        print(f"Similarity Search for {class_name} (with all augmentations)")
        print(f"Query Image: {image_info['filename']} (ID: {image_info['id']})")
        print(f"{'='*80}")

        # Get similar images with all their augmentations
        similar_results = get_similar_images_with_augmentations(image_info['id'], top_k=10)

        if not similar_results:
            print("Error: Could not find similar images")
            continue

        # Display results grouped by original image
        print(f"Top 10 similar images (showing all 6 augmentations each):")
        print(f"{'Rank':<4} {'Similarity':<10} {'Class':<15} {'Augmentation':<15} {'Original Image'}")
        print("-" * 85)

        class_results = []
        current_original = None
        rank = 0

        for i, result in enumerate(similar_results):
            # Start new rank when we encounter a new original image
            if result['original_image'] != current_original:
                current_original = result['original_image']
                rank += 1
                if rank > 10:  # Only show top 10
                    break

            print(f"{rank:<4} {result['similarity_score']:<10.4f} {result['class']:<15} {result['augmentation']:<15} {result['original_image']}")

            class_results.append({
                'rank': rank,
                'original_image': result['original_image'],
                'class': result['class'],
                'augmentation': result['augmentation'],
                'similarity_score': result['similarity_score'],
                'db_id': result['db_id']
            })

        all_results[class_name] = {
            'query_image': image_info,
            'similar_images': class_results
        }

    # Save results to JSON file
    output_file = "similarity_search_with_augmentations.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Similarity search with augmentations completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()