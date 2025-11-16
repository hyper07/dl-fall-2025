#!/usr/bin/env python3
"""
Script to perform similarity search for one image from each class.
Gets top 10 similar images for each selected image.
"""

import os
import sys
import json
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

def main():
    # Selected images from each class (from our database query)
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

    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        port='45432',
        database='db',
        user='admin',
        password='PassW0rd'
    )
    register_vector(conn)
    cursor = conn.cursor()

    # Results storage
    all_results = {}

    # Perform similarity search for each class
    for class_name, image_info in selected_images.items():
        print(f"\n{'='*60}")
        print(f"Similarity Search for {class_name}")
        print(f"Query Image: {image_info['filename']} (ID: {image_info['id']})")
        print(f"{'='*60}")

        # Get the embedding for this specific image
        cursor.execute('SELECT embedding FROM images_features WHERE id = %s', (image_info['id'],))
        result = cursor.fetchone()
        if not result:
            print(f"Error: Could not find embedding for image ID {image_info['id']}")
            continue

        query_embedding = result[0]

        # Perform similarity search using cosine distance
        cursor.execute("""
            SELECT id, content, 1 - (embedding <=> %s::vector) as similarity,
                   model_name, label, augmentation, original_image
            FROM images_features
            WHERE id != %s  -- Exclude the query image itself
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (query_embedding, image_info['id'], query_embedding))

        similar_results = cursor.fetchall()

        # Format and display results
        class_results = []
        print(f"Top 10 similar images:")
        print(f"{'Rank':<4} {'Similarity':<10} {'Class':<15} {'Augmentation':<15} {'Filename'}")
        print("-" * 80)

        for i, result in enumerate(similar_results, 1):
            db_id, content, similarity_score, model_name, label, augmentation, original_image = result
            print(f"{i:<4} {similarity_score:<10.4f} {label:<15} {augmentation:<15} {original_image}")

            class_results.append({
                'rank': i,
                'similarity_score': float(similarity_score),
                'class': label,
                'augmentation': augmentation,
                'filename': original_image,
                'db_id': db_id
            })

        all_results[class_name] = {
            'query_image': image_info,
            'similar_images': class_results
        }

    conn.close()

    # Save results to JSON file
    output_file = "similarity_search_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Similarity search completed!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()