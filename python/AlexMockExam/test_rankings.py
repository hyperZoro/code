#!/usr/bin/env python3
"""
Test script to demonstrate the ranking functionality with different students.
Shows both age-adjusted and non-age-adjusted rankings.
"""

from student_rank import calculate_rank

def test_rankings():
    file_path = 'data/2025_Test_A_hall-based_results_Excel.xlsx'
    
    # Test cases
    test_cases = [
        (7003, "Male student"),
        (6136, "Female student"),
        (2036, "Another female student")
    ]
    
    for candidate_number, description in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {description} (Candidate {candidate_number})")
        print(f"{'='*60}")
        
        result = calculate_rank(file_path, candidate_number)
        
        if result:
            print(f"Ranking results for candidate {candidate_number}:")
            
            print("\n=== BOYS RANKING ===")
            if result['English_Rank_Boys_Raw'] is not None:
                print("--- RAW SCORES (Non-Age-Adjusted) ---")
                print(f"English Rank: {result['English_Rank_Boys_Raw']} out of {result['Total_Boys']} boys ({result['English_Percentile_Boys_Raw']}%)")
                print(f"Maths Rank: {result['Maths_Rank_Boys_Raw']} out of {result['Total_Boys']} boys ({result['Maths_Percentile_Boys_Raw']}%)")
                print(f"Total Rank: {result['Total_Rank_Boys_Raw']} out of {result['Total_Boys']} boys ({result['Total_Percentile_Boys_Raw']}%)")
                
                print("\n--- AGE-ADJUSTED SCORES ---")
                print(f"English Rank: {result['English_Rank_Boys_AgeAdj']} out of {result['Total_Boys']} boys ({result['English_Percentile_Boys_AgeAdj']}%)")
                print(f"Maths Rank: {result['Maths_Rank_Boys_AgeAdj']} out of {result['Total_Boys']} boys ({result['Maths_Percentile_Boys_AgeAdj']}%)")
                print(f"Total Rank: {result['Total_Rank_Boys_AgeAdj']} out of {result['Total_Boys']} boys ({result['Total_Percentile_Boys_AgeAdj']}%)")
            else:
                print("Student is not male - no boys ranking available")
            
            print("\n=== TOTAL RANKING ===")
            print("--- RAW SCORES (Non-Age-Adjusted) ---")
            print(f"English Rank: {result['English_Rank_Total_Raw']} out of {result['Total_Students']} students ({result['English_Percentile_Total_Raw']}%)")
            print(f"Maths Rank: {result['Maths_Rank_Total_Raw']} out of {result['Total_Students']} students ({result['Maths_Percentile_Total_Raw']}%)")
            print(f"Total Rank: {result['Total_Rank_Total_Raw']} out of {result['Total_Students']} students ({result['Total_Percentile_Total_Raw']}%)")
            
            print("\n--- AGE-ADJUSTED SCORES ---")
            print(f"English Rank: {result['English_Rank_Total_AgeAdj']} out of {result['Total_Students']} students ({result['English_Percentile_Total_AgeAdj']}%)")
            print(f"Maths Rank: {result['Maths_Rank_Total_AgeAdj']} out of {result['Total_Students']} students ({result['Maths_Percentile_Total_AgeAdj']}%)")
            print(f"Total Rank: {result['Total_Rank_Total_AgeAdj']} out of {result['Total_Students']} students ({result['Total_Percentile_Total_AgeAdj']}%)")
        else:
            print(f"Student with candidate number {candidate_number} not found or has no valid scores.")

if __name__ == "__main__":
    test_rankings()
