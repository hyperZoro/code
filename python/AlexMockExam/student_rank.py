import pandas as pd

def calculate_rank(file_path, candidate_number):
    """
    Reads an Excel file, calculates the rank of a student in Maths, English, and Total.
    Provides rankings for both age-adjusted and non-age-adjusted scores, for both boys-only and total students.

    Args:
        file_path (str): The path to the Excel file.
        candidate_number (int or str): The candidate number of the student to find the rank for.

    Returns:
        dict: A dictionary containing the student's rank in Maths, English, and Total for both age-adjusted and non-age-adjusted scores.
    """

    # Read the Excel file with header=1 to skip the first row
    df = pd.read_excel(file_path, header=1)
    
    # Filter out rows where students are absent and create a copy
    df_filtered = df[(df.iloc[:, 4] != 'absent') & (df.iloc[:, 5] != 'absent')].copy()
    
    # Convert candidate numbers to string for comparison
    df_filtered.iloc[:, 0] = df_filtered.iloc[:, 0].astype(str)
    candidate_number = str(candidate_number)
    
    # Separate boys and total students
    boys_df = df_filtered[df_filtered.iloc[:, 1] == 'Male'].copy()
    total_df = df_filtered.copy()
    
    # Calculate ranks for boys only - NON-AGE-ADJUSTED (raw scores)
    boys_df.loc[:, 'English_Rank_Boys_Raw'] = boys_df.iloc[:, 4].rank(ascending=False)
    boys_df.loc[:, 'Maths_Rank_Boys_Raw'] = boys_df.iloc[:, 5].rank(ascending=False)
    boys_df.loc[:, 'Total_Rank_Boys_Raw'] = boys_df.iloc[:, 6].rank(ascending=False)
    
    # Calculate ranks for boys only - AGE-ADJUSTED
    boys_df.loc[:, 'English_Rank_Boys_AgeAdj'] = boys_df.iloc[:, 7].rank(ascending=False)
    boys_df.loc[:, 'Maths_Rank_Boys_AgeAdj'] = boys_df.iloc[:, 8].rank(ascending=False)
    boys_df.loc[:, 'Total_Rank_Boys_AgeAdj'] = boys_df.iloc[:, 9].rank(ascending=False)
    
    # Calculate percentage ranks for boys - NON-AGE-ADJUSTED
    total_boys = len(boys_df)
    boys_df.loc[:, 'English_Percentile_Boys_Raw'] = ((total_boys - boys_df['English_Rank_Boys_Raw'] + 1) / total_boys * 100).round(2)
    boys_df.loc[:, 'Maths_Percentile_Boys_Raw'] = ((total_boys - boys_df['Maths_Rank_Boys_Raw'] + 1) / total_boys * 100).round(2)
    boys_df.loc[:, 'Total_Percentile_Boys_Raw'] = ((total_boys - boys_df['Total_Rank_Boys_Raw'] + 1) / total_boys * 100).round(2)
    
    # Calculate percentage ranks for boys - AGE-ADJUSTED
    boys_df.loc[:, 'English_Percentile_Boys_AgeAdj'] = ((total_boys - boys_df['English_Rank_Boys_AgeAdj'] + 1) / total_boys * 100).round(2)
    boys_df.loc[:, 'Maths_Percentile_Boys_AgeAdj'] = ((total_boys - boys_df['Maths_Rank_Boys_AgeAdj'] + 1) / total_boys * 100).round(2)
    boys_df.loc[:, 'Total_Percentile_Boys_AgeAdj'] = ((total_boys - boys_df['Total_Rank_Boys_AgeAdj'] + 1) / total_boys * 100).round(2)
    
    # Calculate ranks for total students - NON-AGE-ADJUSTED
    total_df.loc[:, 'English_Rank_Total_Raw'] = total_df.iloc[:, 4].rank(ascending=False)
    total_df.loc[:, 'Maths_Rank_Total_Raw'] = total_df.iloc[:, 5].rank(ascending=False)
    total_df.loc[:, 'Total_Rank_Total_Raw'] = total_df.iloc[:, 6].rank(ascending=False)
    
    # Calculate ranks for total students - AGE-ADJUSTED
    total_df.loc[:, 'English_Rank_Total_AgeAdj'] = total_df.iloc[:, 7].rank(ascending=False)
    total_df.loc[:, 'Maths_Rank_Total_AgeAdj'] = total_df.iloc[:, 8].rank(ascending=False)
    total_df.loc[:, 'Total_Rank_Total_AgeAdj'] = total_df.iloc[:, 9].rank(ascending=False)
    
    # Calculate percentage ranks for total students - NON-AGE-ADJUSTED
    total_students = len(total_df)
    total_df.loc[:, 'English_Percentile_Total_Raw'] = ((total_students - total_df['English_Rank_Total_Raw'] + 1) / total_students * 100).round(2)
    total_df.loc[:, 'Maths_Percentile_Total_Raw'] = ((total_students - total_df['Maths_Rank_Total_Raw'] + 1) / total_students * 100).round(2)
    total_df.loc[:, 'Total_Percentile_Total_Raw'] = ((total_students - total_df['Total_Rank_Total_Raw'] + 1) / total_students * 100).round(2)
    
    # Calculate percentage ranks for total students - AGE-ADJUSTED
    total_df.loc[:, 'English_Percentile_Total_AgeAdj'] = ((total_students - total_df['English_Rank_Total_AgeAdj'] + 1) / total_students * 100).round(2)
    total_df.loc[:, 'Maths_Percentile_Total_AgeAdj'] = ((total_students - total_df['Maths_Rank_Total_AgeAdj'] + 1) / total_students * 100).round(2)
    total_df.loc[:, 'Total_Percentile_Total_AgeAdj'] = ((total_students - total_df['Total_Rank_Total_AgeAdj'] + 1) / total_students * 100).round(2)

    # Find the student's data in both datasets
    student_boys_data = boys_df[boys_df.iloc[:, 0] == candidate_number]
    student_total_data = total_df[total_df.iloc[:, 0] == candidate_number]

    if not student_total_data.empty:
        result = {}
        
        # Boys ranking (only if student is male)
        if not student_boys_data.empty:
            result.update({
                # Raw scores (non-age-adjusted)
                'English_Rank_Boys_Raw': int(student_boys_data['English_Rank_Boys_Raw'].iloc[0]),
                'Maths_Rank_Boys_Raw': int(student_boys_data['Maths_Rank_Boys_Raw'].iloc[0]),
                'Total_Rank_Boys_Raw': int(student_boys_data['Total_Rank_Boys_Raw'].iloc[0]),
                'English_Percentile_Boys_Raw': float(student_boys_data['English_Percentile_Boys_Raw'].iloc[0]),
                'Maths_Percentile_Boys_Raw': float(student_boys_data['Maths_Percentile_Boys_Raw'].iloc[0]),
                'Total_Percentile_Boys_Raw': float(student_boys_data['Total_Percentile_Boys_Raw'].iloc[0]),
                
                # Age-adjusted scores
                'English_Rank_Boys_AgeAdj': int(student_boys_data['English_Rank_Boys_AgeAdj'].iloc[0]),
                'Maths_Rank_Boys_AgeAdj': int(student_boys_data['Maths_Rank_Boys_AgeAdj'].iloc[0]),
                'Total_Rank_Boys_AgeAdj': int(student_boys_data['Total_Rank_Boys_AgeAdj'].iloc[0]),
                'English_Percentile_Boys_AgeAdj': float(student_boys_data['English_Percentile_Boys_AgeAdj'].iloc[0]),
                'Maths_Percentile_Boys_AgeAdj': float(student_boys_data['Maths_Percentile_Boys_AgeAdj'].iloc[0]),
                'Total_Percentile_Boys_AgeAdj': float(student_boys_data['Total_Percentile_Boys_AgeAdj'].iloc[0]),
                
                'Total_Boys': int(total_boys)
            })
        else:
            result.update({
                # Raw scores (non-age-adjusted)
                'English_Rank_Boys_Raw': None,
                'Maths_Rank_Boys_Raw': None,
                'Total_Rank_Boys_Raw': None,
                'English_Percentile_Boys_Raw': None,
                'Maths_Percentile_Boys_Raw': None,
                'Total_Percentile_Boys_Raw': None,
                
                # Age-adjusted scores
                'English_Rank_Boys_AgeAdj': None,
                'Maths_Rank_Boys_AgeAdj': None,
                'Total_Rank_Boys_AgeAdj': None,
                'English_Percentile_Boys_AgeAdj': None,
                'Maths_Percentile_Boys_AgeAdj': None,
                'Total_Percentile_Boys_AgeAdj': None,
                
                'Total_Boys': int(total_boys)
            })
        
        # Total ranking
        result.update({
            # Raw scores (non-age-adjusted)
            'English_Rank_Total_Raw': int(student_total_data['English_Rank_Total_Raw'].iloc[0]),
            'Maths_Rank_Total_Raw': int(student_total_data['Maths_Rank_Total_Raw'].iloc[0]),
            'Total_Rank_Total_Raw': int(student_total_data['Total_Rank_Total_Raw'].iloc[0]),
            'English_Percentile_Total_Raw': float(student_total_data['English_Percentile_Total_Raw'].iloc[0]),
            'Maths_Percentile_Total_Raw': float(student_total_data['Maths_Percentile_Total_Raw'].iloc[0]),
            'Total_Percentile_Total_Raw': float(student_total_data['Total_Percentile_Total_Raw'].iloc[0]),
            
            # Age-adjusted scores
            'English_Rank_Total_AgeAdj': int(student_total_data['English_Rank_Total_AgeAdj'].iloc[0]),
            'Maths_Rank_Total_AgeAdj': int(student_total_data['Maths_Rank_Total_AgeAdj'].iloc[0]),
            'Total_Rank_Total_AgeAdj': int(student_total_data['Total_Rank_Total_AgeAdj'].iloc[0]),
            'English_Percentile_Total_AgeAdj': float(student_total_data['English_Percentile_Total_AgeAdj'].iloc[0]),
            'Maths_Percentile_Total_AgeAdj': float(student_total_data['Maths_Percentile_Total_AgeAdj'].iloc[0]),
            'Total_Percentile_Total_AgeAdj': float(student_total_data['Total_Percentile_Total_AgeAdj'].iloc[0]),
            
            'Total_Students': int(total_students)
        })

        return result
    else:
        return None

# Example usage
file_path = 'data/2025_Test_A_hall-based_results_Excel.xlsx'
candidate_number = 7003  # Replace with the actual candidate number
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
    print(f"Student with candidate number {candidate_number} not found in the file.")
