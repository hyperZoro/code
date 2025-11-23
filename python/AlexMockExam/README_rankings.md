# Student Ranking System

This system calculates comprehensive ranking results for students based on their exam scores, providing both age-adjusted and non-age-adjusted rankings for boys-only and total student populations.

## Features

### Ranking Types
1. **Boys Ranking**: Rankings calculated only among male students
2. **Total Ranking**: Rankings calculated among all students (male and female)

### Score Types
1. **Raw Scores (Non-Age-Adjusted)**: Original exam scores without age adjustments
2. **Age-Adjusted Scores**: Scores adjusted for student age differences

### Metrics Provided
For each ranking type and score type, the system provides:
- **Count Rank**: The student's position (e.g., 5th out of 100)
- **Percentage Rank**: The percentage of students they beat (e.g., 95% means they beat 95% of students)

### Subjects Ranked
- English
- Maths  
- Total (combined score)

## Usage

```python
from student_rank import calculate_rank

# Calculate rankings for a student
result = calculate_rank('data/2025_Test_A_hall-based_results_Excel.xlsx', 7003)

if result:
    # Raw scores (non-age-adjusted)
    print(f"English Rank (Raw): {result['English_Rank_Total_Raw']} out of {result['Total_Students']} students ({result['English_Percentile_Total_Raw']}%)")
    
    # Age-adjusted scores
    print(f"English Rank (Age-Adjusted): {result['English_Rank_Total_AgeAdj']} out of {result['Total_Students']} students ({result['English_Percentile_Total_AgeAdj']}%)")
```

## Output Format

The function returns a dictionary with the following keys:

### For Male Students:

#### Raw Scores (Non-Age-Adjusted):
- `English_Rank_Boys_Raw`: Rank among boys only
- `Maths_Rank_Boys_Raw`: Rank among boys only  
- `Total_Rank_Boys_Raw`: Rank among boys only
- `English_Percentile_Boys_Raw`: Percentage rank among boys
- `Maths_Percentile_Boys_Raw`: Percentage rank among boys
- `Total_Percentile_Boys_Raw`: Percentage rank among boys

#### Age-Adjusted Scores:
- `English_Rank_Boys_AgeAdj`: Rank among boys only
- `Maths_Rank_Boys_AgeAdj`: Rank among boys only  
- `Total_Rank_Boys_AgeAdj`: Rank among boys only
- `English_Percentile_Boys_AgeAdj`: Percentage rank among boys
- `Maths_Percentile_Boys_AgeAdj`: Percentage rank among boys
- `Total_Percentile_Boys_AgeAdj`: Percentage rank among boys

### For All Students:

#### Raw Scores (Non-Age-Adjusted):
- `English_Rank_Total_Raw`: Rank among all students
- `Maths_Rank_Total_Raw`: Rank among all students
- `Total_Rank_Total_Raw`: Rank among all students  
- `English_Percentile_Total_Raw`: Percentage rank among all students
- `Maths_Percentile_Total_Raw`: Percentage rank among all students
- `Total_Percentile_Total_Raw`: Percentage rank among all students

#### Age-Adjusted Scores:
- `English_Rank_Total_AgeAdj`: Rank among all students
- `Maths_Rank_Total_AgeAdj`: Rank among all students
- `Total_Rank_Total_AgeAdj`: Rank among all students  
- `English_Percentile_Total_AgeAdj`: Percentage rank among all students
- `Maths_Percentile_Total_AgeAdj`: Percentage rank among all students
- `Total_Percentile_Total_AgeAdj`: Percentage rank among all students

### Additional Information:
- `Total_Boys`: Total number of boys
- `Total_Students`: Total number of students

### For Female Students:
- Boys ranking fields will be `None`
- Only total ranking fields will have values

## Example Output

```
Ranking results for candidate 7003:

=== BOYS RANKING ===
--- RAW SCORES (Non-Age-Adjusted) ---
English Rank: 942 out of 1155 boys (18.53%)
Maths Rank: 1155 out of 1155 boys (0.09%)
Total Rank: 1138 out of 1155 boys (1.56%)

--- AGE-ADJUSTED SCORES ---
English Rank: 940 out of 1155 boys (18.7%)
Maths Rank: 1155 out of 1155 boys (0.09%)
Total Rank: 1139 out of 1155 boys (1.47%)

=== TOTAL RANKING ===
--- RAW SCORES (Non-Age-Adjusted) ---
English Rank: 1733 out of 2118 students (18.22%)
Maths Rank: 2118 out of 2118 students (0.05%)
Total Rank: 2076 out of 2118 students (2.03%)

--- AGE-ADJUSTED SCORES ---
English Rank: 1739 out of 2118 students (17.94%)
Maths Rank: 2118 out of 2118 students (0.05%)
Total Rank: 2079 out of 2118 students (1.89%)
```

## Data Requirements

The Excel file should have the following structure:
- Column 0: Candidate number
- Column 1: Gender (Male/Female)
- Column 2: English % (age-adjusted percentages)
- Column 3: Maths % (age-adjusted percentages)
- Column 4: English (raw scores)
- Column 5: Maths (raw scores)
- Column 6: Total (raw scores)
- Column 7: English.1 (age-adjusted scores)
- Column 8: Maths .1 (age-adjusted scores)
- Column 9: Total.1 (age-adjusted scores)

Students marked as 'absent' are automatically filtered out from the rankings.

## Age-Adjusted vs Raw Scores

- **Raw Scores**: Original exam scores without any age-based adjustments
- **Age-Adjusted Scores**: Scores that have been adjusted to account for age differences between students, providing a more fair comparison across different age groups

## Testing

Run the test script to see examples with different types of students:

```bash
python test_rankings.py
```
