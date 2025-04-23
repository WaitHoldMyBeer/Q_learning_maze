import pandas as pd

def main():
    num = "017"
    rat_data = pd.read_csv(f"modules/{num}.csv")
    updated_rat_data = rat_data.copy()

    print("Processing trial switch logic...")

    # Initialize Start column
    updated_rat_data['Start'] = None  # Use nullable integer type
    updated_rat_data.loc[0, 'Start'] = updated_rat_data.loc[0, 'State']

    # Track upcoming positive trial starts
    for i in range(1, len(updated_rat_data)):
        prev_trial = updated_rat_data.loc[i - 1, 'Trial']
        curr_trial = updated_rat_data.loc[i, 'Trial']

        # Positive to negative → update the previous row
        if prev_trial > 0 and curr_trial < 0:
            # Find next positive trial and grab its state
            for j in range(i, len(updated_rat_data)):
                if updated_rat_data.loc[j, 'Trial'] > 0:
                    next_start = updated_rat_data.loc[j, 'State']
                    updated_rat_data.loc[i - 1, 'Start'] = next_start
                    break

        # Negative to positive → treat it as a new start
        elif prev_trial < 0 and curr_trial > 0:
            updated_rat_data.loc[i, 'Start'] = updated_rat_data.loc[i, 'State']

        # Regular continuation within a positive trial → inherit previous
        elif curr_trial > 0 and updated_rat_data.loc[i, 'Start'] is None:
            updated_rat_data.loc[i, 'Start'] = updated_rat_data.loc[i - 1, 'Start']

    # Fill down any remaining missing Start values
    updated_rat_data['Start'].fillna(method='ffill', inplace=True)

    # Ensure 'Start' column contains integers
    updated_rat_data['Start'] = updated_rat_data['Start'].astype(int)

    # Save result
    updated_rat_data.to_csv(f"modules/{num}.csv", index=False)
    print("Update complete.")

if __name__ == "__main__":
    main()