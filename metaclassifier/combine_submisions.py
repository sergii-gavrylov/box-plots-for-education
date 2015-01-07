import sys
import json
import numpy as np
import pandas as pd


with open(sys.argv[1]) as f:
    conf = json.load(f)
submission_file_paths = conf['submission_file_paths']
combined_submission_file_path = conf['combined_submission_file_path']


if __name__ == '__main__':
    submissions = []
    for submission_file_path in submission_file_paths:
        submissions.append(pd.read_csv(submission_file_path, index_col=0, low_memory=False))
    combined_submission = submissions[0]
    for submission in submissions[1:]:
        combined_submission += submission
    combined_submission /= len(submissions)

    # combined_submission = np.log(submissions[0])
    # for submission in submissions[1:]:
    #     combined_submission += np.log(submission)
    # combined_submission /= len(submissions)
    # combined_submission = np.exp(combined_submission)
    # print np.sum(np.isinf(combined_submission.values))

    combined_submission.to_csv(combined_submission_file_path)