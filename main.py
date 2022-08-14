from src.metrics.information_estimator import test_summaries


if __name__ == '__main__':
    print(test_summaries('./data/meansum_summaries_trimmed.json'))
    print(test_summaries('./data/pegasus_summaries.json'))